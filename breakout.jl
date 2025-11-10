using GLMakie, GeometryBasics
using Random, Printf

const W = 1440f0
const H = 900f0
const FIXED_DT = 1 / 300 # fixed physics step

# --- Theme colors (Nord‑inspired) ---
const COLOR_BG     = RGBAf(0.18, 0.20, 0.25, 1.0)   # #2E3440
const COLOR_TEXT   = RGBAf(0.90, 0.93, 0.97, 1.0)   # #E5E9F0
const COLOR_PADDLE = RGBAf(0.53, 0.75, 0.82, 1.0)   # #88C0D0
const COLOR_BALL   = RGBAf(0.93, 0.94, 0.96, 1.0)   # #ECEFF4
const COLOR_BRICK1 = RGBAf(0.51, 0.63, 0.76, 1.0)   # #81A1C1
const COLOR_BRICK2 = RGBAf(0.75, 0.38, 0.42, 1.0)   # #BF616A
const COLOR_BRICK3 = RGBAf(0.92, 0.80, 0.55, 1.0)   # #EBCB8B
const COLOR_STROKE = RGBAf(0.0, 0.0, 0.0, 0.45)
const COLOR_HUDBG  = RGBAf(0.0, 0.0, 0.0, 0.35)

# Power‑up visuals
const COLOR_PU_PIERCE   = RGBAf(0.98, 0.74, 0.26, 1.0)  # amber
const COLOR_PU_MULTIBAL = RGBAf(0.45, 0.80, 0.75, 1.0)  # teal
const COLOR_PU_LASER    = RGBAf(0.94, 0.48, 0.35, 1.0)  # orange‑red
const COLOR_LASER_SHOT  = RGBAf(0.99, 0.86, 0.40, 1.0)

const TRAIL_LEN    = 12
const FLASH_MAX    = 0.18f0
const COMBO_WINDOW = 3.0f0

# Power‑up tuning
const DROP_P      = 0.16f0                   # drop probability per destroyed brick
const DROP_BUDGET = 10                        # max drops per level
const PIERCE_DUR  = 6.0f0
const LASER_DUR   = 10.0f0
const SHOT_COOLD  = 0.7f0                     # ~1.4 shots/s while active
const SHOT_SPEED  = 700f0
const SHOT_TTL    = 1.2f0
const CHILD_TTL   = Inf32            # child balls do not expire by time
const MULTI_COUNT = 2                          # spawn +2 child balls

mutable struct FXState
    ppos::Vector{Point2f}
    pvel::Vector{Vec2f}
    pttl::Vector{Float32}
    ptmax::Vector{Float32}
    pcol::Vector{RGBAf}
    bflash::Vector{Float32}
end

# Minimal power‑up runtime state (kept small and cache‑friendly)
mutable struct PUState
    # falling items
    item_pos::Vector{Point2f}
    item_vel::Vector{Vec2f}
    item_typ::Vector{Symbol}
    # child balls
    ball_pos::Vector{Point2f}
    ball_vel::Vector{Vec2f}
    ball_ttl::Vector{Float32}
    # laser shots
    shot_pos::Vector{Point2f}
    shot_vel::Vector{Vec2f}
    shot_ttl::Vector{Float32}
    # timers
    pierce_ttl::Float32
    laser_ttl::Float32
    next_shot::Float32
    # per‑level drop counter
    drops_left::Int
end

# ---- Utils ----
const P2 = Point2f
const V2 = Vec2f
rectf(x::Real, y::Real, w::Real, h::Real) = GeometryBasics.Rect(P2(Float32(x), Float32(y)), V2(Float32(w), Float32(h)))

function rect_bounds(r::GeometryBasics.Rect{2,T}) where {T}
    o = r.origin
    s = r.widths
    x1 = Float32(o[1])
    y1 = Float32(o[2])
    x2 = x1 + Float32(s[1])
    y2 = y1 + Float32(s[2])
    return x1, y1, x2, y2
end

hp_color(h::Int) = h ≤ 1 ? COLOR_BRICK1 : (h == 2 ? COLOR_BRICK2 : COLOR_BRICK3)

@inline function mix_rgba(a::RGBAf, b::RGBAf, t::Float32)
    tt = clamp(t, 0f0, 1f0)
    return RGBAf(
        a.r + (b.r - a.r) * tt,
        a.g + (b.g - a.g) * tt,
        a.b + (b.b - a.b) * tt,
        a.alpha + (b.alpha - a.alpha) * tt,
    )
end

# Procedural level generator (level 1 fixed pattern; level >=2 random)
function make_level_data_rand(level::Int)
    # grid sizing
    cols, rows = 12, 7
    margin = 36f0
    gutter = 12f0
    scale = 1.35f0

    totalw = W - 2f0 * margin
    bw = (totalw - (cols - 1) * gutter) / cols
    # make brick field occupy a fixed fraction of screen height (50%)
    target_frac = 0.50f0
    area_h      = H * target_frac
    bh          = (area_h - (Float32(rows) - 1f0) * gutter) / Float32(rows)

    hud_gap = 48f0
    y_top_limit = H - hud_gap
    y0 = y_top_limit - bh - (rows - 1f0) * (bh + gutter)

    bricks = GeometryBasics.Rect{2,Float32}[]
    hps    = Int[]
    colors = RGBAf[]

    c  = (cols + 1) / 2
    rm = (rows + 1) / 2

    # pick one pattern per level with random params
    pat = rand(1:6)
    off = rand(0:1)
    period = rand(3:4)
    width = rand(1:2)
    amp = rand(1.0:1.0:2.0)  # amplitude in rows
    freq = rand(1:2)
    phase = rand() * 2f0 * Float32(pi)
    thick = rand(0:1) + 1  # 1..2
    rad = rand(2:3)
    ncent = rand(2:3)
    centers = [(rand(1:cols), rand(2:rows)) for _ = 1:ncent]

    placed = 0

    for j = 1:rows, i = 1:cols
        x = margin + (i - 1f0) * (bw + gutter)
        y = y0 + (j - 1f0) * (bh + gutter)

        keep = false
        hp   = 1

        if level <= 1
            # Original level-1 pattern
            keep = isodd(i + j)
            hp = 1
        elseif pat == 1
            # Checkerboard with offset and a filled band near center
            keep = isodd(i + j + off) || (abs(i - c) <= 1 && isodd(j + off))
            hp   = j <= rm ? 1 : 2
        elseif pat == 2
            # Vertical bars with adjustable period/width
            keep = ((i + off) % period) < width
            hp   = j >= rows - 2 ? 2 : 1
        elseif pat == 3
            # Sine wave ribbon(s)
            yline = rm + amp * sin(phase + (2f0 * Float32(pi) * freq) * ((i - 1f0) / cols))
            keep = abs(j - yline) <= thick
            hp = abs(j - rm) <= 1 ? 2 : 1
        elseif pat == 4
            # Diamond / rhombus fill
            keep = (abs(i - c) + abs(j - rm)) <= (rad + (isodd(i + j) ? 1 : 0))
            hp   = j < rm ? 1 : (j > rm ? 3 : 2)
        elseif pat == 5
            # Border + diagonals
            border = (i == 1 || i == cols || j == 1 || j == rows)
            diag   = (abs(i - j) <= 1) || (abs((cols - i + 1) - j) <= 1)
            keep   = border || (diag && isodd(i + j + off))
            hp     = border ? 2 : 1
        else
            # Clustered blobs around random centers
            keep_any = false
            for (cx, cy) in centers
                if (abs(i - cx) + abs(j - cy)) <= rad
                    keep_any = true
                    break
                end
            end
            keep = keep_any && (rand() < 0.85)
            hp   = (abs(j - rm) <= 1) ? 2 : 1
        end

        # Light random holes to reduce monotony
        if keep && rand() < 0.06
            keep = false
        end

        if keep
            push!(bricks, rectf(x, y, bw, bh))
            push!(hps, hp)
            push!(colors, hp_color(hp))
            placed += 1
        end
    end

    # Ensure a minimum density for playability; fallback to dense stripes
    if placed < 14
        empty!(bricks)
        empty!(hps)
        empty!(colors)
        for j = 1:rows, i = 1:cols
            x    = margin + (i - 1f0) * (bw + gutter)
            y    = y0 + (j - 1f0) * (bh + gutter)
            keep = ((i + off) % 3) < 2
            hp   = j >= rows - 2 ? 2 : 1
            if keep
                push!(bricks, rectf(x, y, bw, bh))
                push!(hps, hp)
                push!(colors, hp_color(hp))
            end
        end
    end

    return bricks, hps, colors
end

function circle_rect_overlap(c::Point2f, r::Float32, rect::GeometryBasics.Rect{2,Float32})
    x1, y1, x2, y2 = rect_bounds(rect)
    nx = clamp(c[1], x1, x2)
    ny = clamp(c[2], y1, y2)
    dx = c[1] - nx
    dy = c[2] - ny
    return dx * dx + dy * dy ≤ r * r
end

# rough collision normal
function collision_normal(p_prev::Point2f, p_new::Point2f, r::Float32, rect::GeometryBasics.Rect{2,Float32})
    x1, y1, x2, y2 = rect_bounds(rect)
    left = (p_new[1] + r) - x1
    right = x2 - (p_new[1] - r)
    bottom = (p_new[2] + r) - y1
    top = y2 - (p_new[2] - r)
    mins = [(left, 1), (right, 2), (bottom, 3), (top, 4)]
    _, idx = findmin(first.(mins))
    sel = mins[idx][2]
    if sel == 1 || sel == 2
        return (p_prev[1] ≤ x1) ? V2(-1f0, 0f0) : (p_prev[1] ≥ x2 ? V2(1f0, 0f0) : (left < right ? V2(-1f0, 0f0) : V2(1f0, 0f0)))
    else
        return (p_prev[2] ≤ y1) ? V2(0f0, -1f0) : (p_prev[2] ≥ y2 ? V2(0f0, 1f0) : (bottom < top ? V2(0f0, -1f0) : V2(0f0, 1f0)))
    end
end

# NOTE: include paddle_h to avoid UndefVarError and place ball correctly
function reset_ball!(
    ball_center::Observable{Point2f}, ball_vel::Base.RefValue{Vec2f}, launched::Observable{Bool},
    paddle_x::Float32, paddle_w::Float32, paddle_y::Float32, paddle_h::Float32,
    r::Float32,
    trail_pts::Observable{Vector{Point2f}}, trail_cols::Observable{Vector{RGBAf}},
)
    ball_center[] = P2(paddle_x + paddle_w / 2, paddle_y + paddle_h + r + 1f0)
    # scale initial speed with screen width baseline 800
    s            = W / 800f0
    ball_vel[]   = V2(176f0 * s, 264f0 * s)     # 1.1x initial speed, scaled
    launched[]   = false
    trail_pts[]  = Point2f[]
    trail_cols[] = RGBAf[]
end

# --- Power‑up helpers ---
@inline function pu_color(sym::Symbol)
    sym === :pierce && return COLOR_PU_PIERCE
    sym === :multiball && return COLOR_PU_MULTIBAL
    sym === :laser && return COLOR_PU_LASER
    return COLOR_TEXT
end

# weighted choice among implemented types
function sample_powerup()
    types   = (:pierce, :multiball, :laser)
    weights = (0.4, 0.35, 0.25)  # tweakable
    u       = rand() * sum(weights)
    acc     = 0.0
    for (t, w) in zip(types, weights)
        acc += w
        if u ≤ acc
            return t
        end
    end
    return :pierce
end

# spawn a falling item at cpos
function maybe_drop!(pu::PUState, cpos::Point2f)
    if pu.drops_left ≤ 0
        return
    end
    if rand(Float32) ≤ DROP_P
        typ = sample_powerup()
        push!(pu.item_pos, cpos)
        push!(pu.item_vel, V2(0f0, -40f0))   # start slow, gravity will pull
        push!(pu.item_typ, typ)
        pu.drops_left -= 1
    end
end

# activate effects on pickup
function activate!(pu::PUState, typ::Symbol, main_pos::Point2f, main_vel::Vec2f)
    if typ === :pierce
        pu.pierce_ttl = max(pu.pierce_ttl, PIERCE_DUR)
    elseif typ === :laser
        pu.laser_ttl = max(pu.laser_ttl, LASER_DUR)
        pu.next_shot = min(pu.next_shot, 0f0)  # allow immediate firing
    elseif typ === :multiball
        # spawn +2 child balls by rotating main vel slightly
        speed = sqrt(main_vel[1]^2 + main_vel[2]^2)
        for θ in (-0.25f0, 0.25f0)  # ~±14°
            c = cos(θ)
            s = sin(θ)
            v = V2(main_vel[1] * c - main_vel[2] * s, main_vel[1] * s + main_vel[2] * c)
            v *= (speed / sqrt(v[1]^2 + v[2]^2))
            push!(pu.ball_pos, main_pos)
            push!(pu.ball_vel, v)
            push!(pu.ball_ttl, CHILD_TTL)
        end
    end
end

# apply slight random deflection while piercing to avoid straight tunnels
@inline function pierce_jitter(v::Vec2f)
    θ = (rand(Float32) - 0.5f0) * 0.08f0   # ±~4.6°
    c = cos(θ)
    s = sin(θ)
    vv = V2(v[1] * c - v[2] * s, v[1] * s + v[2] * c)
    sp = sqrt(v[1]^2 + v[2]^2)
    return vv * (sp / sqrt(vv[1]^2 + vv[2]^2))
end

# common brick hit handling (returns score delta and whether brick destroyed)
function apply_brick_hit!(i::Int, bricks::Vector{GeometryBasics.Rect{2,Float32}}, hps::Vector{Int}, base_cols::Vector{RGBAf}, fx::FXState, score_mult::Float32)
    rect = bricks[i]
    x1, y1, x2, y2 = rect_bounds(rect)
    cpos = P2((x1 + x2) / 2, (y1 + y2) / 2)
    col_pre = base_cols[i]
    hps[i] -= 1
    if hps[i] ≤ 0
        # break particles
        local sw = W / 800f0
        for _ = 1:28
            θ = 2f0 * Float32(pi) * rand(Float32)
            s = (rand(Float32) * 160f0 + 120f0) * sw
            push!(fx.ppos, cpos)
            push!(fx.pvel, V2(cos(θ) * s, sin(θ) * s))
            t = rand(Float32) * 0.35f0 + 0.45f0
            push!(fx.pttl, t)
            push!(fx.ptmax, t)
            push!(fx.pcol, col_pre)
        end
        deleteat!(bricks, i)
        deleteat!(hps, i)
        deleteat!(base_cols, i)
        deleteat!(fx.bflash, i)
        return Int(round(100 * score_mult)), true, cpos
    else
        fx.bflash[i] = FLASH_MAX
        base_cols[i] = hp_color(hps[i])
        local sw = W / 800f0
        for _ = 1:12
            θ = 2f0 * Float32(pi) * rand(Float32)
            s = (rand(Float32) * 120f0 + 90f0) * sw
            push!(fx.ppos, cpos)
            push!(fx.pvel, V2(cos(θ) * s, sin(θ) * s))
            t = rand(Float32) * 0.25f0 + 0.25f0
            push!(fx.pttl, t)
            push!(fx.ptmax, t)
            push!(fx.pcol, col_pre)
        end
        return Int(round(50 * score_mult)), false, cpos
    end
end

function update!(scene,
    dt::Float64,
    ball_center::Observable{Point2f}, ball_vel_ref::Base.RefValue{Vec2f}, r::Float32,
    paddle_x::Observable{Float32}, paddle_y::Float32, paddle_w::Float32, paddle_h::Float32, paddle_speed::Float32,
    launched::Observable{Bool},
    bricks_obs::Observable{Vector{GeometryBasics.Rect{2,Float32}}},
    brick_hp_obs::Observable{Vector{Int}},
    brick_color_obs::Observable{Vector{RGBAf}},
    score::Observable{Int}, lives::Observable{Int}, trail_pts::Observable{Vector{Point2f}}, trail_cols::Observable{Vector{RGBAf}},
    part_pos_obs::Observable{Vector{Point2f}}, part_col_obs::Observable{Vector{RGBAf}}, fx::FXState,
    brick_base_colors_obs::Observable{Vector{RGBAf}}, combo_count::Observable{Int}, combo_timer::Observable{Float32}, combo_text::Observable{String}, combo_color::Observable{RGBAf}, level::Observable{Int},
    # power‑up runtime & visuals
    pu::PUState, items_pos_obs::Observable{Vector{Point2f}}, items_col_obs::Observable{Vector{RGBAf}},
    shots_pos_obs::Observable{Vector{Point2f}}, extras_pos_obs::Observable{Vector{Point2f}}, ball_colors_obs::Observable{Vector{RGBAf}}, pierce_txt::Observable{String}, laser_txt::Observable{String})

    # input → paddle
    vx = 0f0
    ispressed(scene, Keyboard.left) && (vx -= paddle_speed)
    ispressed(scene, Keyboard.a) && (vx -= paddle_speed)
    ispressed(scene, Keyboard.right) && (vx += paddle_speed)
    ispressed(scene, Keyboard.d) && (vx += paddle_speed)

    x = paddle_x[] + vx * Float32(dt)
    x = clamp(x, 0f0, W - paddle_w)
    paddle_x[] = x

    # stick to paddle before launch
    if !launched[]
        ball_center[] = P2(x + paddle_w / 2, paddle_y + paddle_h + r + 1f0)
    end

    # fire lasers if active
    if pu.laser_ttl > 0f0
        pu.next_shot -= Float32(dt)
        if pu.next_shot ≤ 0f0
            # spawn two shots from paddle edges
            off = paddle_w * 0.35f0
            y0  = paddle_y + paddle_h + 2f0
            push!(pu.shot_pos, P2(paddle_x[] + off, y0))
            push!(pu.shot_vel, V2(0f0, SHOT_SPEED))
            push!(pu.shot_ttl, SHOT_TTL)
            push!(pu.shot_pos, P2(paddle_x[] + paddle_w - off, y0))
            push!(pu.shot_vel, V2(0f0, SHOT_SPEED))
            push!(pu.shot_ttl, SHOT_TTL)
            pu.next_shot = SHOT_COOLD
        end
    end

    # helper to advance one ball (returns new pos, vel)
    function step_ball(p::Point2f, v::Vec2f)
        p_new = P2(p[1] + v[1] * Float32(dt), p[2] + v[2] * Float32(dt))
        # walls
        if p_new[1] - r < 0f0
            p_new = P2(r, p_new[2])
            v = V2(abs(v[1]), v[2])
        elseif p_new[1] + r > W
            p_new = P2(W - r, p_new[2])
            v = V2(-abs(v[1]), v[2])
        end
        if p_new[2] + r > H
            p_new = P2(p_new[1], H - r)
            v = V2(v[1], -abs(v[2]))
        end
        # paddle
        padd = rectf(paddle_x[], paddle_y, paddle_w, paddle_h)
        if circle_rect_overlap(p_new, r, padd)
            cx = p_new[1]
            px = paddle_x[] + paddle_w / 2
            rel = clamp((cx - px) / (paddle_w / 2), -1f0, 1f0)
            speed = sqrt(v[1]^2 + v[2]^2)
            min_deg = 25f0
            max_deg = 70f0
            angle = deg2rad(min_deg + (max_deg - min_deg) * abs(rel))
            sx = (rel == 0f0 ? sign(v[1] == 0f0 ? 1f0 : v[1]) : sign(rel))
            dirx = sx * sin(angle)
            diry = cos(angle)
            if diry < 0.2f0
                diry = 0.2f0
            end
            dnorm = sqrt(dirx^2 + diry^2)
            dirx /= dnorm
            diry /= dnorm
            v = V2(dirx * speed, diry * speed)
            v = V2(v[1], abs(v[2]))
            p_new = P2(p_new[1], paddle_y + paddle_h + r + 0.1f0)
            # reset combo on any paddle contact
            combo_count[] = 0
            combo_timer[] = 0f0
            combo_text[]  = ""
            combo_color[] = RGBAf(COLOR_TEXT.r, COLOR_TEXT.g, COLOR_TEXT.b, 0f0)
        end
        # bricks
        bricks = bricks_obs[]
        hps = brick_hp_obs[]
        base_cols = brick_base_colors_obs[]
        hit_idx = 0
        nrm = V2(0f0, 0f0)
        for (i, rect) in pairs(bricks)
            if circle_rect_overlap(p_new, r, rect)
                nrm = collision_normal(p, p_new, r, rect)
                hit_idx = i
                break
            end
        end
        if hit_idx != 0
            # combo HUD update
            combo_count[] += 1
            combo_timer[] = COMBO_WINDOW
            mult          = 1f0 + 0.1f0 * clamp(combo_count[] - 1, 0, 10)
            combo_text[]  = "COMBO x$(combo_count[])  ×$(round(mult; digits=1))"
            combo_color[] = RGBAf(COLOR_TEXT.r, COLOR_TEXT.g, COLOR_TEXT.b, 1f0)

            # apply damage & particles
            add, destroyed, cpos = apply_brick_hit!(hit_idx, bricks, hps, base_cols, fx, Float32(mult))
            score[] += add
            if destroyed
                maybe_drop!(pu, cpos)
            end
            # reflect or pierce
            if pu.pierce_ttl > 0f0
                v = pierce_jitter(v)  # no reflection; slight deflection
            else
                if nrm[1] != 0
                    v = V2(-v[1], v[2])
                end
                if nrm[2] != 0
                    v = V2(v[1], -v[2])
                end
                p_new = P2(p_new[1] + 0.5f0 * nrm[1], p_new[2] + 0.5f0 * nrm[2])
            end
            bricks_obs[] = bricks
            brick_hp_obs[] = hps
            brick_base_colors_obs[] = base_cols
        end
        return p_new, v
    end

    # main ball life‑loss check (only for the main ball)
    if launched[]
        p = ball_center[]
        v = ball_vel_ref[]
        p, v = step_ball(p, v)
        # if main ball dropped but extras exist, promote one extra to main
        if p[2] - r < 0f0 && !isempty(pu.ball_pos)
            idx = length(pu.ball_pos)
            p = pu.ball_pos[idx]
            v = pu.ball_vel[idx]
            deleteat!(pu.ball_pos, idx)
            deleteat!(pu.ball_vel, idx)
            deleteat!(pu.ball_ttl, idx)
            trail_pts[] = Point2f[]
            trail_cols[] = RGBAf[]
        end
        if p[2] - r < 0f0
            # life lost
            lives[] -= 1
            if lives[] ≤ 0
                lives[] = 3
                score[] = 0
                level[] = 1
                b, hps, cols = make_level_data_rand(level[])
                bricks_obs[] = b
                brick_hp_obs[] = hps
                brick_color_obs[] = cols
                brick_base_colors_obs[] = copy(cols)
                fx.bflash = fill(0f0, length(b))
                pu.drops_left = DROP_BUDGET
                pu.item_pos = Point2f[]
                pu.item_vel = Vec2f[]
                pu.item_typ = Symbol[]
                pu.ball_pos = Point2f[]
                pu.ball_vel = Vec2f[]
                pu.ball_ttl = Float32[]
                pu.shot_pos = Point2f[]
                pu.shot_vel = Vec2f[]
                pu.shot_ttl = Float32[]
                pu.pierce_ttl = 0f0
                pu.laser_ttl = 0f0
                pu.next_shot = 0f0
            end
            combo_count[] = 0
            combo_timer[] = 0f0
            combo_text[] = ""
            combo_color[] = RGBAf(COLOR_TEXT.r, COLOR_TEXT.g, COLOR_TEXT.b, 0f0)
            # clear all power-up side effects upon life loss
            pu.item_pos = Point2f[]
            pu.item_vel = Vec2f[]
            pu.item_typ = Symbol[]
            pu.ball_pos = Point2f[]
            pu.ball_vel = Vec2f[]
            pu.ball_ttl = Float32[]
            pu.shot_pos = Point2f[]
            pu.shot_vel = Vec2f[]
            pu.shot_ttl = Float32[]
            pu.pierce_ttl = 0f0
            pu.laser_ttl = 0f0
            pu.next_shot = 0f0
            items_pos_obs[] = Point2f[]
            items_col_obs[] = RGBAf[]
            shots_pos_obs[] = Point2f[]
            extras_pos_obs[] = Point2f[]
            reset_ball!(ball_center, ball_vel_ref, launched, paddle_x[], paddle_w, paddle_y, paddle_h, r, trail_pts, trail_cols)
        else
            # clamp speed
            maxspeed = 600f0
            s = sqrt(v[1]^2 + v[2]^2)
            if s > maxspeed
                v *= (maxspeed / s)
            end
            ball_center[]  = p
            ball_vel_ref[] = v
        end
    end

    # update child balls (no life loss on drop)
    i = length(pu.ball_pos)
    while i ≥ 1
        pu.ball_ttl[i] -= Float32(dt)
        if pu.ball_ttl[i] ≤ 0f0
            deleteat!(pu.ball_pos, i)
            deleteat!(pu.ball_vel, i)
            deleteat!(pu.ball_ttl, i)
        else
            p, v = step_ball(pu.ball_pos[i], pu.ball_vel[i])
            if p[2] - r < 0f0
                deleteat!(pu.ball_pos, i)
                deleteat!(pu.ball_vel, i)
                deleteat!(pu.ball_ttl, i)
            else
                # clamp and store
                maxspeed = 600f0
                s = sqrt(v[1]^2 + v[2]^2)
                if s > maxspeed
                    v *= (maxspeed / s)
                end
                pu.ball_pos[i] = p
                pu.ball_vel[i] = v
            end
        end
        i -= 1
    end

    # update laser shots
    i = length(pu.shot_pos)
    while i ≥ 1
        pu.shot_ttl[i] -= Float32(dt)
        if pu.shot_ttl[i] ≤ 0f0
            deleteat!(pu.shot_pos, i)
            deleteat!(pu.shot_vel, i)
            deleteat!(pu.shot_ttl, i)
        else
            p = pu.shot_pos[i]
            v = pu.shot_vel[i]
            p = P2(p[1] + v[1] * Float32(dt), p[2] + v[2] * Float32(dt))
            hit = false
            bricks = bricks_obs[]
            hps = brick_hp_obs[]
            base_cols = brick_base_colors_obs[]
            for (bi, rect) in pairs(bricks)
                shot_hit_r = 3f0 * (W / 800f0)
                if circle_rect_overlap(p, shot_hit_r, rect)   # small radius hit (scaled)
                    add, destroyed, cpos = apply_brick_hit!(bi, bricks, hps, base_cols, fx, 1f0)
                    score[] += add
                    hit = true
                    if destroyed
                        maybe_drop!(pu, cpos)
                    end
                    bricks_obs[] = bricks
                    brick_hp_obs[] = hps
                    brick_base_colors_obs[] = base_cols
                    break
                end
            end
            if hit || p[2] > H
                deleteat!(pu.shot_pos, i)
                deleteat!(pu.shot_vel, i)
                deleteat!(pu.shot_ttl, i)
            else
                pu.shot_pos[i] = p
            end
        end
        i -= 1
    end

    # update falling items (gravity + pickup)
    i = length(pu.item_pos)
    padd = rectf(paddle_x[], paddle_y, paddle_w, paddle_h)
    while i ≥ 1
        v = pu.item_vel[i]
        v = V2(v[1] * 0.995f0, (v[2] - 220f0 * Float32(dt)))  # gentle gravity + damping
        p = pu.item_pos[i]
        p = P2(p[1], p[2] + v[2] * Float32(dt))
        pu.item_vel[i] = v
        pu.item_pos[i] = p
        # pickup: derive radius from rendered item size for consistency
        items_size = 14f0 * (W / 800f0)
        pickup_r   = 0.67f0 * items_size
        if circle_rect_overlap(p, pickup_r, padd)
            activate!(pu, pu.item_typ[i], ball_center[], ball_vel_ref[])
            deleteat!(pu.item_pos, i)
            deleteat!(pu.item_vel, i)
            deleteat!(pu.item_typ, i)
        elseif p[2] < 0f0
            deleteat!(pu.item_pos, i)
            deleteat!(pu.item_vel, i)
            deleteat!(pu.item_typ, i)
        end
        i -= 1
    end

    # trail for the main ball only
    tp_old = trail_pts[]
    if launched[]
        tp_new       = length(tp_old) ≥ TRAIL_LEN ? vcat(tp_old[end-TRAIL_LEN+2:end], [ball_center[]]) : vcat(tp_old, [ball_center[]])
        n            = length(tp_new)
        alphas       = n == 1 ? Float32[0.35f0] : collect(LinRange{Float32}(0.06f0, 0.35f0, n))
        tbase        = (pu.pierce_ttl > 0f0) ? COLOR_PU_LASER : COLOR_BALL
        tcols        = [RGBAf(tbase.r, tbase.g, tbase.b, a) for a in alphas]
        trail_pts[]  = tp_new
        trail_cols[] = tcols
    else
        trail_pts[] = Point2f[]
        trail_cols[] = RGBAf[]
    end

    # flash → display colors
    base_cols = brick_base_colors_obs[]
    if length(fx.bflash) != length(base_cols)
        fx.bflash = fill(0f0, length(base_cols))
    end
    disp = Vector{RGBAf}(undef, length(base_cols))
    for i in eachindex(base_cols)
        fac = (i ≤ length(fx.bflash) ? clamp(fx.bflash[i] / FLASH_MAX, 0f0, 1f0) : 0f0) * 0.6f0
        disp[i] = fac > 0 ? mix_rgba(base_cols[i], COLOR_BALL, Float32(fac)) : base_cols[i]
        fx.bflash[i] = max(0f0, fx.bflash[i] - Float32(dt))
    end
    brick_color_obs[] = disp

    # particles
    i = length(fx.pttl)
    while i ≥ 1
        fx.pttl[i] -= Float32(dt)
        if fx.pttl[i] ≤ 0f0
            deleteat!(fx.ppos, i)
            deleteat!(fx.pvel, i)
            deleteat!(fx.pttl, i)
            deleteat!(fx.ptmax, i)
            deleteat!(fx.pcol, i)
        else
            fx.ppos[i] = P2(fx.ppos[i][1] + fx.pvel[i][1] * Float32(dt), fx.ppos[i][2] + fx.pvel[i][2] * Float32(dt))
            fx.pvel[i] = V2(fx.pvel[i][1] * 0.99f0, (fx.pvel[i][2] - 200f0 * Float32(dt)) * 0.99f0)
        end
        i -= 1
    end
    part_pos_obs[] = copy(fx.ppos)
    part_col_obs[] = [RGBAf(c.r, c.g, c.b, clamp(fx.pttl[j] / fx.ptmax[j], 0f0, 1f0) * 0.9f0) for (j, c) in enumerate(fx.pcol)]

    # timers decay
    pu.pierce_ttl = max(0f0, pu.pierce_ttl - Float32(dt))
    pu.laser_ttl  = max(0f0, pu.laser_ttl - Float32(dt))
    # update HUD text for timers
    pierce_txt[] = pu.pierce_ttl > 0f0 ? @sprintf("Pierce %.1fs", pu.pierce_ttl) : ""
    laser_txt[]  = pu.laser_ttl > 0f0 ? @sprintf("Laser %.1fs", pu.laser_ttl) : ""

    # level clear → next level
    if isempty(bricks_obs[])
        level[] = max(level[] + 1, 2)
        bricks, hps, cols = make_level_data_rand(level[])
        bricks_obs[] = bricks
        brick_hp_obs[] = hps
        brick_color_obs[] = cols
        brick_base_colors_obs[] = copy(cols)
        fx.bflash = fill(0f0, length(bricks))
        ball_vel_ref[] = V2(ball_vel_ref[][1] * 1.1f0, ball_vel_ref[][2] * 1.1f0)
        launched[] = false
        combo_count[] = 0
        combo_timer[] = 0f0
        combo_text[] = ""
        combo_color[] = RGBAf(COLOR_TEXT.r, COLOR_TEXT.g, COLOR_TEXT.b, 0f0)
        pu.drops_left = DROP_BUDGET
        pu.item_pos = Point2f[]
        pu.item_vel = Vec2f[]
        pu.item_typ = Symbol[]
        pu.ball_pos = Point2f[]
        pu.ball_vel = Vec2f[]
        pu.ball_ttl = Float32[]
        pu.shot_pos = Point2f[]
        pu.shot_vel = Vec2f[]
        pu.shot_ttl = Float32[]
        pu.pierce_ttl = 0f0
        pu.laser_ttl = 0f0
        pu.next_shot = 0f0
    end

    # push visuals for power‑ups
    items_pos_obs[] = copy(pu.item_pos)
    items_col_obs[] = [pu_color(t) for t in pu.item_typ]
    shots_pos_obs[] = copy(pu.shot_pos)
    extras_pos_obs[] = copy(pu.ball_pos)
    # update ball colors: turn orange‑red while pierce active
    nb = 1 + length(pu.ball_pos)
    bcol = pu.pierce_ttl > 0f0 ? COLOR_PU_LASER : COLOR_BALL
    ball_colors_obs[] = fill(bcol, nb)

    return
end

function breakout()
    fig = Figure(size=(Int(W), Int(H)), backgroundcolor=COLOR_BG)
    ax = Axis(fig[1, 1]; limits=((0f0, W), (0f0, H)), aspect=DataAspect(),
        backgroundcolor=COLOR_BG,
        xticksvisible=false, yticksvisible=false,
        xgridvisible=false, ygridvisible=false, xlabelvisible=false, ylabelvisible=false, titlevisible=false)
    hidedecorations!(ax)

    # state
    # unified scaling factors
    sw                 = W / 800f0
    sh                 = H / 600f0
    paddle_w, paddle_h = 100f0 * sw, 14f0 * sw
    paddle_y           = 40f0 * sh
    paddle_x           = Observable(W / 2 - paddle_w / 2)
    paddle_speed       = 480f0 * sw

    # Scale ball size with screen width; 800px width uses the current baseline (8px radius)
    ball_r      = 8f0 * (W / 800f0)
    ball_center = Observable(P2(W / 2, H * 0.3f0))  # will be reset onto paddle before the first frame
    ball_vel    = Ref(V2(176f0, 264f0))
    launched    = Observable(false)
    paused      = Observable(false)
    lives       = Observable(3)
    score       = Observable(0)
    level       = Observable(1)

    # level data
    bricks_obs      = Observable(Vector{GeometryBasics.Rect{2,Float32}}())
    brick_hp_obs    = Observable(Int[])
    brick_color_obs = Observable(RGBAf[])
    begin
        b, hps, cols = make_level_data_rand(level[])
        bricks_obs[] = b
        brick_hp_obs[] = hps
        brick_color_obs[] = cols
    end

    # base colors & fx state
    brick_base_colors_obs = Observable(copy(brick_color_obs[]))
    fx = FXState(Point2f[], Vec2f[], Float32[], Float32[], RGBAf[], fill(0f0, length(bricks_obs[])))

    # power‑up runtime & visuals
    pu = PUState(Point2f[], Vec2f[], Symbol[], Point2f[], Vec2f[], Float32[], Point2f[], Vec2f[], Float32[], 0f0, 0f0, 0f0, DROP_BUDGET)

    items_pos_obs = Observable(Point2f[])
    items_col_obs = Observable(RGBAf[])
    shots_pos_obs = Observable(Point2f[])
    extras_pos_obs = Observable(Point2f[])
    ball_colors_obs = Observable(RGBAf[])

    # draw order: bricks → items/shots → particles/trail → paddle/balls → HUD
    poly!(ax, bricks_obs; color=brick_color_obs, strokecolor=COLOR_STROKE, strokewidth=0.8 * (W / 800f0))

    # Power‑up item visual size (keep consistent with pickup computation)
    items_marker_size = 14f0 * (W / 800f0)
    shots_marker_size = 8f0 * (W / 800f0)
    scatter!(ax, items_pos_obs; marker=:rect, markersize=items_marker_size, color=items_col_obs)
    scatter!(ax, shots_pos_obs; marker=:circle, markersize=shots_marker_size, color=COLOR_LASER_SHOT)

    part_pos_obs = Observable(Point2f[])
    part_col_obs = Observable(RGBAf[])
    scatter!(ax, part_pos_obs; marker=:circle, markersize=6 * (W / 800f0), color=part_col_obs)

    trail_pts  = Observable(Point2f[])
    trail_cols = Observable(RGBAf[])
    scatter!(ax, trail_pts; marker=:circle, markersize=2 * ball_r, color=trail_cols)

    paddle_rect = lift(x -> rectf(x, paddle_y, paddle_w, paddle_h), paddle_x)
    poly!(ax, paddle_rect; color=COLOR_PADDLE, strokecolor=:transparent)

    # balls (main + extras) — ensure Vector{Point2f}
    ball_points = lift((c, extras) -> vcat([c], extras), ball_center, extras_pos_obs)
    ball_colors_obs[] = [COLOR_BALL]
    scatter!(ax, ball_points; marker=:circle, markersize=2 * ball_r, color=ball_colors_obs)

    # HUD bar
    hud_rect = rectf(0f0, H - 36f0 * (H / 600f0), W, 36f0 * (H / 600f0))
    poly!(ax, hud_rect; color=COLOR_HUDBG, strokecolor=:transparent)

    # texts
    text!(ax, lift(s -> "Score: $s", score); position=P2(10, H - 4), align=(:left, :top), color=COLOR_TEXT, fontsize=20)
    text!(ax, lift(l -> "Lives: $l", lives); position=P2(W / 2, H - 4), align=(:center, :top), color=COLOR_TEXT, fontsize=20)
    text!(ax, lift(lv -> "Level: $lv", level); position=P2(W - 10, H - 4), align=(:right, :top), color=COLOR_TEXT, fontsize=20)

    # status text for active timers (minimal)
    pierce_txt = Observable("")
    laser_txt  = Observable("")
    text!(ax, pierce_txt; position=P2(W * 0.12f0, H - 4), align=(:left, :top), color=COLOR_PU_PIERCE, fontsize=16)
    text!(ax, laser_txt; position=P2(W * 0.28f0, H - 4), align=(:left, :top), color=COLOR_PU_LASER, fontsize=16)

    combo_count = Observable(0)
    combo_timer = Observable(0f0)
    combo_text  = Observable("")
    combo_color = Observable(RGBAf(COLOR_TEXT.r, COLOR_TEXT.g, COLOR_TEXT.b, 0f0))
    text!(ax, combo_text; position=P2(W * 0.72f0, H - 4), align=(:center, :top), color=combo_color, fontsize=20)

    hint = lift(launched) do L
        L ? "" : "Press SPACE to launch  |  ←/→ Move  |  P Pause\n[  /  ] : Prev / Next Level\n(If keys don't work, click window to focus)"
    end
    text!(ax, hint; position=P2(W / 2, H * 0.5), align=(:center, :center), color=COLOR_TEXT, fontsize=18)

    display(fig)
    scene = ax.scene

    # Ensure the ball starts on the paddle before the first frame
    reset_ball!(ball_center, ball_vel, launched, paddle_x[], paddle_w, paddle_y, paddle_h, ball_r, trail_pts, trail_cols)

    # keyboard: level select and reset
    set_level! = function (L::Int)
        L2 = clamp(L, 1, 5)
        level[] = L2
        b, hps, cols = make_level_data_rand(level[])
        bricks_obs[] = b
        brick_hp_obs[] = hps
        brick_color_obs[] = cols
        brick_base_colors_obs[] = copy(cols)
        fx.bflash = fill(0f0, length(b))
        empty!(fx.ppos)
        empty!(fx.pvel)
        empty!(fx.pttl)
        empty!(fx.ptmax)
        empty!(fx.pcol)
        pu.item_pos = Point2f[]
        pu.item_vel = Vec2f[]
        pu.item_typ = Symbol[]
        pu.ball_pos = Point2f[]
        pu.ball_vel = Vec2f[]
        pu.ball_ttl = Float32[]
        pu.shot_pos = Point2f[]
        pu.shot_vel = Vec2f[]
        pu.shot_ttl = Float32[]
        pu.pierce_ttl = 0f0
        pu.laser_ttl = 0f0
        pu.next_shot = 0f0
        pu.drops_left = DROP_BUDGET
        part_pos_obs[] = Point2f[]
        part_col_obs[] = RGBAf[]
        items_pos_obs[] = Point2f[]
        items_col_obs[] = RGBAf[]
        shots_pos_obs[] = Point2f[]
        extras_pos_obs[] = Point2f[]
        combo_count[] = 0
        combo_timer[] = 0f0
        combo_text[] = ""
        combo_color[] = RGBAf(COLOR_TEXT.r, COLOR_TEXT.g, COLOR_TEXT.b, 0f0)
        reset_ball!(ball_center, ball_vel, launched, paddle_x[], paddle_w, paddle_y, paddle_h, ball_r, trail_pts, trail_cols)
        nothing
    end

    on(events(scene).keyboardbutton) do e
        if e.action == Keyboard.press
            e.key == Keyboard.space && (launched[] = true)
            e.key == Keyboard.p && (paused[] = !paused[])
            e.key == Keyboard.left_bracket && set_level!(level[] - 1)
            e.key == Keyboard.right_bracket && set_level!(level[] + 1)
            if e.key == Keyboard.r
                (lives[] = 3; score[] = 0; set_level!(1))
            end
            e.key == Keyboard.escape && close(fig)
        end
        return
    end

    # main fixed‑step loop
    prev_t = time()
    acc = 0.0
    @async begin
        while isopen(scene)
            now = time()
            acc += now - prev_t
            prev_t = now
            while acc ≥ FIXED_DT
                if !paused[]
                    try
                        update!(scene, FIXED_DT, ball_center, ball_vel, ball_r,
                            paddle_x, paddle_y, paddle_w, paddle_h, paddle_speed,
                            launched, bricks_obs, brick_hp_obs, brick_color_obs,
                            score, lives, trail_pts, trail_cols,
                            part_pos_obs, part_col_obs, fx, brick_base_colors_obs,
                            combo_count, combo_timer, combo_text, combo_color, level,
                            pu, items_pos_obs, items_col_obs, shots_pos_obs, extras_pos_obs, ball_colors_obs, pierce_txt, laser_txt)
                    catch err
                        @error "update! crashed" exception = (err, catch_backtrace())
                        paused[] = true
                    end
                end
                acc -= FIXED_DT
            end
            sleep(0.001)
        end
    end

    fig
end

breakout()
