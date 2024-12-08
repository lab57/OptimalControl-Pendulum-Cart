using DifferentialEquations
using LinearAlgebra
using Plots

# System parameters
const G = 9.8
const L1 = 1.0
const L2 = L1
const m1 = 1.0
const m2 = 1.0
const mc = 1.0
const MAX_SPEED = 20.0

# f1 computes accelerations (position independent part)
function f1!(dv, v, u, p, t)
    # Extract positions and velocities
    x, θ1, θ2 = u
    v_cart, ω1, ω2 = v
    
    # Precompute trigonometric terms
    s1, c1 = sincos(θ1)
    s2, c2 = sincos(θ2)
    s12 = sin(θ1 - θ2)
    c12 = cos(θ1 - θ2)
    
    # Mass matrix elements
    M11 = mc + m1 + m2
    M12 = (m1 + m2) * L1 * c1
    M13 = m2 * L2 * c2
    M21 = M12
    M22 = (m1 + m2) * L1^2
    M23 = m2 * L1 * L2 * c12
    M31 = M13
    M32 = M23
    M33 = m2 * L2^2
    
    # Force terms
    F1 = (m1 + m2) * L1 * ω1^2 * s1 + m2 * L2 * ω2^2 * s2
    F2 = -(m1 + m2) * G * L1 * s1 - m2 * L1 * L2 * ω2^2 * s12
    F3 = -m2 * G * L2 * s2 + m2 * L1 * L2 * ω1^2 * s12
    
    # Solve the system
    M = [M11 M12 M13; M21 M22 M23; M31 M32 M33]
    F = [F1, F2, F3]
    
    acc = try
        M \ F
    catch
        zeros(3)
    end
    
    # Update acceleration terms
    dv[1] = acc[1]
    dv[2] = acc[2]
    dv[3] = acc[3]
    
    # Apply speed limits
    for i in 1:3
        dv[i] = clamp(dv[i], -MAX_SPEED, MAX_SPEED)
    end
end

# f2 computes velocities (momentum part)
function f2!(du, v, u, p, t)
    du[1] = v[1]  # dx/dt = v_cart
    du[2] = v[2]  # dθ1/dt = ω1
    du[3] = v[3]  # dθ2/dt = ω2
end

function get_positions(x, θ1, θ2)
    # First pendulum end position
    x1 = x + L1 * sin(θ1)
    y1 = -L1 * cos(θ1)
    
    # Second pendulum end position
    x2 = x1 + L2 * sin(θ2)
    y2 = y1 - L2 * cos(θ2)
    
    return [x, x1, x2], [0.0, y1, y2]
end

function simulate_and_animate(θ1_0::Float64, ω1_0::Float64, θ2_0::Float64, ω2_0::Float64, 
                            x_0::Float64, v_0::Float64, duration::Float64; fps=30)
    # Initial conditions
    u0 = [x_0, θ1_0, θ2_0]        # positions
    v0 = [v_0, ω1_0, ω2_0]        # velocities
    tspan = (0.0, duration)
    
    # Create and solve the dynamical ODE problem
    prob = DynamicalODEProblem(f1!, f2!, v0, u0, tspan)
    sol = solve(prob, RK4(), dt=0.005)
    
    # Create animation with time points matching desired fps
    t_points = range(0, duration, length=Int(round(duration * fps)))
    
    # Get x-coordinate bounds for plot
    x_positions = [sol(t)[1] for t in t_points]   # [2] for positions, [1] for x coordinate
    x_min, x_max = minimum(x_positions), maximum(x_positions)
    
    anim = @animate for t in t_points
        # Get state at current time
        state = sol(t)
        x, θ1, θ2 = state[1:3]
        # Get positions for visualization
        xs, ys = get_positions(x, θ1, θ2)
        
        # Create plot
        plot(background_color=:white,
             aspect_ratio=:equal,
             legend=false,
             title="Double Pendulum on Cart (Symplectic)",
             xlabel="x position (m)",
             ylabel="y position (m)",
             xlim=(x_min-2L1-L2, x_max+2L1+L2),
             ylim=(-2(L1+L2), 0.5))
        
        # Draw cart
        cart_width = 0.3
        cart_height = 0.1
        cart_x = [x-cart_width/2, x+cart_width/2, x+cart_width/2, x-cart_width/2, x-cart_width/2]
        cart_y = [0, 0, -cart_height, -cart_height, 0]
        plot!(cart_x, cart_y, fillcolor=:gray, fill=true, color=:black)
        
        # Draw pendulum rods and masses
        plot!([xs[1], xs[2]], [ys[1], ys[2]], color=:black, linewidth=2)
        plot!([xs[2], xs[3]], [ys[2], ys[3]], color=:black, linewidth=2)
        
        scatter!([xs[2]], [ys[2]], color=:blue, markersize=10)
        scatter!([xs[3]], [ys[3]], color=:red, markersize=10)
        
        annotate!(x_min-2L1, 0.3, text("t = $(round(t, digits=2))s", 10))
    end
    
    return gif(anim, "double_pendulum_cart.gif", fps=fps)
end

# Example usage
function main()
    θ1_0 = 2.0   # Initial angle of first pendulum
    ω1_0 = 0.0    # Initial angular velocity of first pendulum
    θ2_0 = 0.0   # Initial angle of second pendulum
    ω2_0 = 0.0    # Initial angular velocity of second pendulum
    x_0 = 0.0     # Initial cart position
    v_0 = 0.0     # Initial cart velocity
    
    simulate_and_animate(θ1_0, ω1_0, θ2_0, ω2_0, x_0, v_0, 10.0, fps=30)
    println("Animation saved as 'double_pendulum_cart.gif'")
end

main()