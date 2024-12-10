using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics
using SciMLStructures:Tunable, replace, replace!
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: Nonnumeric, structural_parameters
using LinearAlgebra, ControlSystems
include("./DP.jl")

# Define the AcroCart struct
struct AcroCart
    l1::Float64         # Length of pendulum 1
    l2::Float64         # Length of pendulum 2
    m::Tuple{Float64, Float64, Float64}  # Masses: cart, pendulum1, pendulum2
    b::Tuple{Float64, Float64, Float64}  # Damping coefficients
    g::Float64          # Gravitational constant
    rail_lims::Tuple{Float64, Float64}   # Rail limits (min, max position)
    force_lims::Tuple{Float64, Float64}  # Force limits (min, max)
    n_u::Int64           # Number of control inputs
    n_d::Int64           # Number of degrees of freedom
    n_q::Int64           # Number of generalized coordinates
end

# Constructor for the AcroCart struct
function AcroCart(l1::Float64, l2::Float64, m::Tuple{Float64, Float64, Float64}, 
                  b::Tuple{Float64, Float64, Float64}, g::Float64, 
                  rail_lims::Tuple{Float64, Float64}, force_lims::Tuple{Float64, Float64}, 
                  n_u::Int64, n_d::Int64, n_q::Int64)
    return new(l1, l2, m, b, g, rail_lims, force_lims, n_u, n_d, n_q)
end




function step(q, u, dt, cart::AcroCart, disturb=0.0)
    """
    Integrates acrocart dynamics one timestep by partial-Verlet method, q_next = step(q, u, dt).
    Applies rail and force constraints.

    q:       array current state [pos, ang1, ang2, vel, angvel1, angvel2]
    u:       scalar current input force on cart
    dt:      timestep for integration
    disturb: optional scalar external disturbance force on cart
    cart:    instance of the AcroCart type
    """
    q = Float64.(q)  # Convert q to Float64 array

    # Enforce input saturation and add disturbance
    u_act = clamp(Float64(u), cart.force_lims[1], cart.force_lims[2]) + disturb

    # Get subarrays of pose and twist, and compute accel
    pose = q[1:cart.n_d]
    twist = q[cart.n_d+1:end]
    accel = f(q, u_act, cart)[cart.n_d+1:end]

    # Partial-Verlet integrate state and enforce rail constraints
    pose_next = pose .+ dt * twist .+ 0.5 * (dt^2) * accel

    if pose_next[1] < cart.rail_lims[1]
        pose_next[1] = cart.rail_lims[1]
        a1, a2 = f(q, 0.0, cart)[5:end]
        twist_next = [0.0, twist[2] + dt * a1, twist[3] + dt * a2]
    elseif pose_next[1] > cart.rail_lims[2]
        pose_next[1] = cart.rail_lims[2]
        a1, a2 = f(q, 0.0, cart)[5:end]
        twist_next = [0.0, twist[2] + dt * a1, twist[3] + dt * a2]
    else
        twist_next = twist .+ dt * accel
    end

    # Unwrap angles onto [-pi, pi]
    pose_next[2:3] .= mod.(pose_next[2:3] .+ π, 2 * π) .- π

    # Return next state = [pose, twist]
    return vcat(pose_next, twist_next)
end



function f(q, u, cart::AcroCart)
    """
    AcroCart continuous-time design-dynamics, qdot = f(q, u).
    Ignores any rail or force limits.

    q: array state [pos, ang1, ang2, vel, angvel1, angvel2]
    u: scalar input force on cart
    cart: instance of the AcroCart type
    """

    # Memoize trigonometric functions for efficiency
    sq1, cq1 = sin(q[2]), cos(q[2])
    sq2, cq2 = sin(q[3]), cos(q[3])
    sq12, cq12 = sin(q[2] - q[3]), cos(q[2] - q[3])

    # State-space equations for cart and pendulums
    eq1 = q[4]  # cart velocity
    eq2 = q[5]  # angular velocity of angle 1
    eq3 = q[6]  # angular velocity of angle 2

    # Dynamics terms
    term1 = cart.l1 * cart.l2 * (-4 * cart.m[2] + 9 * cart.m[3] * cq12^2 - 12 * cart.m[3]) * (-2 * cart.b[1] * q[4] + cart.l1 * cart.m[1] * sq1 * q[5]^2 + 2 * cart.l1 * cart.m[2] * sq1 * q[5]^2 + cart.l2 * cart.m[2] * sq2 * q[6]^2 + 2 * u)
    term2 = 3 * cart.l1 * (3 * (cart.m[1] + 2 * cart.m[2]) * cq12 * cq1 - 2 * (cart.m[1] + 3 * cart.m[2]) * cq2) * (2 * cart.b[3] * q[6] + cart.g * cart.l2 * cart.m[2] * sq2 - cart.l1 * cart.l2 * cart.m[2] * sq12 * q[5]^2)
    term3 = 3 * cart.l2 * (3 * cart.m[2] * cq12 * cq2 - 2 * (cart.m[1] + 2 * cart.m[2]) * cq1) * (2 * cart.b[2] * q[5] + cart.g * cart.l1 * cart.m[1] * sq1 + 2 * cart.g * cart.l1 * cart.m[2] * sq1 + cart.l1 * cart.l2 * cart.m[2] * sq12 * q[6]^2)

    denominator = 2 * cart.l1 * cart.l2 * (-9 * cart.m[2] * (cart.m[1] + 2 * cart.m[2]) * cq12 * cq1 * cq2 + 3 * cart.m[2] * (cart.m[1] + 3 * cart.m[2]) * cq2^2 + 9 * cart.m[2] * (cart.m[1] + cart.m[2]) * cq12^2 + 3 * (cart.m[1] + 2 * cart.m[2])^2 * cq1^2 - 4 * (cart.m[1] + 3 * cart.m[2]) * (cart.m[0] + cart.m[1] + cart.m[2]))

    # Return state derivatives: [velocities, accelerations]
    return [eq1, eq2, eq3, (term1 + term2 + term3) / denominator]
end



function F(Q, U, cart::AcroCart)
    """
    Vectorized version of acrocart continuous-time design-dynamics, Qdot = F(Q, U).
    Ignores any rail or force limits.

    Q: array N-by-n_q state timeseries
    U: array N-by-n_u input timeseries
    cart: instance of the AcroCart type
    """
    
    sQ1, cQ1 = sin(Q[:, 2]), cos(Q[:, 2])
    sQ2, cQ2 = sin(Q[:, 3]), cos(Q[:, 3])
    sQ12, cQ12 = sin.(Q[:, 2] .- Q[:, 3]), cos.(Q[:, 2] .- Q[:, 3])

    eq1 = Q[:, 4]
    eq2 = Q[:, 5]
    eq3 = Q[:, 6]

    term1 = cart.l1 * cart.l2 * (-4 * cart.m[2] + 9 * cart.m[3] * cQ12.^2 - 12 * cart.m[3]) .* (-2 * cart.b[1] * Q[:, 4] .+ cart.l1 * cart.m[1] * sQ1 .* Q[:, 5].^2 .+ 2 * cart.l1 * cart.m[2] * sQ1 .* Q[:, 5].^2 .+ cart.l2 * cart.m[2] * sQ2 .* Q[:, 6].^2 .+ 2 * U[:, 1])
    term2 = 3 * cart.l1 * (3 * (cart.m[1] + 2 * cart.m[2]) * cQ12 .* cQ1 .- 2 * (cart.m[1] + 3 * cart.m[2]) * cQ2) .* (2 * cart.b[3] * Q[:, 6] .+ cart.g * cart.l2 * cart.m[2] * sQ2 .- cart.l1 * cart.l2 * cart.m[2] * sQ12 .* Q[:, 5].^2)
    term3 = 3 * cart.l2 * (3 * cart.m[2] * cQ12 .* cQ2 .- 2 * (cart.m[1] + 2 * cart.m[2]) * cQ1) .* (2 * cart.b[2] * Q[:, 5] .+ cart.g * cart.l1 * cart.m[1] * sQ1 .+ 2 * cart.g * cart.l1 * cart.m[2] * sQ1 .+ cart.l1 * cart.l2 * cart.m[2] * sQ12 .* Q[:, 6].^2)

    denominator = 2 * cart.l1 * cart.l2 * (-9 * cart.m[2] * (cart.m[1] + 2 * cart.m[2]) * cQ12 .* cQ1 .* cQ2 .+ 3 * cart.m[2] * (cart.m[1] + 3 * cart.m[2]) .* cQ2.^2 .+ 9 * cart.m[2] * (cart.m[1] + cart.m[2]) .* cQ12.^2 .+ 3 * (cart.m[1] + 2 * cart.m[2]).^2 .* cQ1.^2 .- 4 * (cart.m[1] + 3 * cart.m[2]) .* (cart.m[0] + cart.m[1] + cart.m[2]))

    return hcat(eq1, eq2, eq3, (term1 .+ term2 .+ term3) ./ denominator)
end

# Example AcroCart struct with initial conditions
cart = AcroCart()

# Initial state: [position, angle1, angle2, velocity, angular velocity1, angular velocity2]
q_initial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Time step and duration of simulation
dt = 0.01  # Time step in seconds
t_max = 10.0  # Total simulation time
steps = Int(t_max / dt)  # Number of steps

# Apply a control input, which could be a time-varying function or constant
u = 1.0  # Example constant force input (could be time-varying)

# Optionally, define a disturbance force
disturb = 0.0

# Initialize state and prepare to animate
q = q_initial
frames = []  # Store the frames for animation

# Loop through timesteps and capture frames for animation
@animate for t in 1:steps
    q = step(q, u, dt, cart, disturb)  # Update state using the step method
    
    # Extract position and angle information from the state
    pos = q[1]
    ang1 = q[2]
    ang2 = q[3]

    # Set up the plot
    plot(
        [pos, pos + cos(ang1)], [0, sin(ang1)], label="Cart", xlabel="Position", ylabel="Height",
        title="AcroCart Simulation", legend=:topright, aspect_ratio=1, xlims=(-5, 5), ylims=(-2, 2),
        color=:blue, lw=2
    )
    
    # Add lines representing the angles (you can adjust this part based on how the cart and angles are visualized)
    plot!([pos, pos + cos(ang1)], [0, sin(ang1)], color=:red, lw=2)  # Represent angle1 (cart angle)
    plot!([pos + cos(ang1), pos + cos(ang1) + cos(ang2)], [sin(ang1), sin(ang1) + sin(ang2)], color=:green, lw=2)  # Represent angle2

    # Append current frame to the list of frames
    push!(frames, plot)
end

# Save the animation as a gif
gif_name = "acrocart_simulation.gif"
@animate for frame in frames
    display(frame)
end

# The gif will be saved automatically, but you can also use the save function explicitly
savefig(gif_name)  # Save the final animation as gif