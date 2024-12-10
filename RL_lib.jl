using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics
using SciMLStructures:Tunable, replace, replace!
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: Nonnumeric
using ReinforcementLearning
using Random

begin

    export DoublePendulumCartEnv

    using ReinforcementLearning
    using ModelingToolkit
    using DifferentialEquations
    using Plots
    using Symbolics
    using SciMLBase: DEIntegrator
    using SciMLStructures: Tunable, replace, replace!
    using SymbolicIndexingInterface: parameter_values, state_values
    using ModelingToolkit: Nonnumeric
    using IntervalSets
    using Random
    
    
    # First, let's create our force function type that will store the current action
    mutable struct ForceFunction{T}
        current_action::T
    end
    
    # Define how the force function is called within the ODE system
    function (f::ForceFunction)(x, θ1, θ2, dx, dθ1, dθ2, args)
        return f.current_action
    end
    
    # System creation function - similar to original but adapted for our needs
    function createSystem(H_ARG::Type)
        @variables t
        @variables x(t) θ1(t) θ2(t) dx(t) dθ1(t) dθ2(t)
        @parameters mc m1 m2 L1 L2 g H(::Real, ::Real, ::Real, ::Real, ::Real, ::Real, ::H_ARG) ARGS::H_ARG
        
        d = Differential(t)
        
        # Define first-order equations
        eqs = [
            dx ~ d(x),
            dθ1 ~ d(θ1),
            dθ2 ~ d(θ2),
        ]
        
        # Define energy terms as in original
        I1 = m1*L1^2
        I2 = m2*L2^2
        l1 = L1
        l2 = L2
        
        T1 = 1/2 * (mc + m1 + m2) * dx^2
        T2 = 1/2 * (m1*l1^2 + m2*L1^2 + I1) * dθ1^2
        T3 = 1/2 * (m2*l2^2 + I2) * dθ2^2
        T4 = (m1*l1 + m2*L1) * cos(θ1) * dx * dθ1
        T5 = m2*l2 * cos(θ2) * dx * dθ2
        T6 = m2*L1*l2 * cos(θ1 - θ2) * dθ1 * dθ2
        
        T = T1 + T2 + T3 + T4 + T5 + T6
        V = g*(m1*l1 + m2*L1)*cos(θ1) + m2*g*l2*cos(θ2)
        L = T - V
        
        # Euler-Lagrange equations with force term
        EL = expand_derivatives(d(Differential(dx)(L))) - expand_derivatives(Differential(x)(L)) ~ 
             H(x, θ1, θ2, dx, dθ1, dθ2, ARGS)
        push!(eqs, EL)
        
        EL = expand_derivatives(d(Differential(dθ1)(L))) - expand_derivatives(Differential(θ1)(L)) ~ 0
        push!(eqs, EL)
        
        EL = expand_derivatives(d(Differential(dθ2)(L))) - expand_derivatives(Differential(θ2)(L)) ~ 0
        push!(eqs, EL)
        
        @named HO = ODESystem(eqs, t, [x, θ1, θ2, dx, dθ1, dθ2], [mc, m1, m2, L1, L2, g, H, ARGS])
        HO = structural_simplify(HO)
        
        return HO
    end
    
    # Environment structure that will maintain the integrator
    mutable struct DoublePendulumCartEnv{T,RNG<:AbstractRNG} <: AbstractEnv
        force_func::ForceFunction{T}
        integrator::DEIntegrator
        state::Vector{T}
        done::Bool
        t::Int
        rng::RNG
        max_steps::Int
        theta_threshold::T
        x_threshold::T
    end
    
    """
        DoublePendulumCartEnv(;kwargs...)
    
    Create a double pendulum cart environment using DifferentialEquations.jl integration.
    
    # Keyword arguments
    - `T = Float64`
    - `rng = Random.default_rng()`
    - `dt = 0.01`
    - `max_steps = 2000`
    - `theta_threshold = 12.0` (degrees)
    - `x_threshold = 2.4`
    """
    function DoublePendulumCartEnv(;
        T=Float64,
        rng=Random.default_rng(),
        dt=0.01,
        max_steps=2000,
        theta_threshold=12.0,
        x_threshold=2.4
    )
        # Create force function with initial zero action
        force_func = ForceFunction{T}(zero(T))
        
        # Create the ODE system
        sys = createSystem(typeof(force_func))
        
        # Initial conditions and timespan
        initial_state = zeros(T, 8)  # [x, θ1, θ2, dx, dθ1, dθ2, 0, 0]
        tspan = (0.0, float(max_steps) * dt)
        
        # Set up parameters
        ps = [
            sys.mc => T(100.0),
            sys.m1 => T(1.0),
            sys.m2 => T(1.0),
            sys.L1 => T(1.0),
            sys.L2 => T(1.0),
            sys.g => T(-9.8),
            sys.H => force_func,
            sys.ARGS => force_func
        ]
        
        # Create the ODE problem
        prob = ODEProblem(sys, initial_state, tspan, ps)
        
        # Create integrator with small dt for accuracy
        integrator = init(prob, ImplicitMidpoint(), dt=dt, adaptive=false)
        
        # Create environment
        env = DoublePendulumCartEnv(
            force_func,
            integrator,
            copy(initial_state),
            false,
            0,
            rng,
            max_steps,
            T(theta_threshold * π / 180),  # Convert to radians
            T(x_threshold)
        )
        
        reset!(env)
        env
    end
    
    # Define the RL interface
    RLBase.state_space(env::DoublePendulumCartEnv{T}) where T = Space(
        [
            -env.x_threshold .. env.x_threshold,           # x
            -env.theta_threshold .. env.theta_threshold,   # θ1
            -env.theta_threshold .. env.theta_threshold,   # θ2
            -T(8.0) .. T(8.0),                            # dx
            -4π .. 4π,                                    # dθ1
            -4π .. 4π                                     # dθ2
        ]
    )
    
    RLBase.action_space(env::DoublePendulumCartEnv{T}) where T = Space(-T(1.0) .. T(1.0))
    RLBase.state(env::DoublePendulumCartEnv) = env.state[1:6]  # Return only physical states
    RLBase.is_terminated(env::DoublePendulumCartEnv) = env.done
    
    function RLBase.reset!(env::DoublePendulumCartEnv{T}) where T
        # Reset the integrator to initial conditions
        reinit!(env.integrator, T(0.1) * rand(env.rng, T, 8) .- T(0.05))
        
        # Reset environment state
        env.state = copy(env.integrator.u)
        env.t = 0
        env.done = false
        env.force_func.current_action = zero(T)
        
        return env.state[1:6]
    end

    function RLBase.reward(env::DoublePendulumCartEnv{T}) where T

        return 0
    end
    
    function RLBase.act!(env::DoublePendulumCartEnv{T}, action) where T
        # Update force function with new action (scaled by 10 for meaningful force)
        env.force_func.current_action = T(10.0) * clamp(action, -one(T), one(T))
        
        # Step the integrator forward
        step!(env.integrator)
        
        # Update environment state
        env.state = copy(env.integrator.u)
        env.t += 1
        
        # Check termination conditions
        env.done = abs(env.state[1]) > env.x_threshold ||
                   abs(env.state[2]) > env.theta_threshold ||
                   abs(env.state[3]) > env.theta_threshold ||
                   env.t >= env.max_steps
        
        # Calculate reward
        if env.done
            return zero(T)
        end
        
        # Reward based on keeping pendulum upright and cart centered
        angle_penalty = (env.state[2]^2 + env.state[3]^2) / (env.theta_threshold^2)
        position_penalty = env.state[1]^2 / (env.x_threshold^2)
        velocity_penalty = T(0.1) * sum(env.state[4:6].^2)
        action_penalty = T(0.001) * (env.force_func.current_action^2)
        
        return one(T) - (angle_penalty + T(0.1) * position_penalty + 
                        velocity_penalty + action_penalty)
    end
end

begin

    using Plots
    using Statistics
    
    """
    Runs an episode of the double pendulum cart environment and returns visualization and statistics.
    This function demonstrates both discrete and continuous action variants of the environment.
    """
    function run_example(;
        num_steps=500,
        continuous=true,
        animate=true,
        save_animation=false
    )
        # Create the environment with specified action type
        env = DoublePendulumCartEnv(
            max_steps=num_steps,
            # Start with slightly offset initial conditions to make it interesting
            x_threshold=2.4,
            theta_threshold=45.0  # More permissive angle threshold for visualization
        )
        
        # Initialize storage for trajectory data
        positions = zeros(num_steps, 3)  # x, θ1, θ2
        velocities = zeros(num_steps, 3)  # dx, dθ1, dθ2
        rewards = zeros(num_steps)
        actions = zeros(num_steps)
        
        # Reset environment to start fresh
        reset!(env)
        
        # Run simulation
        for step in 1:num_steps
            # Generate action: alternating positive and negative for visualization
            if continuous
                action = sin(2π * step / 50) # Smooth sinusoidal control
            else
                action = (step % 2 == 0) ? 1 : 2  # Alternate between left and right
            end
            
            # Take action and record data
            act!(env, action)
            
            # Store state information
            positions[step, :] = env.state[1:3]  # x, θ1, θ2
            velocities[step, :] = env.state[4:6]  # dx, dθ1, dθ2
            rewards[step] = reward(env)
            actions[step] = env.action
            
            # Break if environment is done
            if is_terminated(env)
                positions = positions[1:step, :]
                velocities = velocities[1:step, :]
                rewards = rewards[1:step]
                actions = actions[1:step]
                break
            end
        end
        
        # Create visualization
        if animate
            anim = @animate for i in 1:size(positions, 1)
                # Calculate pendulum endpoints for visualization
                x = positions[i, 1]
                θ1 = positions[i, 2]
                θ2 = positions[i, 3]
                
                # First pendulum coordinates
                x1 = x + env.params.L1 * sin(θ1)
                y1 = -env.params.L1 * cos(θ1)
                
                # Second pendulum coordinates
                x2 = x1 + env.params.L2 * sin(θ2)
                y2 = y1 - env.params.L2 * cos(θ2)
                
                # Create the plot
                plot(
                    xlim=(-3, 3),
                    ylim=(-3, 3),
                    aspect_ratio=:equal,
                    title="Double Pendulum Cart - Step $i",
                    legend=false
                )
                
                # Draw cart
                rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
                cart = rectangle(0.4, 0.2, x-0.2, -0.1)
                plot!(cart, color=:blue, fillalpha=0.3)
                
                # Draw pendulums
                plot!([x, x1], [0, y1], color=:black, linewidth=2)  # First pendulum
                plot!([x1, x2], [y1, y2], color=:black, linewidth=2)  # Second pendulum
                
                # Draw joints
                scatter!([x], [0], color=:blue, markersize=8)  # Cart-pendulum joint
                scatter!([x1], [y1], color=:red, markersize=6)  # Inter-pendulum joint
                scatter!([x2], [y2], color=:red, markersize=6)  # End mass
            end
            
            if save_animation
                gif(anim, "double_pendulum_cart.gif", fps=30)
            end
        end
        
        # Create analysis plots
        p1 = plot(positions, 
            label=["Cart Position" "θ₁" "θ₂"],
            title="Positions over Time",
            xlabel="Step",
            ylabel="Position/Angle")
        
        p2 = plot(velocities,
            label=["Cart Velocity" "ω₁" "ω₂"],
            title="Velocities over Time",
            xlabel="Step",
            ylabel="Velocity/Angular Velocity")
        
        p3 = plot(rewards,
            label="Reward",
            title="Rewards over Time",
            xlabel="Step",
            ylabel="Reward")
        
        p4 = plot(actions,
            label="Action",
            title="Actions over Time",
            xlabel="Step",
            ylabel="Action")
        
        # Combine all plots
        analysis_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(800,600))
        
        # Print summary statistics
        println("\nSimulation Summary:")
        println("Total steps: $(length(rewards))")
        println("Average reward: $(mean(rewards))")
        println("Final position: $(positions[end, 1])")
        println("Final angles: θ₁=$(rad2deg(positions[end, 2]))°, θ₂=$(rad2deg(positions[end, 3]))°")
        
        return analysis_plot
    end
    
    # Run the example with both continuous and discrete actions
    println("Running continuous action example...")
    continuous_plot = run_example(continuous=true, num_steps=500)
    display(continuous_plot)
    
    println("\nRunning discrete action example...")
    discrete_plot = run_example(continuous=false, num_steps=500)
    display(discrete_plot)

end