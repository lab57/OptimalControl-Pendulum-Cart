using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics
using SciMLStructures:Tunable, replace, replace!
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: Nonnumeric
using LinearAlgebra, ControlSystems

open("/dev/null", "w") do io
    redirect_stdout(io) do
        include("./DP.jl")
    end
end

begin

    #A matrix
    using Symbolics

    # Function to calculate the Jacobian matrix
    function calculate_symbolic_jacobian(sys)
        j = expand_derivatives.(calculate_jacobian(sys));
        return j;
    end
     
    function evaluate_jacobian(j, sys, inputs)
        j = ModelingToolkit.expand_derivatives.(calculate_jacobian(sys))
        states = unknowns(sys)
        # print(states)
        params = parameters(sys)
        st_d = Dict(
            [
            sys.x => inputs[1], 
            sys.θ1 => inputs[2], 
            sys.θ2 => inputs[3], 
            sys.dx => inputs[4], 
            sys.dθ1 => inputs[5], 
            sys.dθ2 => inputs[6],
            unknowns(sys)[7] => inputs[7],
            unknowns(sys)[8]=> inputs[8],
            # unknowns(sys)[7] => 1,
            # unknowns(sys)[8]=> 1,
            sys.m1 => 1,
            sys.m2 => 1,
            sys.mc => 5,
            sys.g => -9.8,
            sys.L1 => 1,
            sys.L2 => 1           
            ]
        )

        evalj = substitute(j, st_d)

        return evalj
    end

    #B matrix
    function calculate_parametric_jacobian()
        m_cart = 100
        m1 = 1
        m2 = 1
        l1 = 1
        l2 = 1

        I1 = (1/3) * m1 * l1^2  # Moment of inertia of pendulum 1 (thin rod)
        I2 = (1/3) * m2 * l2^2  # Moment of inertia of pendulum 2 (thin rod)

        # Calculate B matrix elements
        b1 = 1 / (m_cart + m1 + m2)  # Effect of force on cart acceleration
        b2 = -m1 * l1 / (I1 + m1 * l1^2)  # Effect of force on pendulum 1 angular acceleration
        b3 = -m2 * l2 / (I2 + m2 * l2^2)  # Effect of force on pendulum 2 angular acceleration

        # Define the B matrix
        B = [
            0;       # x (position) does not directly depend on input force
            0;       # θ₁ (angle) does not directly depend on input force
            0;       # θ₂ (angle) does not directly depend on input force
            b1;      # dx/dt (cart velocity) depends on input force
            b2;      # dθ₁/dt (pendulum 1 angular velocity)
            b3       # dθ₂/dt (pendulum 2 angular velocity)
        ]
        return B;
    end
    


    function calculateK(inputs, j, Q, R)
            
        function FORCE(x, θ1, θ2, v, ω1, ω2, ARGS)
            return 0
        end

        sys = createSystemNoExt(force_arg_temp);
        init_struct = force_arg_temp(0, 0);
        state_variables = inputs[1:6] 

        prob = createProblem(sys,FORCE, init_struct, state_variables, (0,.1));

        # Evaluate the Jacobian


        A_evaluated = Float64.((evaluate_jacobian(j,sys,inputs))[[1:3; 6:end], [1:3; 6:end]])
        # A_evaluated = Float64.((evaluate_jacobian(j,sys,inputs))[1:6, 1:6]);
        B = Float64.(calculate_parametric_jacobian());

        L = lqr(Discrete,A_evaluated,B,Q,R) 
        return L;
    end

    function setup_system(initial_state, dt)
        function FORCE(x, θ1, θ2, v, ω1, ω2, ARGS)
            return 0;
        end
        
        # sys = createSystem(force_arg_temp)  # Using your force_arg struct
        # init_struct = force_arg_temp(0,0)  # Initial force of 0
        # prob = createProblem(sys, FORCE, init_struct)
        

        sys = createSystem(force_arg_temp3)
        init_struct = force_arg_temp3(0)
        prob = createProblem(sys,FORCE, init_struct)


        # Add the two additional state variables as in your example
        ICs = [initial_state[1:6]..., 0, 0]
        
        params = [1, 1, 5, -9.8, 1, 1]  # Your standard parameters
        
        # Update Problem Parameters using your method
        ps = parameter_values(prob)
        ps = replace(Tunable(), ps, params)
        newprob = remake(prob; u0=ICs, p=ps)
        
        integrator = init(newprob, ImplicitMidpoint(), dt=dt)

        j = calculate_symbolic_jacobian(sys)
            
        return integrator, init_struct, j
    end
    
    function step_system(integrator, init_struct, current_state, j, Q, R, dt)
        xref = [0.0, π, π, 0.0, 0.0, 0.0]
    
        # Recalculate K for current state
        K = calculateK(current_state, j, Q, R)
        
        # Calculate control input relative to reference state
        u = -K * (current_state[1:6] - xref)

        init_struct.a = u[1]  # Update force in init_struct
        
        # Step forward
        step!(integrator, dt, true)
        
        # Return new state (all 8 components as in your example)
        return copy(integrator.u)
    end
    
    function run_until_complete(initial_state, Q, R, dt; max_t=4.0, max_x=10.0)
        integrator, init_struct, j = setup_system(initial_state, dt)
        
        times = Float64[]
        states = Vector{Vector{Float64}}()
        controls = Float64[]
        
        t_curr = 0.0
        current_state = copy(initial_state)
        
        while t_curr <= max_t
            push!(times, t_curr)
            push!(states, current_state[1:6])
            
            # Check termination
            x = current_state[1]
            if abs(x) > max_x
                break
            end
            
            # Step forward
            current_state = step_system(integrator, init_struct, current_state, j, Q, R, dt)
            t_curr += dt
            println(t_curr);
            println();

            
            # Store control input
            push!(controls, init_struct.a)
        end
        
        return times, states, controls
    end


    function animate_pendulum(times, states; fps=30)
        # Animation settings
        L1 = 1.0  # Length of first pendulum
        L2 = 1.0  # Length of second pendulum
        
        # Create animation
        anim = @animate for i in 1:length(times)
            # Get current state
            x = states[i][1]     # cart position
            θ1 = states[i][2]    # first pendulum angle
            θ2 = states[i][3]    # second pendulum angle
            
            # Calculate positions
            # Cart
            cart_x = x
            cart_y = 0
            
            # First pendulum end position
            p1_x = cart_x + L1 * sin(θ1)
            p1_y = -L1 * cos(θ1)
            
            # Second pendulum end position
            p2_x = p1_x + L2 * sin(θ2)
            p2_y = p1_y - L2 * cos(θ2)
            
            # Plot
            plt = plot(
                xlim = (-5, 5),
                ylim = (-1, 3),
                aspect_ratio = :equal,
                legend = false,
                title = "t = $(round(times[i], digits=2))s"
            )
            
            # Draw cart
            plot!([cart_x-0.2, cart_x+0.2], [cart_y, cart_y], color=:blue, linewidth=5)
            
            # Draw pendulums
            plot!([cart_x, p1_x], [cart_y, p1_y], color=:black, linewidth=2)
            plot!([p1_x, p2_x], [p1_y, p2_y], color=:black, linewidth=2)
            
            # Draw masses
            scatter!([p1_x], [p1_y], color=:red, markersize=10)
            scatter!([p2_x], [p2_y], color=:green, markersize=10)
        end
        
        return gif(anim, "pendulum.gif", fps=fps)
    end

    initial_state = [0.1, 3, 3, 0.0, 0.0, 0.0,0,0]
    Q = Float64.(Diagonal([10.0, 100.0, 100.0, 1.0, 1.0, 1.0]))
    R = I
    times, states, controls = run_until_complete(initial_state, Q, R, 0.001)

    animation = animate_pendulum(times,states)

    # Create a plot with multiple subplots
    p1 = plot(times, getindex.(states, 1), label="x", title="Cart Position")
    p2 = plot(times, getindex.(states, 2), label="θ1", title="Pendulum 1 Angle")
    p3 = plot(times, getindex.(states, 3), label="θ2", title="Pendulum 2 Angle")
    p4 = plot(times, controls, label="Control Force", title="Control Input")

    # Combine plots into a single figure
    combined_plot = plot(p1, p2, p3, p4, layout=(4,1), size=(800,800))

    # Display or save the plot
    savefig(combined_plot, "system_states_plot.png")
end