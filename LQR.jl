using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics
using SciMLStructures:Tunable, replace, replace!
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: Nonnumeric
using LinearAlgebra, ControlSystems
include("./DP.jl");

begin

    #A matrix
    using Symbolics

    # Function to calculate the Jacobian matrix
    function calculate_symbolic_jacobian(sys)
        J = expand_derivatives.(calculate_jacobian(sys));
        return J;
    end
     
    function evaluate_jacobian(sys, inputs)
        j = ModelingToolkit.expand_derivatives.(calculate_jacobian(sys))
        states = unknowns(sys)
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
    


    function calculateK(inputs, Q, R)
            
        function FORCE(x, θ1, θ2, v, ω1, ω2, ARGS)
            return 0
        end

        sys = createSystemNoExt(force_arg_temp);
        init_struct = force_arg_temp(0, 0);
        state_variables = inputs[1:6]

        prob = createProblem(sys,FORCE, init_struct, state_variables, (0,.1));

        # Evaluate the Jacobian
        A_evaluated = Float64.((evaluate_jacobian(sys,inputs))[1:6,1:6])
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
        

        sys = createSystem(force_arg_temp)
        init_struct = force_arg_temp(0, 0)
        prob = createProblem(sys,FORCE, init_struct)


        # Add the two additional state variables as in your example
        ICs = [initial_state..., 0, 0]
        
        params = [1, 1, 1, -9.8, 1, 1]  # Your standard parameters
        
        # Update Problem Parameters using your method
        ps = parameter_values(prob)
        ps = replace(Tunable(), ps, params)
        newprob = remake(prob; u0=ICs, p=ps)
        
        integrator = init(newprob, ImplicitMidpoint(), dt=dt)
        return integrator, init_struct
    end
    
    function step_system(integrator, init_struct, current_state, Q, R, dt)
        # Recalculate K for current state
        K = calculateK(current_state, Q, R)
        
        # Calculate control input
        u = -K * current_state
        init_struct.a = u[1]  # Update force in init_struct
        
        # Step forward
        step!(integrator, dt, true)
        
        # Return new state (first 6 components as in your example)
        return copy(integrator.u[1:6])
    end
    
    function run_until_complete(initial_state, Q, R, dt; max_t=20.0, max_x=10.0)
        integrator, init_struct = setup_system(initial_state, dt)
        
        times = Float64[]
        states = Vector{Vector{Float64}}()
        controls = Float64[]
        
        t_curr = 0.0
        current_state = copy(initial_state)
        
        while t_curr <= max_t
            push!(times, t_curr)
            push!(states, current_state)
            
            # Check termination
            x = current_state[1]
            if abs(x) > max_x
                break
            end
            
            # Step forward
            current_state = step_system(integrator, init_struct, current_state, Q, R, dt)
            t_curr += dt
            
            # Store control input
            push!(controls, init_struct.a)
        end
        
        return times, states, controls
    end


    initial_state = [1.0,1.0,1.0,1.0,1.0,1.0]
    Q = Diagonal([10.0, 1.0, 100.0, 1.0, 100.0, 1.0])
    R = [1.0]
    times, states = run_until_complete(initial_state, Q, R, 0.01)

end