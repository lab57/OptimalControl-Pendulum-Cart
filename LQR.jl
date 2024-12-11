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
        @parameters t
        
        # Get the equations and states
        eqs = equations(sys)  # Assumes `sys` has an `equations` function or field
        states = [sys.x, sys.θ1, sys.θ2, sys.dx, sys.dθ1, sys.dθ2]  
        rhs = [eq.rhs for eq in eqs]
    
    # Calculate Jacobian of RHS with respect to states
        J = Symbolics.jacobian(rhs, states)
        return J;
        # return A
    end
     
    function evaluate_jacobian(J, state_params, system_params)
        @independent_variables t
        @variables dθ1(t) dθ2(t) dθ1ˍt(t) dθ2ˍt(t)
        # Create substitution dictionary
        all_subs = Dict()
        
        # First add all our direct parameter values
        
        all_symbols = Set()
        for element in J
            union!(all_symbols, Symbolics.get_variables(element))
        end

        merge!(all_subs, state_params, system_params)

        print('\n')

        print('\n')
        print(all_subs)
        print('\n')
        print('\n')



        # all_subs[dθ1ˍt(t)] = all_subs[dθ1(t)]
        # all_subs[dθ2ˍt(t)] = all_subs[dθ2(t)]



        # Substitute and evaluate
        J_eval = substitute.(J, (all_subs,))
        
        # Convert to numeric matrix
        try
            J_numeric = float.(J_eval)
            return J_numeric
        catch e
            println("Error during numeric conversion: ", e)
            println("Non-numeric terms remaining in:")
            display(J_eval)
            return J_eval
        end
    end

    #B matrix
    function calculate_parametric_jacobian()
        # @independent_variables t
        # Define system parameters
        # @parameters  mc m1 m2 L1 L2 g H(::Real, ::Real, ::Real, ::Real, ::Real, ::Real, ::H_ARG) ARGS::H_ARG
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
    


    function lmao()
            
        function FORCE(x, θ1, θ2, v, ω1, ω2, ARGS)
            return 0
        end

        @independent_variables t
        @variables x(t) θ1(t) θ2(t) dx(t) dθ1(t) dθ2(t) dθ1_t(t) dmc m1 m2

        IC = [1,1,1,1,1,1];
        sys = createSystemNoExt(force_arg);
        init_struct = force_arg(0, 0);
        # createProblem(sys, H, arginit, IC=nothing, tspan = (0,20))
        prob = createProblem(sys,FORCE, init_struct, IC, (0,.1));

        # subs_dict = Dict(Differential(x) => dx, Differential(θ1) => dθ1, Differential(θ2) => 100, x => IC[1], θ1 => IC[2], θ2 => IC[3], mc => 100, m1 =>1, mc =>1)
        
        A = calculate_symbolic_jacobian(sys);

        # symbols = Symbolics.get_variables(A)

        # # For your Jacobian matrix
        # all_symbols = Set()
        # for element in A
        #     union!(all_symbols, Symbolics.get_variables(element))
        # end
        # println(all_symbols)

        state_params = Dict(
            sys.x => 1.0,
            sys.θ1 => 1.0,
            sys.θ2 => 1.0,
            sys.dx => 1.0,
            sys.dθ1 => 1.0,
            sys.dθ2 => 1.0,
        )

        # Example system parameters
        system_params = Dict(
            sys.mc => 100.0,
            sys.m1 => 1.0,
            sys.m2 => 1.0,
            sys.L1 => 1.0,
            sys.L2 => 1.0,
            sys.g => -9.8
        )

        # Evaluate the Jacobian
        A_evaluated = simplify(evaluate_jacobian(A, state_params, system_params))

        B = calculate_parametric_jacobian();
        # Penalize deviations in state variables
        Q = Diagonal([10.0, 1.0, 100.0, 1.0, 100.0, 1.0]) ; # Adjust weights as needed

        # Penalize control effort
        R = [1.0];  # Single input, so this is scalar

        # A_numeric = substitute(A, subs_dict)

        print(A)
        print('\n')
        print('\n')

        println(A_evaluated);
        # println(B)
        # println(Q)
        # println(R)

    end

    lmao()
end