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
     
    function evaluate_jacobian(J,subsitution_values)

        variables = unique(v for row in eachrow(J) for el in row for v in get_variables(el))
        variables = sort(variables, by=string)

        # println(length(variables))
        # println(variables)

        st_d = Dict(
            [
                variables[1] => subsitution_values[1],
                variables[2] => subsitution_values[2],
                variables[3] => subsitution_values[3],
                variables[4] => subsitution_values[4],
                variables[5] => subsitution_values[5],
                variables[6] => subsitution_values[6],
                variables[7] => subsitution_values[7],
                variables[8] => subsitution_values[8],
                variables[9] => subsitution_values[9],
                variables[10] => subsitution_values[10],
                variables[11] => subsitution_values[11],
                variables[12] => subsitution_values[12],
            ]
        )
        evalJ = substitute(J, st_d)
        # println()
        # println(evalJ)
        # println()

        return evalJ
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

        sys = createSystemNoExt(force_arg_temp);
        init_struct = force_arg_temp(0, 0);
        # createProblem(sys, H, arginit, IC=nothing, tspan = (0,20))
        state_variables = [1,1,1,1,1,1]

        prob = createProblem(sys,FORCE, init_struct, state_variables, (0,.1));

        
        A = calculate_symbolic_jacobian(sys);
        
        # x(t) θ1(t) θ2(t) dx(t) dθ1(t) dθ2(t) dθ1_t(t)

        state_params = []

        #[L1, L2, dθ1(t), dθ1ˍt(t), dθ2(t), dθ2ˍt(t), g, m1, m2, mc, θ1(t), θ2(t)        

        substitution_values = [1,1,1,1,1,1,1,1,1,1,1,1]

        # Evaluate the Jacobian
        A_evaluated = evaluate_jacobian(A, substitution_values)

        B = calculate_parametric_jacobian();
        # Penalize deviations in state variables
        Q = Diagonal([10.0, 1.0, 100.0, 1.0, 100.0, 1.0]) ; # Adjust weights as needed

        # Penalize control effort
        R = [1.0];  # Single input, so this is scalar
        A_evaluated = A_evaluated[1:6, 1:6]
        # println(A_evaluated)
        # println(length(A_evaluated))
        lqr_result = lqr(Discrete,A,B,Q,R)
        println(lqr_result)
        # A_numeric = substitute(A, subs_dict)
        # println(typeof(A))
        # println(typeof(A[end]))
        # println(A[end])
        # println(typeof(A[end][end]))
        # println(A[end][end])

        # print('\n')
        # print('\n')

        # println(A_evaluated);
        # println(B)
        # println(Q)
        # println(R)

    end

    lmao()
end