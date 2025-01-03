using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics
using SciMLStructures:Tunable, replace, replace!
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: Nonnumeric
using SciMLSensitivity

begin
    function createSystem( H_ARG::DataType)
        @independent_variables t
        # Define states and their derivatives explicitly
        @variables x(t) θ1(t) θ2(t) dx(t) dθ1(t) dθ2(t)
        @parameters  mc m1 m2 L1 L2 g H(::Real, ::Real, ::Real, ::Real, ::Real, ::Real, ::H_ARG) ARGS::H_ARG
        d = Differential(t) #d/dt


        # Define first-order equations
        eqs = [
            dx ~ d(x),
            dθ1 ~ d(θ1),
            dθ2 ~ d(θ2),
        ]
        I1 = m1*L1^2
        I2 = m2*L2^2
        l1 = L1
        l2 = L2
        
        # Use dx, dθ1, dθ2 instead of ẋ, ω1, ω2
        T1 = 1/2 * (mc + m1 + m2) * dx^2 
        T2 = 1/2 * (m1*l1^2 + m2*L1^2 + I1) * dθ1^2 
        T3 = 1/2 * (m2*l2^2 + I2) * dθ2^2 
        
        T4 = (m1*l1 + m2*L1) * cos(θ1) * dx * dθ1 
        T5 = m2*l2 * cos(θ2) * dx * dθ2 
        T6 = m2*L1*l2 * cos(θ1 - θ2) * dθ1 * dθ2 
        
        T = T1 + T2 + T3 + T4 + T5 + T6
        V = g*(m1*l1 + m2*L1)*cos(θ1) + m2*g*l2*cos(θ2)
        L = T - V
        EL = expand_derivatives(d(Differential(dx)(L))) - expand_derivatives(Differential(x)(L)) ~ H(x, θ1, θ2, dx, dθ1, dθ2, ARGS)  # force(θ1, x, t)
        push!(eqs, EL )
        EL = expand_derivatives(d(Differential(dθ1)(L))) - expand_derivatives(Differential(θ1)(L)) ~ 0#H(θ1, θ2, t)  # force(θ1, x, t)
        push!(eqs, EL )
        EL = expand_derivatives(d(Differential(dθ2)(L))) - expand_derivatives(Differential(θ2)(L)) ~ 0#H(θ1, θ2, t)  # force(θ1, x, t)
        push!(eqs, EL )

        @named HO = ODESystem(eqs, t, [x, θ1, θ2, dx, dθ1, dθ2], [mc, m1, m2, L1, L2, g, H, ARGS])
        HO = structural_simplify(HO)
        for e in equations(HO)
            # println(e)
        end
        return HO
    end


    function createProblem(sys, H, arginit, tspan = (0,20))
        # IC = [sys.x=>.2, sys.θ1=>.1, sys.θ2=>.1, sys.dx=>0, sys.dθ1=>0, sys.dθ2=>0]
        IC = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0]
        ps = [sys.mc=>100, sys.m1=> 1, sys.m2 => 1, sys.L1 => 1, sys.L2=>1, sys.g=>-9.8, sys.H => H, sys.ARGS=>arginit]
        prob = ODEProblem(sys, IC, tspan, ps)
    
        return prob 
    end
    
    function loss(prob,x, ICs)
        ps = parameter_values(prob)
        # println(ps[1])
        ps = replace(Tunable(), ps, x)
        newprob = remake(prob; u0=ICs, p=ps )
        sol = solve(newprob,ImplicitMidpoint(),dt=0.01)
        return sol
    end

    mutable struct force_arg_temp3
        a::Real
    end

    function example()
        
        function FORCE(x, θ1, θ2, v, ω1, ω2, ARGS)
            # println("Called")
            return ARGS.a
        end

        sys = createSystem(force_arg_temp3)
        init_struct = force_arg_temp3(0)
        prob = createProblem(sys,FORCE, init_struct)

        ICs = [0, π/2, π, 0, 0, 0, 0, 0]


        # println("Initial Conditions: $ICs")
        params = [1, 1, 1, -9.8, 1, 1]

        # Update Problem Parameters
        ps = parameter_values(prob)
        ps = replace(Tunable(), ps, params)
        newprob = remake(prob; u0=ICs, p=ps)


        dt = 0.01
        integrator = init(newprob, ImplicitMidpoint(), dt=dt)

        times = Float64[]
        states = Vector{Vector{Float64}}()
        jacobians = []

        t_curr = 0.0
        max_t = 20.0
        while t_curr <= max_t
            state = copy(integrator.u[1:6])
            push!(times, t_curr)
            push!(states, state)

            force_arg_temp3.a = 5.0
            step!(integrator, dt, true)
            
            t_curr += dt
        end

        states_matrix = hcat(states...)
        plot(times, states_matrix[1:3, :]')

    end
    # example()
end