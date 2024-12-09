using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics
using SciMLStructures:Tunable, replace, replace!
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: Nonnumeric

begin

    function createSystem()
        @independent_variables t
        # Define states and their derivatives explicitly
        @variables x(t) θ1(t) θ2(t) dx(t) dθ1(t) dθ2(t)
        @parameters  mc m1 m2 L1 L2 g H(::Real, ::Real, ::Real)
        xs = [x, θ1, θ2] #all motion variables 
        d = Differential(t) #d/dt
        ω1 = d(θ1)
        ω2 = d(θ2)
        ẋ  = d(x)


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

        EL = expand_derivatives(d(Differential(dx)(L))) - expand_derivatives(Differential(x)(L)) ~ H(θ1, θ2, t)  # force(θ1, x, t)
        push!(eqs, EL )
        EL = expand_derivatives(d(Differential(dθ1)(L))) - expand_derivatives(Differential(θ1)(L)) ~ 0#H(θ1, θ2, t)  # force(θ1, x, t)
        push!(eqs, EL )
        EL = expand_derivatives(d(Differential(dθ2)(L))) - expand_derivatives(Differential(θ2)(L)) ~ 0#H(θ1, θ2, t)  # force(θ1, x, t)
        push!(eqs, EL )

        @named HO = ODESystem(eqs, t, [x, θ1, θ2, dx, dθ1, dθ2], [mc, m1, m2, L1, L2, g, H])
        HO = structural_simplify(HO)
        for e in equations(HO)
            println(e)
        end
        return HO
    end


    function createProblem(sys)
        tspan = (0, 20) 
        # IC = [sys.x=>.2, sys.θ1=>.1, sys.θ2=>.1, sys.dx=>0, sys.dθ1=>0, sys.dθ2=>0]
        IC = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0]
        function null(t, a, b)
            return 0
        end
        ps = [sys.mc=>100, sys.m1=> 1, sys.m2 => 1, sys.L1 => 1, sys.L2=>1, sys.g=>-9.8, sys.H=>null]
        prob = ODEProblem(sys, IC, tspan, ps)
    
        return prob 
    end
    
    function loss(prob,x, ICs)
        ps = parameter_values(prob)
       
        ps = replace(Tunable(), ps, x[1:end-1])
        ps = replace(Nonnumeric(), ps, [x[end]])
        println("aaa")
        println(ps[1])
        newprob = remake(prob; u0=ICs, p=ps )
        # println("a")
        # println(newprob.u0)
        sol = solve(newprob,ImplicitMidpoint(),dt=0.05)
        return sol
    end
    
    println("creating sys")
    sys = createSystem()
    println("Creating problem")
    prob = createProblem(sys)
    println("solving")
    sol = solve(prob, ImplicitMidpoint(), dt=0.05) 
    # plot(sol.t, transpose(sol[1:3, :]))


        # m2, m1, L1, g, mc, L2
        # x, theta1, theta2, omega2, omega1, v, 0, 0 (last two must be zero, dont ask)
    function Force(a, b, c)
        return 0.01323
    end
    Force(1, 1, 1)
    props =  [2, 1, 1, -9.8, 50,1 ,Force ]
    IC = [1.23343, .5, .3, 0, 0, 0,0,0]
    println(ps2)
    println("Loss")
    @time sol = loss(prob, props, IC)
    plot(sol.t, transpose(sol[1:3, :]))

    function Force(a, b, c)
        return 0.232
    end
    Force(1, 1, 1)
    props =  [2, 1, 1, -9.8, 50,1 ,Force ]
    IC = [1.23343, .5, .3, 0, 0, 0,0,0]
    println(ps2)
    println("Loss")
    @time sol = loss(prob, props, IC)
    plot(sol.t, transpose(sol[1:3, :]))
    # println(sol[3, :])





end

