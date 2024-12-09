using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics

mutable struct state
    x::Float64,
    θ1::Float64,
    θ2::Float64,
    v::Float64,
    ω1::Float64,
    ω2::Float64
end


begin
    @independent_variables t
    @parameters mc m1 m2 L1 L2 g F(t)
    
    # Define states and their derivatives explicitly
    @variables x(t) θ1(t) θ2(t) dx(t) dθ1(t) dθ2(t) 
    xs = [x, θ1, θ2] #all motion variables 
    d = Differential(t) #d/dt
    ω1 = d(θ1)
    ω2 = d(θ2)
    ẋ  = d(x)


    # Define first-order equations
    eqs = [
        dx ~ d(x),
        dθ1 ~ d(θ1),
        dθ2 ~ d(θ2)
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

    println(T1)
    println(T2)
    println(T3)


    EL = expand_derivatives(d(Differential(dx)(L))) - expand_derivatives(Differential(x)(L))
    push!(eqs, EL ~ F)

    # Add Euler-Lagrange equations
    for (q, dq) in [(θ1, dθ1), (θ2, dθ2)]
        EL = expand_derivatives(d(Differential(dq)(L))) - expand_derivatives(Differential(q)(L))
        push!(eqs, EL ~ 0)
    end

    @named HO = ODESystem(eqs, t, [x, θ1, θ2, dx, dθ1, dθ2], [mc, m1, m2, L1, L2, g, F])
    HO = structural_simplify(HO)
    print(eqs)
    # function control_affect!(integrator)
    #     state = integrator.u
    #     force = CONTROL(state)
    #     integrator.p[F] = force  # Assuming F is last parameter
    #     println("t: $(integrator.t), F: $force")  # To verify it's running

    # end
    # force_update = [x ~ x] => (control_affect!, xs, [mc, m1, m2, L1, L2, g, F], [], nothing)

    function affect!(integ)
        # println(integ.t)
        # println(integ.u)
        println("affect")
    end
    # updatef = [x ~ 1] => (affect!, [x, θ1, θ2], [F], [F], nothing)

    @mtkbuild control_system = ODESystem(eqs, t, [x, θ1, θ2, dx, dθ1, dθ2], [mc, m1, m2, L1, L2, g, F])

    # Initial conditions
    IC = [
        x => 0,
        θ1 => 1,
        θ2 => 0,
        dx => 0,
        dθ1 => 0,
        dθ2 => 0,
        

    ]
    
    tspan = (0, 20)
    
    function cont(t)
        return sin(t)
    end

    prob = ODEProblem(control_system, IC, tspan, [mc=>1, m1=>1, m2=>1, L1=>1, L2=>1, g=>-9.8, F=>0])
    sol = solve(prob, Rodas5P())
    plot(transpose(sol[1:3, :]))
end


begin


    function s()
        IC = [
            x => 0,
            θ1 => 0.053,
            θ2 => π,
            dx => 0,
            dθ1 => 0,
            dθ2 => 0
        ]

        M = [mc=>5,
        m1=>1, m2=>1, L1=>1, L2=>1, g=>9.8]

        tspan = (0, 30)
        println("Defining problem")
        prob = ODEProblem(HO, IC, tspan, M)
        println(prob)
        println("solving")
        sol = solve(prob, ImplicitMidpoint(), dt=0.01)
        println("done")
        return sol
    end
    
    sol = s()
    println("plotting")
    plot(sol.t, transpose(sol[1:3, :]))
end


