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
    @parameters mc m1 m2 L1 L2 g

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


    function force(θ1, x, t)
        return rand()
    end
    force(1, 1, 1)
    println("adding force")
    EL = expand_derivatives(d(Differential(dx)(L))) - expand_derivatives(Differential(x)(L)) ~ force(θ1, x, t)
    push!(eqs, EL )

    # Add Euler-Lagrange equations
    for (q, dq) in [(θ1, dθ1), (θ2, dθ2)]
        EL = expand_derivatives(d(Differential(dq)(L))) - expand_derivatives(Differential(q)(L))
        push!(eqs, EL ~ 0)
    end
    @named HO = ODESystem(eqs, t, [x, θ1, θ2, dx, dθ1, dθ2], [mc, m1, m2, L1, L2, g])
    HO = structural_simplify(HO)
    # Initial conditions
    IC = [
        x => 0,
        θ1 => π-.1,
        θ2 => π,
        dx => 0,
        dθ1 => 0,
        dθ2 => 0,
    
    ]
    
    tspan = (0, 10)
    
#     function cont(t)
#         return sin(t)
#     end
    println("defining problem")
    prob = ODEProblem(HO, IC, tspan, [mc=>5, m1=>1, m2=>1, L1=>1, L2=>1, g=>-9.8])
    println("solving")
    @time solve(prob, ImplicitMidpoint(), dt=0.05)
    println("done")
 

end
function make_animation(sol; L1=1.0, L2=1.0)
    anim = @animate for i in 1:2:length(sol.t)
        t = sol.t[i]
        x_cart = sol[x][i]  # Directly extract x(t)
        θ1_pendulum = sol[θ1][i]  # Directly extract θ1(t)
        θ2_pendulum = sol[θ2][i]  # Directly extract θ2(t)

        # Cart position
        cart_x = x_cart
        cart_y = 0.0

        # First pendulum position
        pendulum1_x = cart_x + L1 * sin(θ1_pendulum)
        pendulum1_y = -L1 * cos(θ1_pendulum)

        # Second pendulum position
        pendulum2_x = pendulum1_x + L2 * sin(θ2_pendulum)
        pendulum2_y = pendulum1_y - L2 * cos(θ2_pendulum)

        # Plot setup
        plot([-5, 5], [0, 0], color=:black, label="", lw=2)  # Ground line
        scatter!([cart_x], [cart_y], color=:blue, label="Cart", ms=5)
        plot!([cart_x, pendulum1_x], [cart_y, pendulum1_y], label="Pendulum 1", lw=2, color=:red)
        scatter!([pendulum1_x], [pendulum1_y], color=:red, label="", ms=5)
        plot!([pendulum1_x, pendulum2_x], [pendulum1_y, pendulum2_y], label="Pendulum 2", lw=2, color=:green)
        scatter!([pendulum2_x], [pendulum2_y], color=:green, label="", ms=5)

        xlims!(-5, 5)
        ylims!(-5, 5)
        title!("Cart-Pendulum System at t = $(round(t, digits=2))")
    end
    gif(anim, "cart_pendulum_animation.gif", fps=120)
end

begin
    make_animation(sol)
end