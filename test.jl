using ModelingToolkit
using Symbolics
begin
# Define independent variables, parameters, and states
@variables t
@parameters mc m1 m2 L1 L2 g F
@variables x(t) θ1(t) θ2(t) dx(t) dθ1(t) dθ2(t)

d = Differential(t)
eqs = [
    dx ~ d(x),
    dθ1 ~ d(θ1),
    dθ2 ~ d(θ2)
]

I1 = m1*L1^2
I2 = m2*L2^2
l1 = L1
l2 = L2

T1 = 1/2 * (mc + m1 + m2) * dx^2 
T2 = 1/2 * (m1*l1^2 + m2*L1^2 + I1) * dθ1^2 
T3 = 1/2 * (m2*l2^2 + I2) * dθ2^2 
T4 = (m1*l1 + m2*L1)*cos(θ1)*dx*dθ1 
T5 = m2*l2*cos(θ2)*dx*dθ2 
T6 = m2*L1*l2*cos(θ1 - θ2)*dθ1*dθ2 

T = T1 + T2 + T3 + T4 + T5 + T6
V = g*(m1*l1 + m2*L1)*cos(θ1) + m2*g*l2*cos(θ2)
L = T - V

EL = expand_derivatives(d(Differential(dx)(L))) - expand_derivatives(Differential(x)(L))
push!(eqs, EL ~ F)

for (q, dq) in [(θ1, dθ1), (θ2, dθ2)]
    EL = expand_derivatives(d(Differential(dq)(L))) - expand_derivatives(Differential(q)(L))
    push!(eqs, EL ~ 0)
end

@named HO = ODESystem(eqs, t, [x, θ1, θ2, dx, dθ1, dθ2], [mc, m1, m2, L1, L2, g, F])
HO = structural_simplify(HO)

param_values = (
    mc => 1.0,
    m1 => 0.5,
    m2 => 0.5,
    L1 => 1.0,
    L2 => 1.0,
    g  => 9.81,
    F  => 0.0 # will override dynamically
)

u0 = [x => 0.0, θ1 => 0.1, θ2 => -0.1, dx => 0.0, dθ1 => 0.0, dθ2 => 0.0]

# Create an ODE function from HO system
f_ode = ODEFunction(HO, expressions=Dict(param_values))

function choose_F(t)
    # Define a custom force that can vary over time.
    # Example: zero force until t=2, then apply F=1.0
    if t < 2.0
        return 0.0
    else
        return 1.0
    end
end

function step_integrator!(t, u, dt, f_ode, param_values)
    du = similar(u)
    current_params = merge(param_values, (F => choose_F(t)))
    f_ode(t, u, du, current_params)
    @. u = u + dt * du
    return t + dt, u
end

# Since statevars or states might not be defined in your version, 
# just manually specify the state variable order as was given in the ODESystem:
vs = [x, θ1, θ2, dx, dθ1, dθ2]

function my_own_integrator(u0, f_ode, param_values; tspan=(0.0,10.0), dt=0.01)
    t = tspan[1]
    tfinal = tspan[2]
    u_array = [u0[v] for v in vs]
    solution = [(t, copy(u_array))]

    while t < tfinal
        t, u_array = step_integrator!(t, u_array, dt, f_ode, param_values)
        push!(solution, (t, copy(u_array)))
    end

    return solution
end

sol = my_own_integrator(u0, f_ode, param_values, tspan=(0.0,5.0), dt=0.01)
# 'sol' now holds time and solution tuples.
# you got it kid?
end