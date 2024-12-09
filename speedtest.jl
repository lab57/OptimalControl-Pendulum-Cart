using DifferentialEquations
using Plots

l = 1.0                             # length [m]
m = 1.0                             # mass [kg]
g = 9.81                            # gravitational acceleration [m/s²]


# 1, x
# 2, θ1 
# 3, θ2 
# 4, v
# 5, ω1 
# 6, ω2


function pendulum!(du, u, p, t)
    du[1] = u[4]                    # θ'(t) = ω(t)
    du[2] = u[5]
    du[3] = u[6]
    du[4] 
    du[5]
    du[6]
    du[7]
    du[8]



    du[2] = -3g / (2l) * sin(u[1]) + 3 / (m * l^2) * p(t) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
end

θ₀ = 0.01                           # initial angular deflection [rad]
ω₀ = 0.0                            # initial angular velocity [rad/s]
u₀ = [θ₀, ω₀]                       # initial state vector
tspan = (0.0, 10.0)                  # time interval

M = t -> 0.1sin(t)                    # external torque [Nm]

prob = ODEProblem(pendulum!, u₀, tspan, M)
sol = solve(prob)

@time prob4= remake(prob, u0=[.234234, .01])
@time sol = solve(prob4)