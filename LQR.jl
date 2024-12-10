using ModelingToolkit, DifferentialEquations
using Plots
using Symbolics
using SciMLStructures:Tunable, replace, replace!
using SymbolicIndexingInterface: parameter_values, state_values
using ModelingToolkit: Nonnumeric, structural_parameters
using LinearAlgebra, ControlSystems
include("./DP.jl")

begin
    
    function linearize_system(sys, equilibrium)
        # Linearize the system around equilibrium (equilibrium is a vector of states [x, θ1, θ2, dx, dθ1, dθ2])
        A = zeros(6, 6)  # Jacobian of system wrt to state variables
        B = zeros(6, 1)  # Jacobian of system wrt to control input

        # Compute the Jacobian matrix A and B (to be filled with partial derivatives)
        # Example:
        # A = ∂f/∂x and B = ∂f/∂u (linearized state-space form)
        # Placeholder for Jacobian computation (to be replaced with actual derivatives)
        
        # Compute A and B for your system
        # This requires calculating the partial derivatives of the equations of motion
        # with respect to the states (position, angles, velocities) and inputs (force)

        return A, B
    end
    
    function lqr(A, B, Q, R)
        # Solve for optimal gain matrix K using the continuous-time algebraic Riccati equation (CARE)
        P = sympy.solve(LinearAlgebra.AffineFunction([A, B, Q, R]), P)
        K = inv(R) * B' * P
        return K
    end
    
    function createProblem(sys, H, arginit)
        tspan = (0, 20)
        IC = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ps = [sys.mc => 100, sys.m1 => 1, sys.m2 => 1, sys.L1 => 1, sys.L2 => 1, sys.g => -9.8, sys.H => H, sys.ARGS => arginit]
        prob = ODEProblem(sys, IC, tspan, ps)
        return prob
    end

    # Example system with linearization and LQR
    sys = createSystem(force_arg)
    init_struct = force_arg(0, 0)
    prob = createProblem(sys, FORCE, init_struct)

    # Linearize around an equilibrium point (example)
    equilibrium = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    A, B = linearize_system(sys, equilibrium)

    # Define Q and R matrices for LQR
    Q = Diagonal([10, 10, 10, 1, 1, 1])  # State weighting matrix
    R = Diagonal([0.1])  # Control weighting matrix

    # Compute the optimal gain matrix K
    K = lqr(A, B, Q, R)

    # Control law: u = -K * x
    # Example control input for simulation (to be used in the loss function or solver)
    # u = -K * x
    @time sol = loss(prob, [1, 1, 1, -9.8, 1000, 1], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    plot(sol.t, transpose(sol[1:3, :]))
end
