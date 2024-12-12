using Distributions
using JLD2
using Flux
using Flux: Adam
using Zygote
using StatsBase
using ProgressMeter
using ModelingToolkit, DifferentialEquations
include("./DP.jl")
using Random

function sample_action(policy_net, s)
    p = policy_net(s)
    mean_ = clamp(p[1], -1.0, 1.0)
    log_std = clamp(p[2], -2.0, 1.0)
    std_ = exp(log_std)

    a_raw = rand(Normal(mean_, std_))
    a_scaled = tanh(a_raw)
    return a_scaled, mean_, std_
end

function logprob_action(a_scaled, mean_, std_)
    raw_action = atanh(a_scaled)  # Undo the tanh transformation
    base_log_prob = logpdf(Normal(mean_, std_), raw_action)
    correction = log(1 - tanh(raw_action)^2 + 1e-6)  # Add stability term
    return base_log_prob - correction
end



function collect_trajectory(prob, params, init_struct, policy_net, value_net, reward_func; dt = 0.01, max_t=20, max_x=10.0)
    # Initialize Random Initial Conditions
    # x = 0.0
    # v = 0.0
    # θ1 = π + (rand() * (π / 18) - π / 36)
    # θ2 = π + (rand() * (π / 18) - π / 36)
    # ω1 = 0.0
    # ω2 = 0.0

    x = rand(-10.0:0.1:10.0)
    v = rand(-5.0:0.1:5.0)
    θ1 = rand(0.0:0.01:2π)
    θ2 = rand(0.0:0.01:2π)
    ω1 = rand(-10.0:0.1:10.0)
    ω2 = rand(-10.0:0.1:10.0)

    ICs = [x, θ1, θ2, ω1, ω2, v, 0, 0]
    println("Initial Conditions: $ICs")

    # Update Problem Parameters
    ps = parameter_values(prob)
    ps = replace(Tunable(), ps, params)
    newprob = remake(prob; u0=ICs, p=ps)

    integrator = init(newprob, ImplicitMidpoint(), dt=dt)

    # Initialize Storage Variables
    times = Float64[]
    states = Vector{Vector{Float64}}()
    actions = Float64[]
    rewards = Float64[]
    log_probs = Float64[]
    dones = Bool[]

    t_curr = 0.0
    while t_curr <= max_t
        state = copy(integrator.u[1:6])

        # println("<-----STATE------>")
        # println(state)

        # normalized_state = [
        #     2 * state[1] / max_x,
        #     2 * state[2] / (π / 18),
        #     2 * state[3] / (π / 18),
        #     2 * state[4] / 1.0,
        #     2 * state[5] / 1.0,
        #     2 * state[6] / 1.0
        # ]
        push!(states, state)
        push!(times, integrator.t)

        # Sample Action
        a, mean_, std_ = sample_action(policy_net, state)
        lp = logprob_action(a, mean_, std_)
        a_scaled = a * 10
        push!(actions, copy(a))
        push!(log_probs, lp)

        # Apply Action to the Environment
        init_struct.a = a_scaled
        reward = reward_func(state, a)
        push!(rewards, reward)

        # Check for Termination Conditions (Only Cart Position)
        x, θ1, θ2, ω1, ω2, v = state
        println("x cart position: $x")
        println("reward: $reward")
        if abs(x) > max_x # || abs(θ1 - π) > π / 18 || abs(θ2 - π) > π / 18
            push!(dones, true)
            break
        else
            push!(dones, false)
        end

        # Step the Environment
        step!(integrator, dt, true)
        t_curr += dt
    end

    T = length(rewards)
    if T == 0
        return Float64[], Float64[], Float64[], Float64[], Float64[], Float64[], Bool[], Float64[]
    end

    values = [value_net(s)[1] for s in states]
    println("Values: $values")
    final_value = (last(dones)) ? 0.0 : values[end]

    return states, actions, rewards, log_probs, values, final_value, dones, times
end


# Compute advantages and returns using GAE
function gae(rewards, values, final_value, dones; γ=0.99, λ=0.95)
    T = length(rewards)
    advantages = zeros(T)
    deltas = zeros(T)

    for t in 1:T
        # If done, the next value is 0.0; else, use the next value
        v_next = (t < T) ? (dones[t] ? 0.0 : values[t + 1]) : final_value
        deltas[t] = rewards[t] + γ * v_next - values[t]
    end

    # Compute advantages using GAE
    gae_adv = 0.0
    for t in reverse(1:T)
        gae_adv = deltas[t] + γ * λ * (dones[t] ? 0.0 : gae_adv)
        advantages[t] = gae_adv
    end

    returns = advantages .+ values
    return advantages, returns
end


# PPO loss function
function ppo_loss(policy_net, value_net, states, actions, old_log_probs, advantages, returns; ε=0.2, c1=1.0, c2=0.1)
    # Compute new log_probs and values without mutation
    log_probs_new = map((s, a) -> begin
        p = policy_net(s)
        mean_ = p[1]
        log_std = p[2]  # Extract scalar log_std
        std_ = exp(log_std)
        logprob_action(a, mean_, std_)
    end, states, actions)

    # println("Variance of log_probs_new: $(var(log_probs_new))")
    # println("Variance of old_log_probs: $(var(old_log_probs))")


    values_new = map(s -> value_net(s)[1], states)

    # Compute ratios
    ratios = exp.(log_probs_new .- old_log_probs)
    obj1 = ratios .* advantages
    obj2 = clamp.(ratios, 1.0 - ε, 1.0 + ε) .* advantages
    policy_loss = -mean(min.(obj1, obj2))

    # Compute value loss
    value_loss = mean((values_new .- returns).^2)

    # Entropy calculation
    entropy_sum = sum(map(s -> begin
        p = policy_net(s)
        log_std = p[2]  # Extract scalar log_std
        0.5 * (1 + log(2π)) + log_std
    end, states))

    entropy = entropy_sum / length(states)

    # Total loss
    total_loss = policy_loss + c1 * value_loss - c2 * entropy
    return total_loss, policy_loss, value_loss, entropy
end


mutable struct force_arg 
    a::Float64
end

function runPPO(p, reward_func)

    policy_network = Chain(
        Dense(6, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 2)  # mean and log_std
    )

    value_network = Chain(
        Dense(6, 64, relu),
        Dense(64, 64, relu),
        Dense(64, 1)
    )

    policy_network = f64(policy_network)
    value_network = f64(value_network)

    function FORCE(x, θ1, θ2, v, ω1, ω2, ARGS)
        # println("a in force func: $a")
        return ARGS.a
    end

    params = p

    sys = createSystem(force_arg)
    init_struct = force_arg(0)
    prob = createProblem(sys, FORCE, init_struct)

    policy_opt = Adam(1e-4)
    value_opt = Adam(1e-4)

    nepochs = 10

    iterations = 1

    for iteration in 1:iterations
        states, actions, rewards, old_log_probs, values, final_value, dones, _ = collect_trajectory(prob, params, init_struct, policy_network, value_network, reward_func)
        adv, ret = gae(rewards, values, final_value, dones)
        adv = (adv .- mean(adv)) ./ (std(adv) + 1e-8)

        idxs = 1:length(states)
        batch_size = length(states)

        total_loss_value = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        @showprogress for epoch in 1:nepochs
            for batch_start in 1:batch_size:length(states)
                batch_end = min(batch_start + batch_size - 1, length(states))
                batch_idx = idxs[batch_start:batch_end]

                bs = states[batch_idx]
                ba = actions[batch_idx]
                bolp = old_log_probs[batch_idx]
                badv = adv[batch_idx]
                bret = ret[batch_idx]

                total_loss, policy_loss, value_loss, entropy = ppo_loss(policy_network, value_network, bs, ba, bolp, badv, bret)

                gs_policy = Zygote.gradient(() -> policy_loss, Flux.params(policy_network))
                gs_value = Zygote.gradient(() -> value_loss, Flux.params(value_network))

                println("Policy Gradient Norms: ", [norm(g) for g in values(gs_policy) if g !== nothing])

                total_loss_value += sum(total_loss)
                total_policy_loss += sum(policy_loss)
                total_value_loss += sum(value_loss)
                total_entropy += sum(entropy)
                n_batches += 1

                Flux.Optimise.update!(policy_opt, Flux.params(policy_network), gs_policy)
                Flux.Optimise.update!(value_opt, Flux.params(value_network), gs_value)
            end
        end
        avg_loss = total_loss_value / n_batches
        avg_policy_loss = total_policy_loss / n_batches
        avg_value_loss = total_value_loss / n_batches
        avg_entropy = total_entropy / n_batches

        println("Iteration $iteration:")
        println("  Average Loss: $avg_loss")
        println("  Average Policy Loss: $avg_policy_loss")
        println("  Average Value Loss: $avg_value_loss")
        println("  Average Entropy: $avg_entropy")
    end

    @save "mymodel.jld2" policy_network
    println("Policy network saved to trained_policy.jld2")
end

function run_experiment()
    p = [1, 1, 1, -9.8, 1,1] # m2, m1, L1, g, mc, L2

    function reward_func(state, action)
        x, θ1, θ2, ω1, ω2, v = state
        return -abs(x)/10  # Reward for staying near the center
    end
    
    runPPO(p, reward_func)
end


function test_model()
    @load "mymodel.jld2" policy_network

    policy_network = f64(policy_network)

    function FORCE(x, θ1, θ2, v, ω1, ω2, ARGS)
        return ARGS.a
    end

    params = [1, 1, 1, -9.8, 1, 1]

    sys = createSystem(force_arg)
    init_struct = force_arg(0)
    prob = createProblem(sys, FORCE, init_struct)

    # x = 0.0
    # v = 0.0
    # θ1 = π + (rand() * (π / 18) - π / 36)
    # θ2 = π + (rand() * (π / 18) - π / 36)

    # ω1 = 0.0
    # ω2 = 0.0

    x = rand(-10.0:0.1:10.0)
    v = rand(-5.0:0.1:5.0)
    θ1 = rand(0.0:0.01:2π)
    θ2 = rand(0.0:0.01:2π)
    ω1 = rand(-10.0:0.1:10.0)
    ω2 = rand(-10.0:0.1:10.0)

    ICs = [x, θ1, θ2, ω1, ω2, v, 0, 0]

    println("Initial Conditions: $ICs")

    # Update Problem Parameters
    ps = parameter_values(prob)
    ps = replace(Tunable(), ps, params)
    newprob = remake(prob; u0=ICs, p=ps)

    dt = 0.01

    integrator = init(newprob, ImplicitMidpoint(), dt=dt)

    times = Float64[]
    states = Vector{Vector{Float64}}()
    actions = []

    t_curr = 0.0
    max_t = 20.0
    max_x = 10.0
    while t_curr <= max_t
        state = copy(integrator.u[1:6])
        push!(times, integrator.t)
        push!(states, state)

        a, _, _ = sample_action(policy_network, state)


        init_struct.a = a

        push!(actions, a)

        x, θ1, θ2, ω1, ω2, v = state
        if abs(x) > max_x # || abs(θ1 - π) > π / 18 || abs(θ2 - π) > π / 18
            break
        end

        step!(integrator, dt, true)
        t_curr += dt
    end

    # Extract x, θ1, θ2 from states
    x_vals = map(s -> s[1], states)
    θ1_vals = map(s -> s[2], states)
    θ2_vals = map(s -> s[3], states)

    plt = plot(
        times, [x_vals θ1_vals θ2_vals],
        label=["x" "θ₁" "θ₂"],
        xlabel="Time (s)",
        ylabel="State Values",
        title="Trajectory of Double Cart-Pole",
        color=[:blue :green :red],
    )

    plot!([times[1], times[end]], [π, π], label="π", linestyle=:dash, color=:black)

    display(plt)

    plot(
        times, actions,
        xlabel="Time (s)",
        ylabel="Action Value",
        title="Actions over time",
    )

end

run_experiment()
# test_model()

