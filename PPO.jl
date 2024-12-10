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
    mean_ = p[1:1]
    log_std = p[2:end]
    std_ = exp.(log_std)
    dists = Normal.(mean_, std_)
    a = [rand(d) for d in dists][1]
    return a, mean_, std_
end


function logprob_action(a, mean_, std_)
    log_probs = @. -0.5 * ((a - mean_)^2 / (std_^2) + 2*log(std_) + log(2π))
    return sum(log_probs)
end


function collect_trajectory(prob, params, init_struct, policy_net, value_net, reward_func; dt = 0.001, max_t=20)

    x = rand() * 4.8 - 2.4             # uniform in [-1, 1]
    θ1 = (rand() - 0.5) * π            # uniform in [-π/2, π/2]
    θ2 = (rand() - 0.5) * π            # uniform in [-π/2, π/2]
    ω1 = rand() * 2.0 - 1.0            # uniform in [-1, 1]
    ω2 = rand() * 2.0 - 1.0            # uniform in [-1, 1]
    v  = rand() * 1.0 - 0.5            # uniform in [-0.5, 0.5]

    ICs = [clamp(x, -1, 1), clamp(θ1, -π/2, π/2), clamp(θ2, -π/2, π/2), ω1, ω2, v, 0, 0]
    println(ICs)

    ps = parameter_values(prob)
    ps = replace(Tunable(), ps, params)
    newprob = remake(prob; u0=ICs, p=ps)

    integrator = init(newprob, ImplicitMidpoint(), dt=dt)

    times = []
    states = []
    actions = []
    rewards = []
    log_probs = []

    t_curr = 0.0
    while t_curr < max_t
        # println(integrator.t)
        state = copy(integrator.u[1:6])
        push!(times, integrator.t)
        push!(states, state)
        a, mean_, std_ = sample_action(policy_net, state)
        # println("a: $a")
        lp = logprob_action(a, mean_, std_)
        push!(actions, copy(a))
        push!(log_probs, lp)
        init_struct.a = a
        push!(rewards, reward_func(state, a))

        step!(integrator, dt, true)
        t_curr += dt
    end

    values = [value_net(s)[1] for s in states]
    final_value = values[end]

    # sol_matrix = hcat(states...)
    # plt = plot(times, sol_matrix[1:3, :]', label=["x" "θ1" "θ2"])
    # display(plt)

    return states, actions, rewards, log_probs, values, final_value
end

# Compute advantages and returns using GAE
function gae(rewards, values, final_value; γ=0.99, λ=0.95)
    T = length(rewards)
    advantages = zeros(T)
    deltas = zeros(T)
    
    dones = falses(T)
    dones[end] = true

    # Compute deltas
    for t in 1:T
        # If dones[t] is true at step t, we set v_next = 0.0 to avoid bootstrapping beyond episode boundary.
        v_next = (t < T) ? (dones[t] ? 0.0 : values[t+1]) : final_value
        deltas[t] = rewards[t] + γ * v_next - values[t]
    end

    # Compute advantages using GAE
    gae_adv = 0.0
    for t in reverse(1:T)
        # If dones[t] is true, reset the advantage accumulation.
        gae_adv = deltas[t] + γ * λ * (dones[t] ? 0.0 : gae_adv)
        advantages[t] = gae_adv
    end

    returns = advantages .+ values
    return advantages, returns
end


# PPO loss function
function ppo_loss(policy_net, value_net, states, actions, old_log_probs, advantages, returns; ε=0.2, c1=1.0, c2=0.01)
    # Compute new log_probs and values without mutation
    log_probs_new = map((s, a) -> begin
        p = policy_net(s)
        mean_ = p[1]
        log_std = p[2]  # Extract scalar log_std
        std_ = exp(log_std)
        logprob_action(a, mean_, std_)
    end, states, actions)

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
        a = ARGS.a
        # println("a in force func: $a")
        return ARGS.a
    end

    params = p

    sys = createSystem(force_arg)
    init_struct = force_arg(0)
    prob = createProblem(sys, FORCE, init_struct)

    policy_opt = Adam(1e-3)
    value_opt = Adam(1e-3)

    nepochs = 10
    batch_size = 64
    iterations = 1000

    for iteration in 1:iterations
        states, actions, rewards, old_log_probs, values, final_value = collect_trajectory(prob, params, init_struct, policy_network, value_network, reward_func)
        adv, ret = gae(rewards, values, final_value)
        adv = (adv .- mean(adv)) ./ (std(adv) + 1e-8)

        idxs = shuffle(collect(1:length(states)))

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

                gs = Zygote.gradient(() -> begin
                    total_loss
                end, Flux.params(policy_network, value_network))

                # Ensure losses are scalar before accumulation
                total_loss_value += sum(total_loss)
                total_policy_loss += sum(policy_loss)
                total_value_loss += sum(value_loss)
                total_entropy += sum(entropy)
                n_batches += 1

                Flux.Optimise.update!(policy_opt, Flux.params(policy_network), gs)
                Flux.Optimise.update!(value_opt, Flux.params(value_network), gs)
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
    @save "trained_policy.jld2" policy_network
    println("Policy network saved to trained_policy.jld2")
end

begin
    p = [1, 1, 1, -9.8, 5,1] # m2, m1, L1, g, mc, L2

    function reward_func(state, action)
        x, θ1, θ2, ω1, ω2, v = state
    
        k_angle = 10.0      # Penalty for pole angles
        k_position = 1.0    # Penalty for cart position
        k_velocity = 0.1    # Penalty for cart and angular velocities
        boundary_penalty = -100.0  # Large penalty for boundary violations
        max_x = 10.0         # Maximum allowed cart position (e.g., 2.4 for standard cart-pole)
    
        angle_penalty = k_angle * (θ1^2 + θ2^2)
    
        position_penalty = k_position * x^2
    
        velocity_penalty = k_velocity * (v^2 + ω1^2 + ω2^2)
    
        if abs(x) > max_x || abs(θ1) > π/2 || abs(θ2) > π/2
            return boundary_penalty
        end
    
        reward = -(angle_penalty + position_penalty + velocity_penalty)
        return reward
    end
    
    runPPO(p, reward_func)
end