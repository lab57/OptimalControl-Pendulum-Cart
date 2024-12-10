using Flux
using Flux: Adam
using Zygote
using Distributions
using StatsBase
using ProgressMeter

env = MyDoubleCartPoleEnv() # Here is whre we will import the environment
ns = length(state(env))       # number of state variables


policy_network = Chain(
    Dense(ns, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 2)  # mean and log_std
)

value_network = Chain(
    Dense(ns, 64, relu),
    Dense(64, 64, relu),
    Dense(64, 1)
)

policy_network = f64(policy_network)
value_network = f64(value_network)


function sample_action(policy_net, s)
    p = policy_net(s)
    mean_ = p[1]
    log_std = p[1:end]
    std_ = exp.(log_std)
    dists = Normal.(mean_, std_)
    a = [rand(d) for d in dists]
    return a, mean_, std_
end

function logprob_action(a, mean_, std_)
    log_probs = @. -0.5 * ((a - mean_)^2 / (std_^2) + 2*log(std_) + log(2π))
    return sum(log_probs)
end

# This will depend on the environment

# function collect_trajectory(env, policy_net, value_net; max_steps=2048)
#     states = Vector{Vector{Float64}}()
#     actions = Vector{Vector{Float64}}()
#     rewards = Float64[]
#     dones = Bool[]
#     log_probs = Float64[]

#     reset!(env)
#     s = state(env)
#     for step in 1:max_steps
#         a, mean_, std_ = sample_action(policy_net, s)
#         lp = logprob_action(a, mean_, std_)

#         r = step!(env, a)
#         s_next = state(env)
#         done = is_terminated(env)

#         push!(states, copy(s))
#         push!(actions, copy(a))
#         push!(rewards, r)
#         push!(dones, done)
#         push!(log_probs, lp)

#         s = s_next
#         if done
#             reset!(env)
#             s = state(env)
#         end
#     end

#     values = [value_net(s)[1] for s in states]
#     s_final = state(env)
#     final_value = value_net(s_final)[1]

#     return states, actions, rewards, dones, log_probs, values, final_value
# end

# Compute advantages and returns using GAE
function gae(rewards, values, dones, final_value; γ=0.99, λ=0.95)
    T = length(rewards)
    advantages = zeros(T)
    deltas = zeros(T)

    for t in 1:T
        v_next = (t < T) ? (dones[t] ? 0.0 : values[t+1]) : final_value
        deltas[t] = rewards[t] + γ * v_next - values[t]
    end

    gae_adv = 0.0
    for t in reverse(1:T)
        gae_adv = deltas[t] + γ * λ * (dones[t] ? 0.0 : gae_adv)
        advantages[t] = gae_adv
    end

    returns = advantages .+ values
    return advantages, returns
end

# PPO loss function
function ppo_loss(policy_net, value_net, states, actions, old_log_probs, advantages, returns; ε=0.2, c1=1.0, c2=0.01)
    log_probs_new = similar(old_log_probs)
    values_new = similar(returns)

    for (i, (s, a)) in enumerate(zip(states, actions))
        p = policy_net(s)
        mean_ = p[1]
        log_std = p[1:end]
        std_ = exp.(log_std)
        lp = logprob_action(a, mean_, std_)
        log_probs_new[i] = lp
        values_new[i] = value_net(s)[1]
    end

    ratios = exp.(log_probs_new .- old_log_probs)
    obj1 = ratios .* advantages
    obj2 = clamp.(ratios, 1.0 - ε, 1.0 + ε) .* advantages
    policy_loss = -mean(min.(obj1, obj2))

    value_loss = mean((values_new .- returns).^2)

    # Entropy of the Gaussian policy
    entropy_sum = 0.0
    for s in states
        p = policy_net(s)
        log_std = p[1:end]
        # Entropy for each dimension: H = 0.5*(1 + log(2π)) + log_std
        H_i = sum(0.5 * (1 + log(2π)) + log_std)
        entropy_sum += H_i
    end
    entropy = entropy_sum / length(states)

    total_loss = policy_loss + c1 * value_loss - c2 * entropy
    return total_loss, policy_loss, value_loss, entropy
end


policy_opt = Adam(1e-3)
value_opt = Adam(1e-3)

nepochs = 10
batch_size = 64
iterations = 1000

for iteration in 1:iterations
    states, actions, rewards, dones, old_log_probs, values, final_value = collect_trajectory(env, policy_network, value_network; max_steps=2048)
    adv, ret = gae(rewards, values, dones, final_value)
    adv = (adv .- mean(adv)) ./ (std(adv) + 1e-8)

    idxs = shuffle(collect(1:length(states)))

    @showprogress for epoch in 1:nepochs
        for batch_start in 1:batch_size:length(states)
            batch_end = min(batch_start + batch_size - 1, length(states))
            batch_idx = idxs[batch_start:batch_end]

            bs = states[batch_idx]
            ba = actions[batch_idx]
            bolp = old_log_probs[batch_idx]
            badv = adv[batch_idx]
            bret = ret[batch_idx]

            gs = Zygote.gradient(() -> begin
                loss, pl, vl, ent = ppo_loss(policy_network, value_network, bs, ba, bolp, badv, bret)
                loss
            end, Flux.params(policy_network, value_network))

            Flux.Optimise.update!(policy_opt, Flux.params(policy_network), gs)
            Flux.Optimise.update!(value_opt, Flux.params(value_network), gs)
        end
    end

    println("Iteration: $iteration completed.")
end
