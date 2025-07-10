module RLUtilities

export SMDPTDLearner, jump_time
export StopWhenValueAndPolicyReaches, StepsEpisodesPerExperiment
export CircularArraySARJTSTraces, CircularArraySARJTSATraces

using ReinforcementLearning
using LinearAlgebra: norm
using ProgressMeter
import CircularArrayBuffers.CircularArrayBuffer
import CircularArrayBuffers

#-----------------------------------------------------------------
# Trajectory Utilities
#-----------------------------------------------------------------

# SARS traces with Jump time
const CircularArraySARJTSTraces = Traces{
    (:state,:next_state,:action,:reward,:jump_time,:terminal),
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:CircularArrayBuffer}},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer}
    }
}

function CircularArraySARJTSTraces(;
    capacity::Int,
    state=Int => (),
    action=Int => (),
    reward=Float64 => (),
    jump_time=Float64 => (),
    terminal=Bool => ())
    
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    jump_eltype, jump_size = jump_time
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(CircularArrayBuffer{state_eltype}(state_size..., capacity+1)) +
    Traces(
        action = CircularArrayBuffer{action_eltype}(action_size..., capacity),
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        jump_time=CircularArrayBuffer{jump_eltype}(jump_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end

CircularArrayBuffers.capacity(t::CircularArraySARJTSTraces) = CircularArrayBuffers.capacity(minimum(map(capacity,t.traces)))

# SARSA traces with Jump time
const CircularArraySARJTSATraces = Traces{
    (:state, :next_state, :action, :next_action, :reward, :jump_time, :terminal),
    <:Tuple{
        <:MultiplexTraces{SS′,<:Trace{<:CircularArrayBuffer}},
        <:MultiplexTraces{AA′,<:Trace{<:CircularArrayBuffer}},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
    }
}

function CircularArraySARJTSATraces(;
    capacity::Int,
    state=Int => (),
    action=Int => (),
    reward=Float64 => (),
    jump_time=Float64 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    jump_eltype, jump_size = jump_time
    terminal_eltype, terminal_size = terminal

    MultiplexTraces{SS′}(CircularArrayBuffer{state_eltype}(state_size..., capacity+2)) +
    MultiplexTraces{AA′}(CircularArrayBuffer{action_eltype}(action_size..., capacity+1)) +
    Traces(
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity+1),
        jump_time=CircularArrayBuffer{jump_eltype}(jump_size..., capacity+1),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity+1),
    )
end

CircularArrayBuffers.capacity(t::CircularArraySARJTSATraces) = CircularArrayBuffers.capacity(minimum(map(capacity,t.traces)))

function Base.push!(eb::EpisodesBuffer{<:Any,<:Any,<:CircularArraySARJTSATraces}, xs::PartialNamedTuple)
    if max_length(eb) == RL.RLTrajectories.capacity(eb.traces)
        popfirst!(eb)
    end
    push!(eb.traces, xs.namedtuple)
    eb.sampleable_inds[end-1] = 1 #completes the episode trajectory.
end

max_length(eb::EpisodesBuffer) = max_length(eb.traces)
max_length(t::Traces) = mapreduce(length, max, t.traces)


#-----------------------------------------------------------------
# SMDP TD learner algorithm 
#-----------------------------------------------------------------
mutable struct SMDPTDLearner{algo} <: AbstractLearner
    approximator::TabularApproximator
    α::Float64    # learning rate
    d::Float64    # continuous time discount rate

    function SMDPTDLearner(; approximator::TabularApproximator, method::Symbol, α = 0.1, d = 0.)
        if method ∉ [:SARS, :SARSA]
            @error "Method $method is not supported"
        else
            new{method}(approximator, α, d)
        end
    end
end

jump_time(::AbstractEnv) = 1.
jump_time(env::StateTransformedEnv) = jump_time(env.env)
jump_time(env::ActionTransformedEnv) = jump_time(env.env)

RLCore.forward(L::SMDPTDLearner, env::AbstractEnv) = RLCore.forward(L.approximator, state(env))
RLCore.forward(L::SMDPTDLearner, s::I) where {I<:Integer} = RLCore.forward(L.approximator, s)
RLCore.forward(L::SMDPTDLearner, s::I, a::I) where {I<:Integer} = RLCore.forward(L.approximator, s, a)

function Base.push!(agent::Agent{<:QBasedPolicy{<:SMDPTDLearner}}, ::PostActStage, env::AbstractEnv, action)
    next_state = state(env)
    push!(agent.trajectory, (state = next_state, action = action, reward = reward(env), jump_time = jump_time(env), terminal = is_terminated(env)))
end

function RLBase.optimise!(learner::SMDPTDLearner, ::PostActStage, trajectory::Trajectory)
    idx = findlast(trajectory.container.sampleable_inds)
    if !isnothing(idx)
        optimise!(learner, trajectory.container[idx])
    end
end

function RLBase.optimise!(
    L::SMDPTDLearner{:SARS},
    t::@NamedTuple{state::I1, next_state::I1, action::I2, reward::F, jump_time::F, terminal::Bool},
) where {I1<:Integer,I2<:Integer,F<:AbstractFloat}
    γ = exp(-L.d * t.jump_time)
    current_value = RLCore.forward(L,t.state,t.action)
    next_value = maximum(RLCore.forward(L,t.next_state))
    Δ = t.reward + γ * next_value - current_value # Discount factor γ is applied here
    L.approximator.model[t.action, t.state] += L.α * Δ
end

function RLBase.optimise!(
    L::SMDPTDLearner{:SARSA},
    t::@NamedTuple{state::I1, next_state::I1, action::I2, next_action::I2, reward::F, jump_time::F, terminal::Bool},
) where {I1<:Integer,I2<:Integer,F<:AbstractFloat}
    γ = exp(-L.d * t.jump_time)
    current_value = RLCore.forward(L,t.state,t.action)
    next_value = RLCore.forward(L,t.next_state,t.next_action)
    Δ = t.reward + γ * next_value - current_value # Discount factor γ is applied here
    L.approximator.model[t.action, t.state] += L.α * Δ
end

#-----------------------------------------------------------------
# Hook struct to count total steps and episodes in an experiment
#-----------------------------------------------------------------
Base.@kwdef mutable struct StepsEpisodesPerExperiment <: AbstractHook
    steps::Int = 0
    episodes::Int = 0
end

Base.getindex(h::StepsEpisodesPerExperiment) = (h.steps, h.episodes)

Base.push!(hook::StepsEpisodesPerExperiment, ::PostActStage, agent::AbstractPolicy, env::AbstractEnv) = hook.steps += 1
Base.push!(hook::StepsEpisodesPerExperiment, ::PostEpisodeStage, agent::AbstractPolicy, env::AbstractEnv) = hook.episodes += 1

#----------------------------------------------------------------------------------------------
# Stop condition to stop when the average value reaches within a threshold of a target value 
# and policy composition reaches within a threshold of the target policy composition
#----------------------------------------------------------------------------------------------
mutable struct StopWhenValueAndPolicyReaches{progress} <: AbstractStopCondition
    targetValue::Vector{Float64}
    targetPolicy::Vector{Int64}
    threshold::Float64
    max_episodes::Int64
    check_interval::Int64
    current_episode::Int64
    eval_points::Int64
    progress::Union{<:ProgressMeter.AbstractProgress, Nothing}
    min_error::Float64
    internal_variables::Dict{Symbol, Array{<:Real}}
    function StopWhenValueAndPolicyReaches(; targetValue, targetPolicy, threshold, max_episodes, check_interval = 1, eval_points = 100, show_progress = true)
        progress = if show_progress
            ProgressThresh(threshold; desc = "Reaching Optimal Value")
        else
            nothing
        end
        new{show_progress}(targetValue, targetPolicy, threshold, max_episodes, check_interval, 0, eval_points, progress, threshold*1000., Dict{Symbol, Array{<:Real}}())
    end
end

function value_policy(app::TabularQApproximator, stop::StopWhenValueAndPolicyReaches)
    @views vals, policy_indices = findmax(app.model[:, 2:end-2]; dims = 1)
    policy = getindex.(policy_indices,1)[:]
    vals = vals[:]
    return vals, policy
end

function vp_error(stop::StopWhenValueAndPolicyReaches, vals, policy)
    max(
        norm(vals .- stop.targetValue, 2)/norm(stop.targetValue, 2),
        norm(policy .!= stop.targetPolicy, 1)/length(policy)
    )
end

function RLCore.check!(stop::StopWhenValueAndPolicyReaches, agent::Agent, env::AbstractEnv)
    if is_terminated(env) 
        stop.current_episode += 1
        if stop.current_episode >= stop.max_episodes
            finish!(stop)
            return true
        elseif stop.current_episode % stop.check_interval == 0
            error = vp_error(stop, value_policy(agent.policy.learner.approximator, stop)...)
            stop.min_error = min(error, stop.min_error)
            update!(stop, error)
            return (error < stop.threshold)
        else
            return false
        end
    else
        return false
    end
end

function finish!(stop::StopWhenValueAndPolicyReaches{true})
    ProgressMeter.finish!(stop.progress)
end

function finish!(::StopWhenValueAndPolicyReaches{false}) end

function update!(stop::StopWhenValueAndPolicyReaches{true}, error)
    ProgressMeter.update!(stop.progress, error)
end

function update!(_::StopWhenValueAndPolicyReaches{false}, _) end

end # module end