## ----------------------------------------------------------------------------
# Set Parameters
# -----------------------------------------------------------------------------
# Ec: animal needs to reach this to get a reward 
# ρ:  proportion of dangerous prey
# m:  rate of metabolism: time*metabolism = energy spent (should be positive) (integer)
# d:  mortality rate per unit time. Mortality is a Poisson process with time to death being exponentially distributed
# Es: safe prey energy 
# ΔEs: std of safe prey energy
# hs: safe prey handling time
# Ed: dangerous prey energy
# ΔEd: std of dangerous prey energy
# hd: dangerous prey handling time
# ϕ:  probability of success for dangerous prey
# Ei: injury energy
# ΔEi: std of injury energy
# hi: injury recovery time
# d: mortality rate (used in the algorithm and not directly in the simulation of the environment)
Ec = 20
ρ = 0.05:0.05:0.95
ϕ = 0.05:0.05:0.95
d = 0.1
m = 1
Δm = 0.01
Es = 2
ΔEs = 1.5
hs = 1.
Ed = 10
ΔEd = 1.5
hd = 1.0
Ei = 4
ΔEi = 1.5
hi = 1.0

num_episodes = 10_000               # number of episodes a single experiment lasts
num_experiments = 10_000            # for each parameter combination, these many experiments are done to obtain average metrics

# File to which the output of the simulation will be stored as a .arrow file
output_file_path = ""
output_file_name = "adult_environment_optimal_performance"

## ----------------------------------------------------------------------------
# Load packages and set up
# -----------------------------------------------------------------------------
using Distributed
using Base.Iterators
using Arrow
using DataFramesMeta
using Chain
using Folds
using SlurmClusterManager

# Automatically add as many processes as available.
# For a personal computer this is the number of CPU threads.
# For a batch job on a SLURM cluster, this requires the option "ntasks" to be set.
if nprocs() == 1
    if haskey(ENV, "SLURM_JOB_ID") || haskey(ENV, "SLURM_JOBID")
        addprocs(SlurmManager())
    else
        addprocs(Sys.CPU_THREADS)
    end
end
@everywhere using ReinforcementLearning
@everywhere if !@isdefined RLUtilities
    include("RLUtilities.jl")
    using .RLUtilities
end
@everywhere using Random
@everywhere using Distributions
@everywhere using Memoize
@everywhere using Measurements
@everywhere using LinearAlgebra: dot
@everywhere using ProgressMeter
@everywhere using DataFrames
@everywhere import Dates

# functions for saving and retreiving of Measurement type in Arrow tables
const NAME = Symbol("JuliaLang.Measurement")
ArrowTypes.ArrowKind(::Type{Measurement{T}}) where {T} = ArrowTypes.ListKind
ArrowTypes.ArrowType(::Type{Measurement{T}}) where {T} = Tuple{T, T}
ArrowTypes.toarrow(m::Measurement{T}) where {T} = (m.val, m.err)
ArrowTypes.arrowname(::Type{Measurement{T}}) where {T} = NAME
ArrowTypes.JuliaType(::Val{NAME}, ::Type{Tuple{T, T}}) where {T} = Measurement{T}
ArrowTypes.fromarrow(::Type{Measurement{T}}, m::Tuple{T, T}) where {T} = measurement(m...)

@everywhere begin

## ------------------------------------------------------------------------------
# Discrete normal distribution for energy changes
# It is a normal distribution that is discretised at integer values 
# and then truncated at 0 and Ec - E
# --------------------------------------------------------------------------------
struct DiscreteNormal
    dist
    function DiscreteNormal(μ, σ)
        d = truncated(Normal(μ, σ); lower = 0.)
        new(d)
    end
end

Distributions.rand(rng::AbstractRNG, d::DiscreteNormal) = floor(Int, rand(rng, d.dist))
Distributions.rand!(rng::AbstractRNG, d::DiscreteNormal, A::AbstractArray{<:Integer}) = A .= floor.(Int, rand!(rng, d.dist, similar(A, Float64)))
@memoize Distributions.pdf(dist::DiscreteNormal, x::Int)::Float64 = cdf(dist.dist, x+1) - cdf(dist.dist, x) 
@memoize Distributions.pdf(dist::DiscreteNormal, x::AbstractArray{Int}) = cdf(dist.dist, collect(x) .+ 1) - cdf(dist.dist, collect(x))
@memoize Distributions.ccdf(dist::DiscreteNormal, x::Any) = ccdf(dist.dist, x)
@memoize Distributions.cdf(dist::DiscreteNormal, x::Int) = cdf(dist.dist, x+1)
@memoize Distributions.cdf(dist::DiscreteNormal, x::AbstractArray{Int}) = cdf(dist.dist, collect(x) .+ 1)
@memoize Distributions.ccdf(dist::Dirac, x::Real) = x <= dist.value ? 1.0 : 0.0

## ------------------------------------------------------------------------------
# Deterministic dynamic programming functions
# ------------------------------------------------------------------------------
function surv_prob(rate, time) 
    exp(-rate*time)
end

function single_outcome_return(V, Ec, dist_pdf, dist_ccdf, energy, ::Val{:positive})
    Δes = 1:Ec-energy
    indices = energy+1:Ec
    @views G = dot(dist_pdf[Δes], V[indices]) + dist_ccdf[Ec-energy+1]*V[Ec+1]
    return G
end

function single_outcome_return(V, Ec, dist_pdf, dist_ccdf, energy, ::Val{:negative})
    Δes = 1:energy
    indices = energy+1:-1:2
    @views G = dot(dist_pdf[Δes], V[indices])
    return G
end

# Calculate the expected return of taking action [action] when in energy [energy],
# as a sum of expected returns of all the possible outcomes from that action
function exp_return(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s, action, energy)
    G = 0.
    @inbounds for outcome in 1:3
        g = single_outcome_return(V, Ec, dist_pdfs[action, outcome], dist_ccdfs[action, outcome], energy, e_signs[action, outcome])
        G += p[action, outcome]*s[action, outcome]*g
    end
    return G
end

# Perform the value iteration algorithm to calculate optimal value for given the transition matrices
# For each energy level, calculate the estimate of expected value given the current estimate
# Repeat this loop until it converges and the changes are below tolerance [tol]
# Output: value function - vector of length Ec+1, storing the value of energies 0, ..., Ec.
function value_iteration!(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = 1e-6, maxiter = 10_000)
    Δ = 0.
    actions = [2;3;1]
    @inbounds for _ in 1:maxiter
        Δ = 0.
        @inbounds for energy in 1:Ec-1
            v = V[energy+1]
            V[energy+1] = maximum(action->exp_return(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s, action, energy), actions)
            Δ = max(Δ, abs(v-V[energy+1])/(v==0 ? 1 : v))
        end
        
        @inbounds V[Ec+1] = 1.

        if Δ < tol 
            break 
        end
    end
    return V
end

# Given the optimal value, calculate the optimal policy
# For each energy level, check the expected returns of each action,
# and pick the one with the highest expected return 
# Output: optimal policy - vector of length Ec+1, storing the optimal action for energy level 0, ..., Ec.
# The actions are represented by integers- 1: indiscriminate attack, 2: risk-prone attack, 3: risk-averse attack  
function policy_from_optimal_value(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
    policy = zeros(Int, Ec-1)
    actions = [2;3;1]
    @inbounds for energy in 1:Ec-1
        policy[energy] = argmax(action->exp_return(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s, action, energy), actions)
    end
    return policy
end

# For a given policy and value function, calculate the state action value function for that
# For each energy level, and action, calculate the expected value and store it.
# Output: matrix of size (3, Ec+1). The first index represents the 
# action taken (indiscriminate attack, risk-prone attack, risk-averse attack),
# and the second index represents energy level 0, ..., Ec.
function state_action_value_from_policy(V, policy, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
    Q = zeros(3, Ec + 1)
    Q[:, end] .= 1.
    for energy in 1:Ec-1
        Q[policy[energy], energy+1] = V[energy+1]
    end

    for energy in 1:Ec-1
        for action in [1;2;3]
            if action == policy[energy]
                continue
            end
            Q[action, energy+1] = exp_return(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s, action, energy)
        end
    end
    return Q
end

# Calculates the compressed probability matrix for the action and outcome, given parameters. 
# The rows represent the actions taken and the columns, the outcomes.
# Row1: indiscriminate attack, Row2: risk-prone attack, Row3: risk-averse attack
# The outcomes are represented as follows:
#[ Obtain safe prey, obtain dangerous prey, get injured ;
#  Metabolic loss,   obtain dangerous prey, get injured ;
#  Obtain safe prey, metabolic loss,           ---       ]
function prob(ρ, ϕ)
    [ 1-ρ  ρ*ϕ  ρ*(1-ϕ) ;    
      1-ρ  ρ*ϕ  ρ*(1-ϕ) ;    
      1-ρ  ρ    0        ]
end

# Sets the constant compressed matrix for the signs of energy changes 
# for action and outcome, given parameters. 
# The format is the same as the probability matrix above.
const e_signs = Val.([ :positive   :positive   :negative ;
                       :negative   :positive   :negative ;
                       :positive   :negative   :positive  ])

# Sets the compressed transition time matrix 
# for action and outcome, given parameters. 
# The format is the same as the probability matrix above.
function jump_times(hs, hd, hi)
    [ hs + 1   hd + 1   hi + 1 ;
      1        hd + 1   hi + 1 ;
      hs + 1   1        1       ]
end

# Calculates the compressed survival probability matrix 
# for action and outcome, given parameters. 
# The format is the same as the probability matrix above.
function surv(d, hs, hd, hi) 
    surv_prob.(d,jump_times(hs, hd, hi))
end

# Sets the compressed matrix of energy distributions for the different 
# actions and outcomes. The distributions show what are the possible 
# energy changes for each given outcome, with a given mean and variance.
# The distributions are discretised normal distributions that are 
# truncated at 0.
# The format is the same as the probability matrix above.
function distributions(m, Δm, Es, ΔEs, Ed, ΔEd, Ei, ΔEi)
    ds = (ΔEs*Es > 0) ? DiscreteNormal(Es, ΔEs*Es) : Dirac(Es)
    dd = (ΔEd*Ed > 0) ? DiscreteNormal(Ed, ΔEd*Ed) : Dirac(Ed)
    di = (ΔEi*Ei > 0) ? DiscreteNormal(Ei, ΔEi*Ei) : Dirac(Ei)
    dm = (Δm*m > 0) ? DiscreteNormal(m, Δm*m) : Dirac(m)

    [ ds   dd   di       ;
      dm   dd   di       ;
      ds   dm   Dirac(0)  ]
end

# Calculates the optimal value, policy and state-action value functions given the parameter values of the environment.
# Calculates the transition probability matrices and passes this to the main value iteration function.
function calc_state_action_value(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi; tol = 1e-6, maxiter = 1_000)
    # transition probabilities
    p = prob(ρ, ϕ)
    dists = distributions(m, Δm, Es, ΔEs, Ed, ΔEd, Ei, ΔEi)
    dist_pdfs = pdf.(dists, (0:Ec,))
    dist_ccdfs = ccdf.(dists, (0:Ec,))
    s = surv(d, hs, hd, hi)

    V = zeros(Float64, Ec + 1)
    V[end] = 1.

    V = value_iteration!(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = tol, maxiter = maxiter)
    policy = policy_from_optimal_value(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
    Q = state_action_value_from_policy(V, policy, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
    return Q
end

## ------------------------------------------------------------------------------
# Define the environment as required by the ReinforcementLearning.jl package
# -------------------------------------------------------------------------------
mutable struct TwoPreyEnv <: AbstractEnv
    Ec::Int64
    energy::Int64
    Δt::Float64
    reward::Float64

    e_signs::Matrix{Int}
    c::Matrix{Float64}
    t::Matrix{Float64}
    s::Matrix{Float64}
    dists::Matrix

    function TwoPreyEnv(; Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)
        # signs of energy changes
        e_signs = [ +1  +1  -1  ;
                    -1  +1  -1  ;
                    +1  -1  +1   ]
        # transition probability function (action , outcome)
        p = [ 1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
            1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
            1-ρ  ρ    0        ]

        c = zeros(Float64, 3, 3) # cumulative transition probabilities
        c[:,1] .= p[:,1]
        c[:,2] .= c[:,1] .+ p[:,2]
        c[:,3] .= c[:,2] .+ p[:,3]

        # jump times for different (action, outcome) pairs
        t = [ hs + 1   hd + 1   hi + 1  ;
              1        hd + 1   hi + 1  ;
              hs + 1   1        0        ]

        s = surv_prob.(d, t)     # survival probabilties

        ds = (ΔEs*Es > 0) ? DiscreteNormal(Es, ΔEs*Es) : Dirac(Es)
        dd = (ΔEd*Ed > 0) ? DiscreteNormal(Ed, ΔEd*Ed) : Dirac(Ed)
        di = (ΔEi*Ei > 0) ? DiscreteNormal(Ei, ΔEi*Ei) : Dirac(Ei)
        dm = (Δm*m > 0) ? DiscreteNormal(m, Δm*m) : Dirac(m)

        # distributions of energy benefits (and costs)
        dists = [ ds   dd   di       ;
                  dm   dd   di       ;
                  ds   dm   Dirac(0)  ]

        new(Ec, rand(1:Ec-1), 0., 0., e_signs, c, t, s, dists)
    end
end

# Possible actions taken by the agent. 1: indiscriminate attack, 2: risk-prone attack, 3: risk-averse attack
RLBase.action_space(::TwoPreyEnv) = [1; 2; 3]   

# Possible state (energy) values of the environment. E = 0, ..., Ec. 
# Ec + 1 is the terminal goal state. From Ec the state moves to Ec + 1 automatically. 
RLBase.state_space(env::TwoPreyEnv) = 0:env.Ec+1  

# Reward function. Reward is given once the agent reaches the terminal state, which is Ec + 1.
RLBase.reward(env::TwoPreyEnv) = env.energy == env.Ec+1

# A single environment is terminated, when the energy reaches zero or the terminal goal state.
RLBase.is_terminated(env::TwoPreyEnv) = (env.energy == 0) || (env.energy == env.Ec + 1)

RLBase.state(env::TwoPreyEnv, ::Observation{Any}, ::DefaultPlayer) = env.energy
RLUtilities.jump_time(env::TwoPreyEnv) = env.Δt

# Resetting the environment after an episode is completed means clearing the jump time, reward 
# and randomly initialising the energy between 1 and Ec-1.
function RLBase.reset!(env::TwoPreyEnv) 
    env.energy = rand(1:env.Ec-1)
    env.Δt = 0.
    env.reward = 0.
end

# This function enacts the main dynamics of the Two-prey environment.
function RLBase.act!(x::TwoPreyEnv, action)
    if x.energy == x.Ec # if energy is at the threshold, move to terminal state
        x.energy += 1
    else                # otherwise, continue with the process
        draw = rand()
        for outcome in 1:3
            if draw < x.c[action, outcome]
                # here the individual doesn't die but is always given some energy if d is set to zero
                # if that is the case, the probability of death is assumed to be known by the individual (perhaps evolutionarily)
                x.energy += if rand() <= x.s[action, outcome]     #if it survives
                                x.e_signs[action,outcome] * rand(x.dists[action, outcome])   #then add energy
                            else
                                -x.Ec                          #otherwise it dies
                            end
                x.Δt = x.t[action, outcome]
                break
            end
        end
        if x.energy < 0
            x.energy = 0
        elseif x.energy > x.Ec
            x.energy = x.Ec
        end
    end 
end

## ------------------------------------------------------------------------------
# Reinforcement learning simulation functions
# ------------------------------------------------------------------------------

# Performs a single learning run to calculate the optimal policy and value and the 
# learning time to reach that optimal policy and value
function single_learning_run(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi, num_episodes)

    NS = Ec + 2
    NA = 3

    # the optimal state-action value is calculated in order to measure the performance of the optimal policy
    Q = calc_state_action_value(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)
    approximator = TabularQApproximator(; n_state = NS, n_action = NA, init = 0.)
    approximator.model[:, 1:Ec+1] .= Q

    # for testing the post development performance 
    policy = QBasedPolicy(
        learner = approximator,
        explorer = GreedyExplorer()
    )

    env = TwoPreyEnv(; Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)
    
    state_mapping = s -> s + 1
    wrapped_env = StateTransformedEnv(
            env;
            state_mapping = state_mapping,   
            state_space_mapping = _ -> Base.OneTo(NS)
    );

    stop_condition = StopAfterNEpisodes(num_episodes, is_show_progress = false)
    performance_hook = TotalRewardPerEpisode()

    # running the learning processes
    run(
        policy,
        wrapped_env,
        stop_condition,
        performance_hook
    ) 

    rewards = performance_hook.rewards
    rewards_per_episode = mean(rewards)

    return  rewards_per_episode
end

end # distributed functions

# Runs the learning experiment for multiple parameter combinations in a parallelized manner
# and outputs the progress to the standard error stream.
function multiple_parameter_learning_runs(pars_iterable)
    num_pars = length(pars_iterable)
    rewards_per_episode = zeros(Float64, num_pars)

    results = @showprogress pmap(pars -> single_learning_run(pars...), pars_iterable; batch_size = 100)

    rewards_per_episode .= results

    return  rewards_per_episode
end

## ------------------------------------------------------------------------------
# creating the iterable of parameters
# -------------------------------------------------------------------------------

all_pars_iterable = product(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi, num_episodes)
all_pars_iterable = collect(all_pars_iterable)[:]
all_pars_iterable = repeat(all_pars_iterable, num_experiments)
## ------------------------------------------------------------------------------
# running the experiments for all the specified parameters
# -------------------------------------------------------------------------------

start = time()

rewards_per_episode = multiple_parameter_learning_runs(all_pars_iterable)

fin = time()
println("Time elapsed: $(round(fin - start; digits = 3)) s")

## -------------------------------------------------------------------------------
# adding the parameters to the dataframe and saving iterable
# --------------------------------------------------------------------------------
columns = (;  
    Ec = Int64[], 
    ρ = Float64[],
    ϕ = Float64[],
    d = Float64[],
    m = Float64[],
    Δm = Float64[],
    Es = Float64[], 
    ΔEs = Float64[], 
    hs = Float64[], 
    Ed = Float64[], 
    ΔEd = Float64[], 
    hd = Float64[], 
    Ei = Float64[], 
    ΔEi = Float64[], 
    hi = Float64[],

    num_episodes = Int64[],

    rewards_per_episode_optimum = Float64[],
)

df = DataFrame(columns)

for (pars, reward) in zip(all_pars_iterable, rewards_per_episode)
    push!(df, (pars..., reward))
end

df = @chain df begin
    @groupby(Not([:rewards_per_episode]))
    @combine(:rewards_per_episode = mean(:rewards_per_episode) ± std(:rewards_per_episode))
end

Arrow.write(output_file_path*output_file_name*".arrow", df; compress = :lz4)