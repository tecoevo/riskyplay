## -------------------------------------------------------
# Set parameters
# -------------------------------------------------------
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
ρ = 0.01:0.01:0.99           
m = 1             
d = 0.1          
Δm = 0.1         
Es = 2           
ΔEs = 1.5         
hs = 1.           
Ed = 10           
ΔEd = 1.5        
hd = 1.0          
ϕ = 0.01:0.01:0.99           
Ei = 4.           
ΔEi = 1.5         
hi = 1.0          

number_experiments = 10_000

#learning algorithm
α = 0.01 # learning rate of learning algorithm
maximum_episodes = 200_000 # number of episodes after changing the environmental parameters
learning_convergence_threshold = 0.25        # stops learning when reaching within this tolerance of the optimal value and policy. Set to 0 for fixed number of episodes

# explorer parameters
ϵ = 0.2 # exploration rate 

# File to which the output of the simulation will be stored as a .arrow file
output_file_path = ""
output_file_name = "optimality_of_RL"

## -------------------------------------------------------
# load packages and set up
# -------------------------------------------------------
using Distributed
using CairoMakie
using Base.Iterators
using DataFrames
using Arrow
using Folds

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
    include("../RLUtilities.jl")
    using .RLUtilities
end
@everywhere using Random
@everywhere using Distributions
@everywhere using Memoize
@everywhere using Measurements
@everywhere using LinearAlgebra: dot
@everywhere using ProgressMeter
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
Distributions.pdf(dist::DiscreteNormal, x::Int)::Float64 = cdf(dist.dist, x+1) - cdf(dist.dist, x) 
Distributions.pdf(dist::DiscreteNormal, x::AbstractArray{Int}) = cdf(dist.dist, collect(x) .+ 1) - cdf(dist.dist, collect(x))
Distributions.ccdf(dist::DiscreteNormal, x::Real) = ccdf(dist.dist, x)
Distributions.ccdf(dist::DiscreteNormal, x::AbstractArray{<:Real}) = ccdf(dist.dist, x)
surv_prob(d, time) = exp(-d*time)

end # everywhere block

## ------------------------------------------------------------------------------
# Deterministic dynamic programming functions
# ------------------------------------------------------------------------------
@everywhere  begin
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
function exp_return(V,Ec, dist_pdfs, dist_ccdfs,e_signs,p,s,action,energy)
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
function value_iteration!(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = 1e-6, maxiter = 1_000)
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
# The format is the same as the probability matrix above
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

# Calculates the probability distribution functions for all 
# required energy values from the distribution matrix calculated above
function distribution_pdfs(dists, Ec)
    pdf.(dists, (0:Ec,))
end

# Calculates the complementary cumulative distribution functions for all 
# required energy values from the distribution matrix calculated above
# This is required in order to calculated expected values for all cases when 
# energy obtained would increase the energy to Ec or decrease it to 0
function distribution_ccdfs(dists, Ec)
    ccdf.(dists, (0:Ec,))
end

# Calculates the optimal value and policy functions given the parameter values of the environment
# Calculates the transition probability matrices and passes this to the main value iteration function.
function calc_value_policy(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi; tol = 1e-6, maxiter = 10_000)
    # transition probabilities
    p = prob(ρ, ϕ)
    dists = distributions(m, Δm, Es, ΔEs, Ed, ΔEd, Ei, ΔEi)
    dist_pdfs = distribution_pdfs(dists, Ec)
    dist_ccdfs = distribution_ccdfs(dists, Ec)
    s = surv(d, hs, hd, hi)

    V = zeros(Float64, Ec + 1)
    V[end] = 1.

    V = value_iteration!(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = tol, maxiter = maxiter)
    policy = policy_from_optimal_value(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
    return V[2:end-1], policy
end

end # everywhere block

## ------------------------------------------------------------------------------
# Define the environment as required by the ReinforcementLearning.jl package
# -------------------------------------------------------------------------------
@everywhere begin
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

    function TwoPreyEnv(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)
        # signs of energy changes
        e_signs = [ +1  +1  -1  ;
                    -1  +1  -1  ;
                    +1  -1  +1   ]
        # transition probability function (option , outcome)
        p = [ 1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
            1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
            1-ρ  ρ    0        ]

        c = zeros(Float64, 3, 3) # cumulative transition probabilities
        c[:,1] .= p[:,1]
        c[:,2] .= c[:,1] .+ p[:,2]
        c[:,3] .= c[:,2] .+ p[:,3]

        # jump times for different (option, outcome) pairs
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
RLBase.state_space(env::TwoPreyEnv) = 0:env.Ec+1  # Ec is the terminal state.

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
function RLBase.act!(x::TwoPreyEnv,action)
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

# Policy composition is the fraction of energy levels at which the 
# action is indiscriminate attack, risk-prone attack and risk-averse attack
function calc_policy_composition(Q_table)
    policy_idcs = argmax(Q_table, dims = 1)
    policy = getindex.(policy_idcs, 1)[:]
    policy_length = length(policy)
    both = sum(policy .== 1)/policy_length
    dangerous = sum(policy .== 2)/policy_length
    safe = 1 - both - dangerous
    return [both; dangerous; safe]
end

calc_average_value(Q_table) = mean(maximum(Q_table, dims = 1))

## ------------------------------------------------------------------------------
# Reinforcement learning simulation functions
# ------------------------------------------------------------------------------

# perform a single experiment of the learning process for a given set of parameters
function single_learning_run(parameters, maximum_episodes, learning_convergence_threshold, opt_value, opt_policy, α, ϵ)
    (Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi) = parameters

    # creating the agents constructs
    NS = Ec + 2
    NA = 3
    learner = SMDPTDLearner(
        approximator = TabularQApproximator(; n_state = NS, n_action = NA, init = 0.),
        method = :SARSA,
        α = α,
        d = d
    )

    trajectory = Trajectory(
        CircularArraySARJTSATraces(capacity = 100,
            state = Int64 => (),
            reward = Float64 => (),
            jump_time = Float64 => (),
            action = Int64 => ()
        ),
    DummySampler());

    explorer = EpsilonGreedyExplorer(
        kind = :exp,
        ϵ_init = 0.1,
        warmup_steps = 0,
        ϵ_stable = ϵ,
        decay_steps = 0,
        is_break_tie = true
    )

    rand!(@view learner.approximator.model[:, 2:end-1])

    agent = Agent(
        policy = QBasedPolicy(learner = learner, explorer = explorer),
        trajectory = trajectory,
    )

    # creating the environments and wrapped environments

    env = TwoPreyEnv(Ec, ρ, ϕ, 0., m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)
    state_mapping = s -> s + 1
    
    wrapped_env = StateTransformedEnv(
            env;
            state_mapping = state_mapping,   
            state_space_mapping = _ -> Base.OneTo(NS)
    );
    
    # test environment
    test_env = TwoPreyEnv(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)

    wrapped_test_env = StateTransformedEnv(
            test_env;
            state_mapping = state_mapping,   
            state_space_mapping = _ -> Base.OneTo(NS)
    );

    if learning_convergence_threshold == 0
        stop_condition = StopAfterNEpisodes(maximum_episodes; is_show_progress = false)
    else
        stop_condition = StopWhenValueAndPolicyReaches(;
            targetValue = opt_value,
            targetPolicy = opt_policy,
            threshold = learning_convergence_threshold,
            max_episodes = maximum_episodes,
            show_progress = false
        )
    end

    measure_time = TimeEpisodesPerExperiment()
    # running the learning processes
    run(
        agent,
        wrapped_env,
        stop_condition,
        measure_time
    ) 

    # testing the post development performance 
    performance_policy = QBasedPolicy(
        learner = learner.approximator,
        explorer = GreedyExplorer()
    );

    performance_hook = TotalRewardPerEpisode()
    run(
        performance_policy,
        wrapped_test_env,
        StopAfterNEpisodes(1_000; is_show_progress = false),
        performance_hook
    )
    reward_per_episode = mean(performance_hook.rewards) 

    policy_composition = calc_policy_composition(learner.approximator.model[:, 2:end-1])

    learning_time = measure_time.time
    learning_time_episodes = measure_time.episodes

    return reward_per_episode, learning_time, learning_time_episodes, policy_composition
end

end # everywhere block

# Runs learning experiments multiple times for a single set of parameters 
# to obtain averaged metrics
function ensemble_learning_runs(parameters, maximum_episodes, learning_convergence_threshold, α, ϵ; number_experiments = 1, channel)

    opt_value, opt_policy = calc_value_policy(parameters...; tol = 1e-6, maxiter = 1_000)

    results = pmap(1:number_experiments) do _
        res = single_learning_run(parameters, maximum_episodes, learning_convergence_threshold, opt_value, opt_policy, α, ϵ)
        put!(channel, true)
        res
    end

    reward_per_episode = zeros(number_experiments)
    policy_composition = zeros(3, number_experiments)
    learning_time = zeros(number_experiments)
    learning_time_episodes = zeros(number_experiments)

    for i in 1:number_experiments
        reward_per_episode[i], learning_time[i], learning_time_episodes[i], policy_composition[:,i] = results[i]
    end

    filtered_indices = learning_time_episodes .< maximum_episodes
    reward_per_episode = reward_per_episode[filtered_indices]
    learning_time = learning_time[filtered_indices]
    policy_composition = policy_composition[:, filtered_indices]
    number_experiments_actual = sum(filtered_indices)

    return  mean(reward_per_episode) ± std(reward_per_episode), 
            mean(learning_time) ± std(learning_time),
            mean(policy_composition; dims = 2)[:],
            std(policy_composition; dims = 2)[:], 
            number_experiments_actual
end

# Runs the ensemble learning experiments for multiple parameter combinations in a parallelized manner.
function multiple_parameter_learning_runs(pars_iterable, number_experiments)
    num_pars = length(pars_iterable)
    pbar = Progress(num_pars * number_experiments; dt = 10)
    reward_per_episode = zeros(Measurement, num_pars)
    learning_time = zeros(Measurement, num_pars)
    policy_composition = Vector{Vector{Float64}}(undef, num_pars)
    policy_composition_std = Vector{Vector{Float64}}(undef, num_pars)
    number_experiments_actual = zeros(Int64, num_pars)

    channel = RemoteChannel(() -> Channel{Bool}(), 1)

    results = @sync begin
        @async while take!(channel)
            next!(pbar; showvalues = [(:Time, Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))])
        end

        results = @async begin
            res = map(pars -> ensemble_learning_runs(pars...; channel = channel, number_experiments = number_experiments), pars_iterable)
            put!(channel, false)
            res
        end
    end
    results = fetch(results)

    for i in 1:num_pars
        reward_per_episode[i], 
        learning_time[i], 
        policy_composition[i], 
        policy_composition_std[i], 
        number_experiments_actual[i] = results[i]
    end

    return  reward_per_episode, 
            learning_time,
            policy_composition, 
            policy_composition_std,
            number_experiments_actual
end

# Write to file which parameters will be used in the simulation
parameters = ["maximum_episodes", "number_experiments", "learning_convergence_threshold", "α", "ϵ", "output_file_path", "output_file_name"]

open(output_file_path*output_file_name*"_pars.txt", "w") do file
    write(file, string(Dates.now()), "\n")
    for key in parameters
        value = string(eval(Symbol(key)))
        write(file,key," => ",value,"\n")
    end
end

## ------------------------------------------------------------------------------
# creating the iterable of parameters
# -------------------------------------------------------------------------------
pars_iterable = product(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi, maximum_episodes, learning_convergence_threshold, α, ϵ)

pars_iterable = collect(pars_iterable)[:]
pars_iterable = map(pars_iterable) do (Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi, rest...)
    ((Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi), rest...)
end

## ------------------------------------------------------------------------------
# running the experiments for all the specified parameters
# -------------------------------------------------------------------------------

start = time()

(reward_per_episode,
learning_time, 
policy_composition, 
policy_composition_std,
number_experiments_actual) = multiple_parameter_learning_runs(pars_iterable, number_experiments)

fin = time()
println("Time elapsed: $(round(fin - start; digits = 3)) s")

## -------------------------------------------------------------------------------
# adding the parameters to the dataframe and saving iterable
# --------------------------------------------------------------------------------
Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi
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

    maximum_episodes = Int64[],
    learning_convergence_threshold = Float64[],
    learning_rate = Float64[],
    exploration_rate = Float64[],
    
    number_experiments = Int64[],
    number_experiments_actual = Int64[],
    
    reward_per_episode = Measurement{Float64}[],
    learning_time = Measurement{Float64}[],
    policy_composition = Vector{Float64}[],
    policy_composition_std = Vector{Float64}[],
)

function unpack_pars(pars) 
    env_pars, rest... = pars
    return (env_pars..., rest...)
end

number_experiments_repeated = fill(number_experiments, size(number_experiments_actual))

df = DataFrame(columns)

for (env_pars, rest...) in zip(pars_iterable, number_experiments_repeated, number_experiments_actual, reward_per_episode, learning_time, policy_composition, policy_composition_std)
    push!(df, (unpack_pars(env_pars)..., rest...))
end

Arrow.write(output_file_path*output_file_name*".arrow", df; compress = :lz4)