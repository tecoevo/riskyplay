## -------------------------------------------------------
# parameters
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
juvenile_parameters = (
    Ec = 20,
    ρ = 0.05:0.05:0.95, 
    ϕ = 0.05:0.05:0.95, 
    d = 0.1,
    m = 1, 
    Δm = 0.01, 
    Es = 2, 
    ΔEs = 1.5, 
    hs = 1., 
    Ed = 10, 
    ΔEd = 1.5, 
    hd = 1.0,
    Ei = 0:4, 
    ΔEi = 1.5, 
    hi = 1.0  
)

adulthood_parameters = (; juvenile_parameters..., Ei = 4)

developmental_time = vcat(0:1_000:50_000, 55_000:5_000:150_000, 160_000:10_000:230_000)            # number of steps to run the learning_process

learning_convergence_threshold = 0.25 # how close the average value and policy needs to come to the average optimal value to consider learning complete

number_experiments = 10_000

#learning algorithm
α = 0.01 # learning rate of learning algorithm
ϵ = 0.2 # exploration rate 
maximum_episodes_adulthood = 200_000 # number of episodes after changing the environmental parameters

output_file_path = ""
output_file_name = "protected_environment_learning"

## -------------------------------------------------------
# load packages and set up
# -------------------------------------------------------
using Distributed
using CairoMakie
using Base.Iterators
using Arrow
using Folds
using DataFramesMeta
using Chain
using SlurmClusterManager

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

using Measurements: value, uncertainty
load_data(name::String) = open(name, "r") do file 
    copy(DataFrame(Arrow.Table(file))) 
end

# functions for saving and retreiving of Measurement type in Arrow tables
const NAME = Symbol("JuliaLang.Measurement")
ArrowTypes.ArrowKind(::Type{Measurement{T}}) where {T} = ArrowTypes.ListKind
ArrowTypes.ArrowType(::Type{Measurement{T}}) where {T} = Tuple{T, T}
ArrowTypes.toarrow(m::Measurement{T}) where {T} = (m.val, m.err)
ArrowTypes.arrowname(::Type{Measurement{T}}) where {T} = NAME
ArrowTypes.JuliaType(::Val{NAME}, ::Type{Tuple{T, T}}) where {T} = Measurement{T}
ArrowTypes.fromarrow(::Type{Measurement{T}}, m::Tuple{T, T}) where {T} = measurement(m...)

@everywhere begin
# discrete normal disctribution for energy
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

# Deterministic dynamic programming functions
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


function exp_return(V,Ec, dist_pdfs, dist_ccdfs,e_signs,p,s,action,energy)
    G = 0.
    @inbounds for outcome in 1:3
        g = single_outcome_return(V, Ec, dist_pdfs[action, outcome], dist_ccdfs[action, outcome], energy, e_signs[action, outcome])
        G += p[action, outcome]*s[action, outcome]*g
    end
    return G
end

function value_iteration!(V, Ec, er, sr, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = 1e-6, maxiter = 10_000)
    Δ = 0.
    actions = [2;3;1]
    @inbounds for _ in 1:maxiter
        Δ = 0.
        @inbounds for energy in 1:Ec-1
            v = V[energy+1]
            V[energy+1] = maximum(action->exp_return(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s, action, energy), actions)
            Δ = max(Δ, abs(v-V[energy+1])/(v==0 ? 1 : v))
        end
        
        @inbounds v = V[Ec+1]
        @inbounds V[Ec+1] = 1. + sr*V[Ec+1 - er]
        Δ = max(Δ, abs(v-V[Ec+1])/(v==0 ? 1 : v))

        if Δ < tol 
            break 
        end
    end
    return V, Δ
end

function value_from_policy(policy, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = 1e-6, maxiter = 10_000)
    V = zeros(Float64, Ec+1)
    V[end] = 1.
    Δ = 0.
    @inbounds for _ in 1:maxiter
        Δ = 0.
        @inbounds for energy in 1:Ec-1
            v = V[energy+1]
            V[energy+1] = exp_return(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s, policy[energy], energy)
            Δ = max(Δ, abs(v-V[energy+1])/(v==0 ? 1 : v))
        end
        if Δ < tol
            break
        end
    end
    return V, Δ
end

function policy_from_optimal_value(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
    policy = zeros(Int, Ec-1)
    actions = [2;3;1]
    @inbounds for energy in 1:Ec-1
        policy[energy] = argmax(action->exp_return(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s, action, energy), actions)
    end
    return policy
end

function prob(ρ, ϕ)
    [ 1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
      1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
      1-ρ  ρ    0        ]
end

function energy(m, Es, Ed, Ei)
    [ Es   Ed   -Ei ;
      -m   Ed   -Ei ;
      Es   -m   0    ]
end

const e_signs = Val.([ :positive   :positive   :negative ;
                       :negative   :positive   :negative ;
                       :positive   :negative   :positive  ])

function jump_times(hs, hd, hi)
    [ hs + 1   hd + 1   hi + 1 ;
      1        hd + 1   hi + 1 ;
      hs + 1   1        1       ]
end

function surv(d, hs, hd, hi) 
    surv_prob.(d,jump_times(hs, hd, hi))
end

function distributions(m, Δm, Es, ΔEs, Ed, ΔEd, Ei, ΔEi)
    ds = (ΔEs*Es > 0) ? DiscreteNormal(Es, ΔEs*Es) : Dirac(Es)
    dd = (ΔEd*Ed > 0) ? DiscreteNormal(Ed, ΔEd*Ed) : Dirac(Ed)
    di = (ΔEi*Ei > 0) ? DiscreteNormal(Ei, ΔEi*Ei) : Dirac(Ei)
    dm = (Δm*m > 0) ? DiscreteNormal(m, Δm*m) : Dirac(m)

    [ ds   dd   di       ;
      dm   dd   di       ;
      ds   dm   Dirac(0)  ]
end

function calc_value_policy(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi, er, sr; tol = 1e-6, maxiter = 10_000)
    # transition probabilities
    p = prob(ρ, ϕ)
    dists = distributions(m, Δm, Es, ΔEs, Ed, ΔEd, Ei, ΔEi)
    dist_pdfs = pdf.(dists, (0:Ec,))
    dist_ccdfs = ccdf.(dists, (0:Ec,))
    s = surv(d, hs, hd, hi)

    V = zeros(Float64, Ec + 1)
    V[end] = 1.

    V, err = value_iteration!(V, Ec, er, sr, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = tol, maxiter = maxiter)
    policy = policy_from_optimal_value(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
    return V[2:end-1], policy
end

function calc_value_policy(; Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)
    calc_value_policy(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi)
end

function calc_value_policy(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi) 
    calc_value_policy(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi, 1, 0.; tol = 1e-9)
end

## -------------------------------------------------------
# define the environment
# -------------------------------------------------------
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
TwoPreyEnv(Ec, ρ, ϕ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, Ei, ΔEi, hi) = TwoPreyEnv(; Ec = Ec, ρ = ρ, ϕ = ϕ, d = d, m = m, Δm = Δm, Es = Es, ΔEs = ΔEs, hs = hs, Ed = Ed, ΔEd = ΔEd, hd = hd, Ei = Ei, ΔEi = ΔEi, hi = hi)
TwoPreyEnv() = TwoPreyEnv(100, 0.5, 0.9, 0.1, 1, 0.01, 2, 1.5, 1, 10, 1.5, 1, 4, 1.5, 1)

RLBase.action_space(::TwoPreyEnv) = [1; 2; 3]
RLBase.state_space(env::TwoPreyEnv) = 0:env.Ec+1  # Ec is the terminal state.
RLBase.reward(env::TwoPreyEnv) = env.energy == env.Ec+1
RLBase.is_terminated(env::TwoPreyEnv) = (env.energy == 0) || (env.energy == env.Ec + 1)
RLBase.state(env::TwoPreyEnv, ::Observation{Any}, ::DefaultPlayer) = env.energy
RLUtilities.jump_time(env::TwoPreyEnv) = env.Δt

function RLBase.reset!(env::TwoPreyEnv) 
    env.energy = rand(1:env.Ec-1)
    env.Δt = 0.
    env.reward = 0.
end

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

function create_agent(NS, NA, α, d, ϵ)
    approximator = TabularQApproximator(; n_state = NS, n_action = NA, init = 0.)
    learner = SMDPTDLearner(
        approximator = approximator,
        method = :SARSA,
        α = α,
        d = d
    )

    trajectory = Trajectory(
        CircularArraySARJTSATraces(
            capacity = 10,
            state = Int64 => (),
            reward = Float64 => (),
            jump_time = Float64 => (),
            action = Int64 => ()
        ),
        DummySampler()
    )

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

    # for testing the post development performance 
    performance_policy = QBasedPolicy(
        learner = agent.policy.learner.approximator,
        explorer = GreedyExplorer()
    )
    return agent, performance_policy
end

function single_learning_run(juvenile_parameters, adulthood_parameters, developmental_time, maximum_episodes_adulthood, learning_convergence_threshold, opt_value_adult, opt_policy_adult, opt_value_dev, opt_policy_dev, agent, performance_policy)

    # creating the environments and wrapped environments
    dev_env = TwoPreyEnv(; (juvenile_parameters..., d = 0)...)
    state_mapping = s -> s + 1
    
    wrapped_dev_env = StateTransformedEnv(
            dev_env;
            state_mapping = state_mapping,   
            state_space_mapping = _ -> Base.OneTo(NS)
    );
    
    # adulthood environment
    adult_env = TwoPreyEnv(; (adulthood_parameters..., d = 0)...)
    
    wrapped_adult_env = StateTransformedEnv(
            adult_env;
            state_mapping = state_mapping,   
            state_space_mapping = _ -> Base.OneTo(NS)
    );

    # adulthood test environment
    adult_test_env = TwoPreyEnv(; adulthood_parameters...)

    wrapped_adult_test_env = StateTransformedEnv(
            adult_test_env;
            state_mapping = state_mapping,   
            state_space_mapping = _ -> Base.OneTo(NS)
    );
    

    stop_condition_1 = StopAfterNSteps(developmental_time, is_show_progress = false)

    stop_condition_2 = StopWhenValueAndPolicyReaches(; 
        targetValue = opt_value_dev, 
        targetPolicy = opt_policy_dev, 
        threshold = learning_convergence_threshold, 
        max_episodes = maximum_episodes_adulthood,
        show_progress = false
    )
    composite_stop_condition = StopIfAny(stop_condition_1, stop_condition_2)
    # running the learning processes in juvenile stage
    run(
        agent,
        wrapped_dev_env,
        composite_stop_condition,
        EmptyHook()
    ) 

    # measuring the adult performance
    performance_hook = TotalRewardPerEpisode()
    run(
        performance_policy,
        wrapped_adult_test_env,
        StopAfterNEpisodes(10_000, is_show_progress = false),
        performance_hook
    )
    rewards_per_episode = mean(performance_hook.rewards) 

    # measuring relearning time
    stop_condition = StopWhenValueAndPolicyReaches(; 
        targetValue = opt_value_adult, 
        targetPolicy = opt_policy_adult, 
        threshold = learning_convergence_threshold, 
        max_episodes = maximum_episodes_adulthood,
        show_progress = false
    )
    num_steps_adult = StepsEpisodesPerExperiment()

    agent.policy.learner.d = adulthood_parameters.d
    run(
        agent,
        wrapped_adult_env,
        stop_condition,
        num_steps_adult
    ) 
    
    relearning_time_steps = num_steps_adult.steps
    relearning_time_episodes = num_steps_adult.episodes

    return  rewards_per_episode, relearning_time_steps, relearning_time_episodes
end

end # distributed functions

function ensemble_learning_runs(juvenile_parameters, adulthood_parameters, developmental_time, maximum_episodes_adulthood, learning_convergence_threshold, α, ϵ; channel, number_experiments = 1)

    # calculate optimal state value for the adulthood environment
    opt_value_adult, opt_policy_adult = calc_value_policy(; adulthood_parameters...)
    opt_value_dev, opt_policy_dev = calc_value_policy(; juvenile_parameters...)

    # some constant parameters
    NS = juvenile_parameters.Ec + 2
    NA = 3
    d = juvenile_parameters.d

    results = pmap(1:number_experiments) do _
        agent, performance_policy = create_agent(NS, NA, α, d, ϵ)
        res = single_learning_run(juvenile_parameters, adulthood_parameters, developmental_time, maximum_episodes_adulthood, learning_convergence_threshold, opt_value_adult, opt_policy_adult, opt_value_dev, opt_policy_dev, agent, performance_policy)
        put!(channel, true)
        res
    end

    rewards_per_episode = zeros(number_experiments)
    relearning_time_steps = zeros(number_experiments)
    relearning_time_episodes = zeros(number_experiments)
    for i in 1:number_experiments
        rewards_per_episode[i], relearning_time_steps[i], relearning_time_episodes[i] = results[i]
    end

    filtered_indices = relearning_time_episodes .< maximum_episodes_adulthood
    rewards_per_episode = rewards_per_episode[filtered_indices]
    relearning_time_steps = relearning_time_steps[filtered_indices]
    number_experiments_actual = sum(filtered_indices)

    return  mean(rewards_per_episode)     ±   std(rewards_per_episode),
            mean(relearning_time_steps)   ±   std(relearning_time_steps),
            number_experiments_actual
end

function multiple_parameter_learning_runs(pars_iterable, number_experiments)
    num_pars = length(pars_iterable)
    pbar = Progress(num_pars * number_experiments; dt = 10)
    rewards_per_episode = zeros(Measurement, num_pars)
    relearning_time_steps = zeros(Measurement, num_pars)
    number_experiments_actual = zeros(Int64, num_pars)

    channel = RemoteChannel(() -> Channel{Bool}(), 1)

    results = @sync begin
        @async while take!(channel)
            next!(pbar; showvalues = [(:Time,Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))])
        end

        results = @async begin
            res = map(pars -> ensemble_learning_runs(pars...; channel = channel, number_experiments = number_experiments), pars_iterable)
            put!(channel, false)
            res
        end
    end
    results = fetch(results)

    for i in 1:num_pars
        rewards_per_episode[i], relearning_time_steps[i], number_experiments_actual[i] = results[i]
    end

    return  rewards_per_episode, relearning_time_steps, number_experiments_actual
end

# plotting theme
theme = Theme(
    font = "Poppins Regular" ,
    Axis = (;
        xticklabelsize = 14, 
        xticklabelfont = "Poppins Regular",
        yticklabelsize = 14, 
        yticklabelfont = "Poppins Regular",
        xlabelsize = 20, 
        xlabelfont = "Poppins Regular",
        ylabelsize = 20, 
        ylabelfont = "Poppins Regular",
        titlefont = "Poppins Medium", 
        titlesize = 20
        ), 
    Label = (; font = "Poppins Regular"),
    Colorbar = (;labelfont = "Poppins Regular", labelsize = 20)
)
set_theme!(theme)

parameters = ["juvenile_parameters", "adulthood_parameters", "full_factorial", "developmental_time", "learning_convergence_threshold", "number_experiments", "α", "ϵ", "output_file_path", "output_file_name"]

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

env_keys = keys(juvenile_parameters)
juv_values_iterable = collect(product(values(juvenile_parameters)...))[:]
juv_pars_iterable = map(juv_values_iterable) do pars
    NamedTuple{env_keys}(pars)
end

difference = [key => adulthood_parameters[key] 
                for key in keys(juvenile_parameters) 
                    if juvenile_parameters[key] != adulthood_parameters[key]]

env_pars_iterable = map(juv_pars_iterable) do dev_pars 
    adult_pars = (; dev_pars..., difference...)
    (dev_pars, adult_pars)
end

all_pars_iterable = product(env_pars_iterable, developmental_time, maximum_episodes_adulthood, learning_convergence_threshold, α, ϵ)

all_pars_iterable = collect(all_pars_iterable)[:]
all_pars_iterable = map(all_pars_iterable) do (env_pars, rest...)
    (env_pars..., rest...)
end

shuffle!(all_pars_iterable)
## ------------------------------------------------------------------------------
# running the experiments for all the specified parameters
# -------------------------------------------------------------------------------

start = time()

rewards_per_episode, relearning_time_steps, number_experiments_actual = multiple_parameter_learning_runs(all_pars_iterable, number_experiments)

fin = time()
println("Time elapsed: $(round(fin - start; digits = 3)) s")

## -------------------------------------------------------------------------------
# adding the parameters to the dataframe and saving iterable
# --------------------------------------------------------------------------------
types = typeof.(values(first(juv_pars_iterable)))
types = map(type -> type[], types)

all_names = string.(keys(first(juv_pars_iterable)))

juv_names = Symbol.(all_names .* "_juvenile")
adult_names = Symbol.(all_names)

juv_columns = NamedTuple{juv_names}(types)
adult_columns = NamedTuple{adult_names}(types)

columns = (;  
    juv_columns..., 
    adult_columns...,
    developmental_time = Int64[],
    maximum_episodes_adulthood = Int64[],

    learning_convergence_threshold = Float64[],
    learning_rate = Float64[],
    exploration_rate = Float64[],
    number_experiments = Int64[],

    adult_rewards_per_episode = Measurement{Float64}[],
    relearning_time_steps = Measurement{Float64}[],
    number_experiments_actual = Int64[]
)

df = DataFrame(columns)

function unpack_pars(pars) 
    juv_pars, adult_pars, rest... = pars
    return (values(juv_pars)..., values(adult_pars)..., rest...)
end

for (env_pars, rest...) in zip(
    all_pars_iterable, 
    rewards_per_episode,
    relearning_time_steps,
    number_experiments_actual
)
    push!(df, (unpack_pars(env_pars)..., number_experiments, rest...))
end

df = @chain df begin
    @select(:ρ, :ϕ, :Ei_juvenile, :Ei, :Ec, :d, :m, :Es, :Ed, :hs, :hd, :hi, :developmental_time, :maximum_episodes_adulthood, :learning_convergence_threshold, :learning_rate, :exploration_rate, :number_experiments, :number_experiments_actual, :adult_rewards_per_episode, :relearning_time_steps)
    rename(:Ei => "Ei_adult")
end

# calculate and add useful metrics to the dataframe
df_dp = load_data("adult_environment_optimal_performance.arrow")
df_dp = @chain df_dp begin
    @select(:ρ, :ϕ, :Ei, :rewards_per_episode_optimum)
    rename(:Ei => "Ei_adult")
end

df = @chain df begin
    leftjoin(df_dp, on = [:ρ, :ϕ, :Ei_adult])
    @rtransform(:rewards_per_episode_optimum = measurement(:rewards_per_episode_optimum))
    sort(:developmental_time)

    @groupby(:ρ, :ϕ, :Ei_juvenile)
    @transform(
        :scaled_developmental_time = :developmental_time ./ value(:relearning_time_steps[1]),
        :normalised_adult_performance = (:adult_rewards_per_episode .- :adult_rewards_per_episode[1]) ./ (:rewards_per_episode_optimum .- :adult_rewards_per_episode[1]),
        :normalised_relearning_time = :relearning_time_steps ./ :relearning_time_steps[1],
    )
    sort(:Ei_juvenile, rev = true)
    @groupby(:ρ, :ϕ, :developmental_time)
    @transform(
        :relative_adult_performance = :normalised_adult_performance .- :normalised_adult_performance[1],
        :relative_relearning_time = :normalised_relearning_time .- :normalised_relearning_time[1],
    )
end

Arrow.write(output_file_path*output_file_name*".arrow", df)