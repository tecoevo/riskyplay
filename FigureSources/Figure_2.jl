# -------------------------------------------------------------------------------------------------------
# This script uses Dynamic Programming to calculate the optimal policy 
# for a chosen range of parameter values and plot them as a heatmap and save to disk.
# This script might take a few minutes to run on fast personal computers.
# -------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------
# Load required packages
# --------------------------------------------------------------------
using Distributed: @everywhere, nworkers, addprocs, pmap
nworkers() > 1 || addprocs()
using CairoMakie
using TernaryDiagrams
using Base.Iterators
@everywhere using Distributions: Normal, Dirac, truncated, pdf, cdf, ccdf
@everywhere import Distributions
@everywhere using Memoize
@everywhere using LinearAlgebra: dot
using StatsBase: mean, var
using Colors
@everywhere using Interpolations: linear_interpolation
@everywhere using QuadGK: quadgk

## ------------------------------------------------------------------------------
# Discrete normal distribution for energy changes
# It is a normal distribution that is discretised at integer values 
# and then truncated at 0 and Ec - E
# --------------------------------------------------------------------------------
# Calculation functions
@everywhere struct DiscreteNormal
    dist
    function DiscreteNormal(μ, σ)
        d = truncated(Normal(μ, σ); lower = 0.)
        new(d)
    end
end

@everywhere @memoize Distributions.pdf(dist::DiscreteNormal, x::Int)::Float64 = cdf(dist.dist, x+1) - cdf(dist.dist, x) 
@everywhere @memoize Distributions.pdf(dist::DiscreteNormal, x::AbstractArray{Int}) = cdf(dist.dist, collect(x) .+ 1) - cdf(dist.dist, collect(x))
@everywhere @memoize Distributions.ccdf(dist::DiscreteNormal, x::Any) = ccdf(dist.dist, x)
@everywhere @memoize Distributions.cdf(dist::DiscreteNormal, x::Int) = cdf(dist.dist, x+1)
@everywhere @memoize Distributions.cdf(dist::DiscreteNormal, x::AbstractArray{Int}) = cdf(dist.dist, collect(x) .+ 1)
@everywhere @memoize Distributions.ccdf(dist::Dirac, x::Real) = x <= dist.value ? 1.0 : 0.0
@memoize Distributions.mean(dist::DiscreteNormal) = dot(pdf(dist, 0:100), 0:100)
@memoize Distributions.var(dist::DiscreteNormal) = dot(pdf(dist, 0:100), (0:100) .^2 ) - (mean(dist))^2


## ------------------------------------------------------------------------------
# Deterministic dynamic programming functions
# ------------------------------------------------------------------------------
@everywhere function surv_prob(rate, time) 
    exp(-rate*time)
end

@everywhere function single_outcome_return(V, Ec, dist_pdf, dist_ccdf, energy, ::Val{:positive})
    Δes = 1:Ec-energy
    indices = energy+1:Ec
    @views G = dot(dist_pdf[Δes], V[indices]) + dist_ccdf[Ec-energy+1]*V[Ec+1]
    return G
end

@everywhere function single_outcome_return(V, Ec, dist_pdf, dist_ccdf, energy, ::Val{:negative})
    Δes = 1:energy
    indices = energy+1:-1:2
    @views G = dot(dist_pdf[Δes], V[indices])
    return G
end

# Calculate the expected return of taking action [action] when in energy [energy],
# as a sum of expected returns of all the possible outcomes from that action
@everywhere function exp_return(V,Ec, dist_pdfs, dist_ccdfs,e_signs,p,s,action,energy)
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
@everywhere function value_iteration!(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s; tol = 1e-6, maxiter = 10_000)
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
@everywhere function policy_from_optimal_value(V, Ec, dist_pdfs, dist_ccdfs, e_signs, p, s)
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
@everywhere function prob(ρ, ϕ)
    [ 1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
      1-ρ  ρ*ϕ  ρ*(1-ϕ) ;
      1-ρ  ρ    0        ]
end

# Sets the constant compressed matrix for the signs of energy changes 
# for action and outcome, given parameters. 
# The format is the same as the probability matrix above.
@everywhere const e_signs = Val.([ :positive   :positive   :negative ;
                                   :negative   :positive   :negative ;
                                   :positive   :negative   :positive  ])

# Sets the compressed transition time matrix 
# for action and outcome, given parameters. 
# The format is the same as the probability matrix above.
@everywhere function jump_time(hs, hd, hi)
    [ hs + 1   hd + 1   hi + 1 ;
      1        hd + 1   hi + 1 ;
      hs + 1   1        1       ]
end

# Calculates the compressed survival probability matrix 
# for action and outcome, given parameters. 
# The format is the same as the probability matrix above.
@everywhere function surv(d, hs, hd, hi) 
    surv_prob.(d,jump_time(hs, hd, hi))
end

# Sets the compressed matrix of energy distributions for the different 
# actions and outcomes. The distributions show what are the possible 
# energy changes for each given outcome, with a given mean and variance.
# The distributions are discretised normal distributions that are 
# truncated at 0.
# The format is the same as the probability matrix abo
@everywhere function distributions(m, Δm, Es, ΔEs, Ed, ΔEd, Ei, ΔEi)
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
@everywhere function distribution_pdfs(dists, Ec)
    pdf.(dists, (0:Ec,))
end

# Calculates the complementary cumulative distribution functions for all 
# required energy values from the distribution matrix calculated above
# This is required in order to calculated expected values for all cases when 
# energy obtained would increase the energy to Ec or decrease it to 0
@everywhere function distribution_ccdfs(dists, Ec)
    ccdf.(dists, (0:Ec,))
end

# Calculates the optimal value and policy functions given the parameter values of the environment
# Calculates the transition probability matrices and passes this to the main value iteration function.
@everywhere function calc_value_policy(Ec, ρ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, ϕ, Ei, ΔEi, hi; tol = 1e-6, maxiter = 10_000)
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
    return V, policy
end

# Calculates the optimal policy for a grid of parameter values 
# Is parallelized to improve performance
grid_solver(pars_iterable; tol = 1e-6, maxiter = 10_000) =
    pmap(x->calc_value_policy(x...; tol = tol, maxiter = maxiter)[2], pars_iterable)

# Calculates the composition of a given policy. Composition is the fraction
# of energy levels that the action is indiscriminate attack, risk-prone attack and risk-averse attack
function policy_to_composition(policy)
    L = length(policy)
    B = sum(policy .== 1) / L
    D = sum(policy .== 2) / L
    (B,D)
end

## ------------------------------------------------------------------------------
# Setting up the theme for plotting using CairoMakie.jl
# Colors, fonts, sizes, etc.
# ------------------------------------------------------------------------------
color_a = colorant"tomato" #"#ff4e00" #"#FF8C00"                     #dangerous
color_b = colorant"#50f4d5ff" #"#00FF8C"         #both
color_c = colorant"mediumpurple1"  #"#8C00FF"   #safe
color_d = color_a + color_b - color_c
A = [ color_c  color_a ; color_b  color_d ]
color_itp = linear_interpolation((0:1, 0:1), A)

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

## ------------------------------------------------------------------------------
# Choosing the parameters for the calculations
# To reduce the computation time, one can reduce the number of points of ρ and ϕ
# -------------------------------------------------------------------------------

ρ = 0.001:0.001:0.999
d = 0.1
Es = 2.
hs = 1.0
Ed = 10.
hd = 1.
ϕ = 0.001:0.001:0.999
Ei = 4.
hi = 1.

Ec = 20
m = 1
Δm = 0.01
ΔEs = 1.5
ΔEd = 1.5
ΔEi = 1.5

## ------------------------------------------------------------------------------
# Performing the calculations for the central heatmap
# ------------------------------------------------------------------------------

pars_vec = [Ec, ρ, d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, ϕ, Ei, ΔEi, hi]

pars_iterable = collect(product(pars_vec...))
policy_grid = grid_solver(pars_iterable; tol = 1e-6, maxiter = 1_000)
policy_composition_grid = policy_to_composition.(policy_grid)
policy_color_grid = splat(color_itp).(policy_composition_grid)


## ------------------------------------------------------------------------------
# Performing the calculations for the surrounding policies
# ------------------------------------------------------------------------------

calc_value_policy(point::Tuple{<:Real, <:Real}) = calc_value_policy(Ec, point[1], d, m, Δm, Es, ΔEs, hs, Ed, ΔEd, hd, point[2], Ei, ΔEi, hi; tol = 1e-12, maxiter = 10_000)[2]

pointa = (0.1, 0.8)
pointb = (0.1, 0.1)
pointc = (0.97, 0.97)
pointd = (0.75, 0.75)

policya, policyb, policyc, policyd = calc_value_policy.((pointa, pointb, pointc, pointd))

## ------------------------------------------------------------------------------
# Plotting the figure step by step
# ------------------------------------------------------------------------------

## Create the figure and its internal layout
fig = Figure(size = (1300, 500))

ga = fig[1:3, 1] = GridLayout()
gb = fig[1:3, 2:4] = GridLayout()
gc = fig[1:3, 5] = GridLayout()
gd = fig[1:3, 6] = GridLayout()

## Create the axes for plotting the example policies
xlabel = "Energy"
yticks = ([1,2,3],["Generalize", "Specialize\non\nDangerous", "Specialize\non\nSafe"])
yreversed = true
width = 200
yticklabelsize = 10
yaxisposition = :right

axa = Axis(ga[1, 1]; yticks, yreversed, width, yticklabelsize) 
axb = Axis(ga[2, 1]; xlabel, yticks, yreversed, yticklabelsize)
axc = Axis(gd[1, 1]; yticks, yreversed, width, yticklabelsize, yaxisposition) 
axd = Axis(gd[2, 1]; xlabel, yticks, yreversed, width, yticklabelsize, yaxisposition) 

# plotting the example policies
scatter!.((axa, axb, axc, axd), (policya, policyb, policyc, policyd); markersize = 10)

limits!.((axa, axb, axc, axd), -1, Ec+1, 3.25, 0.75)

## plotting the central heatmap
ax1 = Axis(gb[1, 1]; ylabel = rich("Capture probability ", rich("ϕ", font = :italic)), xlabel = rich("Dangerous prey availability ", rich("ρ", font = :italic)), aspect = DataAspect())
hm = heatmap!(ax1, ρ, ϕ, policy_color_grid; rasterize = 2)

## Creating the ternary colormap legend for heatmap 
function ternary_to_cartesian(a, b)
    c = 1 - a - b
    x = 0.5 * (2b + c) / (a + b + c)
    y = sqrt(3)/2 * c / (a + b + c)
    return (x, y)
end

hexagon = Makie.Polygon([Point2f(cos(a), sin(a)) for a in range(1/6 * pi, 13/6 * pi, length = 7)])
g = range(0, 1, 100)
gap = (g[2]-g[1])/2 + 1e-3
grid = collect(product(g, g))
indices = map(x -> sum(x) <= 1.0, grid)
filtered_grid = grid[indices]
rotate_colors = (x, y) -> (1 - x - y, y)
cartesian_grid = splat(ternary_to_cartesian).(splat(rotate_colors).(filtered_grid))
x = getindex.(cartesian_grid, 1)
y = getindex.(cartesian_grid, 2)
colors = splat(color_itp).(filtered_grid)

ax2 = Axis(gc[1,1]; aspect = DataAspect())
hidedecorations!(ax2)
hidespines!(ax2)
scatter!(ax2, x, y; color = colors, marker = hexagon, markersize = gap / cos(deg2rad(30)), markerspace = :data, rasterize = 2)
trax = ternaryaxis!(
    ax2; 
    label_fontsize = 8,
    grid_line_width = 0.0,
    grid_line_color = :white,
    tick_fontsize = 4,
    label_vertex_vertical_adjustment = 0.1,
    arrow_label_fontsize = 0,
    hide_vertex_labels = true
)
for obj in trax.plots
    if obj isa Arrows
        obj.visible[] = false
    end
end

text!(ax2, 0.5, 1.; text = "Always generalize", markerspace = :data, fontsize = 0.08, align = (:center, :center))
text!(ax2, 0.95, -0.05; text = "Always\ndangerous", markerspace = :data, fontsize = 0.08, align = (:left, :top))
text!(ax2, 0, -0.05; text = "Always\nsafe", markerspace = :data, fontsize = 0.08, align = (:right, :top))

limits!(ax2, -0.3, 1.45, 0 -1, 1 + 1)

colsize!(fig.layout, 5, Relative(0.2))

colgap!(fig.layout, 4, 5)
colgap!(fig.layout, 5, 10)

## drawing lines from points to example heatmaps
function figure_to_axis_coords(fig_point, ax)
    fig_scene = ax.parent.scene 
    axis_scene = ax.scene
    fig_screen_pt = Point2f(fig_point)
    viewport = axis_scene.viewport[]
    axis_screen_x = fig_point[1] - viewport.origin[1]
    axis_screen_y = fig_point[2] - viewport.origin[2]
    axis_screen_pt = Point2f(axis_screen_x, axis_screen_y)
    data_pt = Makie.to_world(axis_scene, axis_screen_pt)
    return data_pt
end

function shrink_distance(anchor, moving, shrinkby)
    distance = sqrt( sum( (moving - anchor).^2 ) )
    cosθsinθ = (moving - anchor)/distance
    shifted_point = anchor + (distance - shrinkby) * cosθsinθ
    return shifted_point
end

# drawing the lines from the points on the heatmap to the example policies
pointa, pointb, pointc, pointd = Point2f.((pointa, pointb, pointc, pointd))
pointa_fig, pointb_fig, pointc_fig, pointd_fig = Makie.shift_project.(ax1.scene, (pointa, pointb, pointc, pointd))
anchora = Point2f(22, 2)
anchorb = Point2f(22, 2)
anchorc = Point2f(-2, 2)
anchord1 = Point2f(1.01, 0.25)
anchord2 = Point2f(-2, 2)

anchora_fig, anchorb_fig, anchorc_fig, anchord1_fig, anchord2_fig = 
    Makie.shift_project.(
        (axa.scene, axb.scene, axc.scene, ax1.scene, axd.scene), 
        (anchora, anchorb, anchorc, anchord1, anchord2)
    )

pointa_shifted_fig, pointb_shifted_fig, pointc_shifted_fig, pointd_shifted_fig = 
    shrink_distance.(
        (anchora_fig, anchorb_fig, anchorc_fig, anchord1_fig),
        (pointa_fig, pointb_fig, pointc_fig, pointd_fig), 
        8
    )

pointa_shifted_ax, pointb_shifted_ax, pointc_shifted_ax, pointd_shifted_ax = 
    figure_to_axis_coords.(
        (pointa_shifted_fig, pointb_shifted_fig, pointc_shifted_fig, pointd_shifted_fig), 
        ax1
    )

anchora_ax, anchorb_ax, anchorc_ax, anchord_ax = 
    figure_to_axis_coords.(
        (anchora_fig, anchorb_fig, anchorc_fig, anchord1_fig), 
        ax1
    )

map(zip((pointa_shifted_fig, pointb_shifted_fig, pointc_shifted_fig), (anchora_fig, anchorb_fig, anchorc_fig))) do (p, a)
    lines!(fig.scene, [a, p]; color = :black, linewidth = 3)
end
lines!(fig.scene, [pointd_shifted_fig, anchord1_fig, anchord2_fig]; color = :black, linewidth = 3)

map(zip((pointa_shifted_ax, pointb_shifted_ax, pointc_shifted_ax, pointd_shifted_ax), (anchora_ax, anchorb_ax, anchorc_ax, anchord_ax))) do (p, a)
    lines!(ax1, [a, p]; color = :black, linewidth = 3)
end
xlims!(ax1, 0.01, 0.99)

scatter!.(ax1, (pointa, pointb, pointc, pointd); color = :transparent, strokecolor = :black, strokewidth = 3, markersize = 20)
scatter!(fig.scene, pointc_fig; color = :transparent, strokecolor = :black, strokewidth = 3, markersize = 20)

display(fig)

# ------------------------------------------------------------
# Save the figure to disk
# ------------------------------------------------------------
save("Figure_2.pdf", fig)

