# load packages and setup
using DataFrames
using DataFramesMeta
using Arrow
using CairoMakie
using Chain
using StatsBase
using LsqFit
using Measurements
using Measurements: value, uncertainty

# plotting theme
theme = Theme(
    font = "Poppins Regular" ,
    fonts = (; regular = "Poppins Regular", bold = "Poppins Medium"),
    Axis = (;
        xticklabelsize = 18, 
        xticklabelfont = "Poppins Regular",
        yticklabelsize = 18, 
        yticklabelfont = "Poppins Regular",
        xlabelsize = 21, 
        xlabelfont = "Poppins Regular",
        ylabelsize = 21, 
        ylabelfont = "Poppins Regular",
        titlefont = "Poppins Medium", 
        titlesize = 21,
        xticklabelpad = 0,
        yticklabelpad = 0,
        ), 
    Label = (; font = "Poppins Regular"),
    Colorbar = (;labelfont = "Poppins Regular", labelsize = 21, ticklabelsize = 18),
    axislegend = (;labelfont = "Poppins Regular", labelsize = 18)
)
set_theme!(theme)

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

# functions for plotting with measurement values
function Makie.convert_arguments(T2::Type{<: Band}, x::AbstractArray, y::AbstractArray{Measurement{T}}) where {T} 
    Makie.convert_arguments(T2, value.(x), value.(y) .- uncertainty.(y), value.(y) .+ uncertainty.(y))
end

function Makie.convert_single_argument(x::AbstractArray{Measurement{T}}) where {T}
    value.(x)
end

Base.typemin(::Type{Measurement{T}}) where {T <: AbstractFloat} = typemin(T) ± zero(T)

# load the dataset
df = load_data("../Datasets/protected_environment_learning.arrow")

# functions for fitting exponential decay
model_exp_decay(x, p) = exp.(-p[1]*x)

function fit_exponential_decay(x, y::AbstractVector{<:Measurement})
    # x_new = value.(x)
    x_new = x
    x_max = maximum(x_new)
    y_max = maximum(y)
    x_norm = x_new ./ x_max
    y_norm = y ./ y_max
    fit = curve_fit(model_exp_decay, x_norm, y_norm, [1. ± 0.])
    p = fit.param[1]/x_max
    return p
end

function relative_learning_speeds(df)
    df_learning_speeds = @chain df begin
        @groupby(:ρ, :ϕ, :Ei_juvenile)
        @combine(:decay_rate = fit_exponential_decay(:developmental_time, :relearning_time_steps))
        sort(:Ei_juvenile, rev = true)
        @groupby(:ρ, :ϕ)
        @transform(:decay_rate_reference = :decay_rate[1])
        @rtransform(:relative_decay_rate = :decay_rate ./ :decay_rate_reference)
        @rtransform(:relative_decay_rate_log = log10(:relative_decay_rate))
    end

    return df_learning_speeds
end

df_relative_learning_speeds = @chain df begin
    @rsubset(:scaled_developmental_time <= 1.0)
    relative_learning_speeds()
    @rsubset(:Ei_juvenile != 4)
    @rtransform(:relative_learning_speed = :relative_decay_rate_log)
end

df_relative_asymptotic_performance = @chain df begin
    @groupby(:ρ, :ϕ, :Ei_juvenile)
    @combine(:asymptotic_adult_performance = maximum(:normalised_adult_performance))
    sort(:Ei_juvenile; rev = true)
    @groupby(:ρ, :ϕ)
    @transform(:relative_asymptotic_adult_performance = :asymptotic_adult_performance .- :asymptotic_adult_performance[1])
    @rsubset(:Ei_juvenile != 4)
end

function subset_scaled_developmental_time(df, value)
    @chain df begin
        @rtransform(:difference = abs(:scaled_developmental_time - value))
        sort(:difference)
        @rsubset(:difference <= 0.1)
        @groupby(:ρ, :ϕ, :Ei_juvenile)
        combine(first, _)
        @select(Not([:difference]))
    end
end

df_relative_early_performance = @chain df begin
    subset_scaled_developmental_time(0.1)
    @rtransform(:relative_early_adult_performance = :relative_adult_performance)
end 

function compare_three(a, b, c; lthres = 0, mthres = 0, rthres = 0)
    s1 = a <= lthres ? "0" : "1"
    s2 = b <= mthres ? "0" : "1"
    s3 = c <= rthres ? "0" : "1"
    s = parse(UInt, s3 * s2 * s1; base = 2)
    return Int(s)
end

df_all = innerjoin(df_relative_learning_speeds, df_relative_asymptotic_performance, df_relative_early_performance; on = [:ρ, :ϕ, :Ei_juvenile], makeunique = true)
df_all = @rtransform(df_all, :safe_environment_benefit = compare_three(:relative_early_adult_performance, :relative_asymptotic_adult_performance, :relative_learning_speed; lthres = 0.02, mthres = -0.01, rthres = 0.00))


colorrange = (0, 7)
colormap = cgrad(:jet, 8, categorical = true)
xticks = 0.2:0.2:0.8
yticks = 0.2:0.2:0.8
aspect = DataAspect()

fig = Figure(size = (1600, 450))

Ei_juvenile = 3
df3 = @rsubset(df_all, :Ei_juvenile == Ei_juvenile)
ax1 = Axis(fig[1,1]; ylabel = rich("Capture probability ", rich("ϕ", font = :italic)), xticks, yticks, aspect)
hm = heatmap!(ax1, df3.ρ, df3.ϕ, df3.safe_environment_benefit; colormap, colorrange)

Ei_juvenile = 2
df3 = @rsubset(df_all, :Ei_juvenile == Ei_juvenile)
ax2 = Axis(fig[1,2];  xlabel = rich("Dangerous prey abundance ", rich("ρ", font = :italic), offset = (15, 0)), xticks, aspect)
hm = heatmap!(ax2, df3.ρ, df3.ϕ, df3.safe_environment_benefit; colormap, colorrange)

Ei_juvenile = 1
df3 = @rsubset(df_all, :Ei_juvenile == Ei_juvenile)
ax3 = Axis(fig[1,3]; xticks, aspect)
hm = heatmap!(ax3, df3.ρ, df3.ϕ, df3.safe_environment_benefit; colormap, colorrange)

Ei_juvenile = 0
df3 = @rsubset(df_all, :Ei_juvenile == Ei_juvenile)
ax4 = Axis(fig[1,4]; xticks, aspect)
hm = heatmap!(ax4, df3.ρ, df3.ϕ, df3.safe_environment_benefit; colormap, colorrange)
Colorbar(fig[1,5], hm, ticks = (0:7, ["No benefit", "Only early perf.", "Only max perf.", "Early & max perf.", "Only learning speed", "Learning speed & early perf.", "Learning speed & max perf.", "All"]))

topax = Axis(fig[0, 1:4], height = 0, xlabel = "Protection level of juvenile environment", xticks = ([1.215, 3.74, 6.27, 8.79], string.([1, 2, 3, 4])), xaxisposition = :top, xlabelpadding = -5, xticklabelpad = 0)
hideydecorations!(topax)

hideydecorations!.((ax2, ax3, ax4), grid = false, minorgrid = false)
rowsize!(fig.layout, 1, Aspect(1, 1))
colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)

points = lift(px -> [px], @lift($(topax.xaxis.attributes.endpoints)[2]))
directions = [Vec2f(1, 0)]
arrows!(topax.parent.scene, points, directions)

display(fig)

save("Figure_7.pdf", fig)