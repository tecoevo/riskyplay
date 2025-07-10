# load packages and setup
using DataFrames
using DataFramesMeta
using Arrow
using CairoMakie
using Chain
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

# maximum reward post juvenile stage
df_max_reward = @chain df begin
    @groupby(:ρ, :ϕ, :Ei_juvenile)
    @combine(:asymptotic_adult_performance = maximum(:normalised_adult_performance))
    sort(:Ei_juvenile; rev = true)
    @groupby(:ρ, :ϕ)
    @transform(:relative_asymptotic_adult_performance = :asymptotic_adult_performance .- :asymptotic_adult_performance[1])
    @rsubset(:Ei_juvenile != 4)
end

fig = Figure(size = (1200, 710))
colorrange = (-0.5, 0.5)
colormap = :seismic
xticks = 0.2:0.2:0.8
yticks = 0.2:0.2:0.8
aspect = DataAspect()

Ei_juvenile = 3
df3 = @rsubset(df_max_reward, :Ei_juvenile == Ei_juvenile)
ax1 = Axis(fig[1,1]; ylabel = rich("Capture probability ", rich("ϕ", font = :italic)), xticks, yticks, xticklabelpad = 0, aspect, title = "A. After extended juvenile phase", titlealign = :left)
hm = heatmap!(ax1, df3.ρ, df3.ϕ, df3.relative_asymptotic_adult_performance; colormap, colorrange)

Ei_juvenile = 2
df3 = @rsubset(df_max_reward, :Ei_juvenile == Ei_juvenile)
ax2 = Axis(fig[1,2]; xlabel = rich("Dangerous prey abundance ", rich("ρ", font = :italic), offset = (15, 0)), xticks, xlabelpadding = 0, aspect)
hm = heatmap!(ax2, df3.ρ, df3.ϕ, df3.relative_asymptotic_adult_performance; colormap, colorrange)

Ei_juvenile = 1
df3 = @rsubset(df_max_reward, :Ei_juvenile== Ei_juvenile)
ax3 = Axis(fig[1,3]; xticks, aspect)
hm = heatmap!(ax3, df3.ρ, df3.ϕ, df3.relative_asymptotic_adult_performance; colormap, colorrange)

Ei_juvenile = 0
df3 = @rsubset(df_max_reward, :Ei_juvenile == Ei_juvenile)
ax4 = Axis(fig[1,4]; xticks, aspect)
hm = heatmap!(ax4, df3.ρ, df3.ϕ, df3.relative_asymptotic_adult_performance; colormap, colorrange)

topax = Axis(fig[0, 1:4], height = 0, xlabel = "Protection level of juvenile environment", xticks = ([1.215, 3.74, 6.27, 8.79], string.([1, 2, 3, 4])), xaxisposition = :top, xlabelpadding = -5, xticklabelpad = 0)
hideydecorations!(topax)

hideydecorations!.((ax2, ax3, ax4), grid = false, minorgrid = false)
hidexdecorations!.((ax1, ax2, ax3, ax4); ticks = false)

points = lift(px -> [px], @lift($(topax.xaxis.attributes.endpoints)[2]))
directions = [Vec2f(1, 0)]
arrows!(topax.parent.scene, points, directions)

# reward post short  juvenile stage

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

df_early_reward = subset_scaled_developmental_time(df, 0.1)

Ei_juvenile = 3
df3 = @rsubset(df_early_reward, :Ei_juvenile == Ei_juvenile)
ax1 = Axis(fig[2,1]; ylabel = rich("Capture probability ", rich("ϕ", font = :italic)), xticks, yticks, xticklabelpad = -0, aspect, title = "B. After short juvenile phase", titlealign = :left)
hm = heatmap!(ax1, df3.ρ, df3.ϕ, df3.relative_adult_performance; colormap, colorrange)

Ei_dev = 2
df3 = @rsubset(df_early_reward, :Ei_juvenile == Ei_juvenile)
ax2 = Axis(fig[2,2]; xlabel = rich("Dangerous prey abundance ", rich("ρ", font = :italic), offset = (15, 0)), xticks, xlabelpadding = 0, aspect)
hm = heatmap!(ax2, df3.ρ, df3.ϕ, df3.relative_adult_performance; colormap, colorrange)

Ei_dev = 1
df3 = @rsubset(df_early_reward, :Ei_juvenile == Ei_juvenile)
ax3 = Axis(fig[2,3]; xticks, aspect)
hm = heatmap!(ax3, df3.ρ, df3.ϕ, df3.relative_adult_performance; colormap, colorrange)

Ei_dev = 0
df3 = @rsubset(df_early_reward, :Ei_juvenile == Ei_juvenile)
ax4 = Axis(fig[2,4]; xticks, aspect)
hm = heatmap!(ax4, df3.ρ, df3.ϕ, df3.relative_adult_performance; colormap, colorrange)
Colorbar(fig[1:2,5], hm, label = "Relative adult performance", labelsize = 21, ticklabelsize = 18, alignmode = Outside())

hideydecorations!.((ax2, ax3, ax4), grid = false, minorgrid = false)
rowsize!(fig.layout, 1, Aspect(1, 1))
rowsize!(fig.layout, 2, Aspect(1, 1))
colgap!(fig.layout, 10)
rowgap!(fig.layout, 5)

display(fig)

save("Figure_4.pdf", fig)

