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

# load the dataset
df = load_data("../Datasets/protected_environment_learning.arrow")

# learning time curves for all parameter values
Eis = 3:-1:0
scaled_developmental_times = 0.1:0.1:1.0
fig = Figure(size = (2500, 1000))
for (j, dev_time) in enumerate(scaled_developmental_times)
    df2 = subset_scaled_developmental_time(df, dev_time)
    for (i, Ei) in enumerate(Eis)
        df3 = @rsubset(df2, :Ei_juvenile == Ei)
        ax = Axis(fig[i,j]; xlabelpadding = 10, xticklabelpad = -2, yaxisposition = :right, xaxisposition = :top, ylabelpadding = 10, yticklabelpad = 3, xticks = 0.2:0.2:0.8, yticks = 0.2:0.2:0.8, xlabelsize = 25, ylabelsize = 25, xticklabelsize = 20, yticklabelsize = 20)
        Ei == 3 && j == 5 && (ax.xlabel = rich("Dangerous prey abundance ", rich("ρ", font = :italic), offset = (9, 0)))
        j == 10 && Ei == 1 && (ax.ylabel = rich("Capture probability ", rich("ϕ", font = :italic), offset = (8, 0)))
        Ei != first(Eis)  && hidexdecorations!(ax)
        dev_time != last(scaled_developmental_times) && hideydecorations!(ax)
        global hm = heatmap!(ax, df3.ρ, df3.ϕ, df3.relative_relearning_time, colormap = :seismic, colorrange = (-0.5, 0.5))
    end
end

num_plots_x = length(scaled_developmental_times)
Colorbar(fig[:, num_plots_x+1], hm, label = "Difference in relearning time", labelsize = 26, ticklabelsize = 21, width = 20)

length_of_axis = 10
length_of_plot = length_of_axis / num_plots_x
ticks_positions_x = (length_of_plot/2):length_of_plot:(10-length_of_plot/2)

leftax = Axis(fig[1:4, 0], width = 0, ylabel = "Protection level of juvenile environment", yticks = ([1, 3, 5, 7], string.([4, 3, 2, 1])), ylabelsize = 40, yticklabelsize = 26, yticklabelpad = 5, spinewidth = 2, ytickwidth = 2)
ylims!(leftax, 0, 8)
hidespines!(leftax, :t, :r, :b)
hidedecorations!(leftax, ticks = false, ticklabels = false, label = false)
hidexdecorations!(leftax)

bottomax = Axis(fig[5, 1:num_plots_x], height = 0, xlabel = "Scaled developmental time", xticks = (ticks_positions_x, string.(scaled_developmental_times)), xlabelsize = 40, xticklabelsize = 26, spinewidth = 2, xtickwidth = 2)
hidespines!(bottomax, :t, :r, :l)
hidedecorations!(bottomax, ticks = false, ticklabels = false, label = false)
hideydecorations!(bottomax)

colgap!(fig.layout, 10)
rowgap!(fig.layout, 10)
rowgap!(fig.layout, 4, 15)
colgap!(fig.layout, 1, 15)
colgap!(fig.layout, 11, 20)

points = lift(px -> [px], @lift($(leftax.xaxis.attributes.endpoints)[1]))
directions = [Vec2f(0, -1)]
arrows!(leftax.parent.scene, points, directions; arrowsize = 15)

points = lift(px -> [px], @lift($(bottomax.xaxis.attributes.endpoints)[2]))
directions = [Vec2f(1, 0)]
arrows!(leftax.parent.scene, points, directions; arrowsize = 15)

display(fig)

save("Figure_A3.pdf", fig)