# load packages and setup
using DataFrames
using Arrow
using CairoMakie
using Colors
using Interpolations
using Measurements
using Measurements: value, uncertainty
using Base.Iterators: product
using TernaryDiagrams

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
Measurements.value(x::Any) = x

# converting policy into color
color_a = colorant"tomato" #"#ff4e00" #"#FF8C00"                     #dangerous
color_b = colorant"#50f4d5ff" #"#00FF8C"         #both
color_c = colorant"mediumpurple1"  #"#8C00FF"   #safe
color_d = color_a + color_b - color_c
A = [ color_c  color_a ; color_b  color_d ]
color_itp = linear_interpolation((0:1, 0:1), A);
composition_to_color((a, b, c)) = color_itp(a,b);
composition_to_color(a) = color_itp(a[1], a[2])

# load dataset of optimal policy learnt by reinforcement learning
df = load_data("../Datasets/optimality_of_RL.arrow")

x = unique(df.ρ)
n = length(x)
color_grid =  reshape(composition_to_color.(df.policy_composition), (n, n))

fig = Figure(size = (1200, 480))

### policy heatmap
ax1 = Axis(fig[1,1]; xlabel = rich("Dangerous prey abundance ", rich("ρ", font = :italic)), ylabel = rich("Capture probability ", rich("ϕ", font = :italic)), xticks = 0.2:0.2:0.8, yticks = 0.2:0.2:0.8, title = "Optimal policy", aspect = DataAspect())
hm = heatmap!(ax1, x, x, color_grid)

### legend for heatmap
ax2 = Axis(fig[1,2]; aspect = DataAspect())

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

hidedecorations!(ax2)
hidespines!(ax2)
scatter!(ax2, x, y; color = colors, marker = hexagon, markersize = gap / cos(deg2rad(30)), markerspace = :data, rasterize = 2)
trax = ternaryaxis!(
    ax2; 
    label_fontsize = 10,
    grid_line_width = 0.0,
    grid_line_color = :white,
    tick_fontsize = 6,
    label_vertex_vertical_adjustment = 0.1,
    arrow_label_fontsize = 0,
    hide_vertex_labels = true
)
for obj in trax.plots
    if obj isa Arrows
        obj.visible[] = false
    end
end

text!(ax2, 0.5, 1.1; text = "Always\nindiscriminate", markerspace = :pixel, fontsize = 10, align = (:center, :center))
text!(ax2, 1, -0.12; text = "Always\nrisk-prone", markerspace = :pixel, fontsize = 10, align = (:center, :top))
text!(ax2, -0., -0.12; text = "Always\nrisk-averse", markerspace = :pixel, fontsize = 10, align = (:center, :top))

limits!(ax2, -0.32, 1.4, -0.4, 1.25)

### learning time heatmap

ax3 = Axis(fig[1,3]; xlabel = rich("Dangerous prey abundance ", rich("ρ", font = :italic)), ylabel = rich("Capture probability ", rich("ϕ", font = :italic)), xticks = 0.2:0.2:0.8, yticks = 0.2:0.2:0.8, title = "Learning time", aspect = DataAspect())
hm = heatmap!(ax3, df.ρ, df.ϕ, df.learning_time_steps, colorscale = log10)
Colorbar(fig[1,4], hm; label = "Learning time in steps")

rowsize!(fig.layout, 1, Aspect(1, 1))
colsize!(fig.layout, 2, 200)
colgap!(fig.layout, 1, 10)
colgap!(fig.layout, 3, 10)

for (pos, label) in zip([1,3], ["A", "B"])
    Label(
        fig[1, pos, TopLeft()], label,
        fontsize = 21,
        font = :bold,
        padding = (0, 0, 0, 0),
        halign = :right,
        width = 0 
    )
end

display(fig)

save("Figure_A1.pdf", fig)