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

# representative examples of performance curves

df = load_data("../Datasets/protected_environment_learning.arrow")
df_dang = @rsubset(df, :ρ == 0.9, :ϕ == 0.6, :Ei_juvenile == 4)
df_safe = @rsubset(df, :ρ == 0.9, :ϕ == 0.6, :Ei_juvenile == 0)
df_safe_2 = @rsubset(df, :ρ == 0.5, :ϕ == 0.2, :Ei_juvenile == 0, :scaled_developmental_time <= 2.1)
sort!(df_dang, :scaled_developmental_time)
sort!(df_safe, :scaled_developmental_time)
sort!(df_safe_2, :scaled_developmental_time)

ylims = (0, 1.1)
xlims = (0, nothing)
yticks = [0, 1]
xticks = 0:2
ygridcolor = (:black, 0.7)
fig = Figure(size = (1200, 440))

x, y = df_dang.scaled_developmental_time, df_dang.normalised_adult_performance
ax1 = Axis(fig[1,1]; ylabel = "Adult performance", title = "Dangerous juv. env.", yticks, xticks, ygridcolor, ylabelpadding = 0)
vlines!(ax1, [0.1, maximum(x)]; color = :black, linewidth = 2)
lines!(ax1, x, y; label = "Simulations", linewidth = 3, color = :darkorange1)
band!(ax1, x, y; color = (:darkorange1, 0.3))
xlims = (-0.1, maximum(x) + 0.1)
limits!(ax1, xlims..., ylims...)

x, y = df_safe.scaled_developmental_time, df_safe.normalised_adult_performance
ax2 = Axis(fig[1,2]; xlabel = "Scaled Developmental Time", title = "Protected juv. env.", yticks, xticks, ygridcolor, xlabelpadding = 0)
vlines!(ax2, [0.1, maximum(x)]; color = :black, linewidth = 2)
lines!(ax2, x, y; label = "Simulations", linewidth = 3, color = :magenta4)
band!(ax2, x, y; color = (:magenta4, 0.3))
hideydecorations!(ax2; grid = false, minorgrid = false)
xlims = (-0.1, maximum(x) + 0.1)
limits!(ax2, xlims..., ylims...)

x, y = df_safe_2.scaled_developmental_time, df_safe_2.normalised_adult_performance
ax3 = Axis(fig[1,3]; title = "Protected juv. env. 2", yticks, xticks, ygridcolor)
vlines!(ax3, [0.1, maximum(x)]; color = :black, linewidth = 2)
lines!(ax3, x, y; label = "Simulations", linewidth = 3, color = :magenta4)
band!(ax3, x, y; color = (:magenta4, 0.3))
hideydecorations!(ax3; grid = false, minorgrid = false)
xlims = (-0.1, maximum(x) + 0.1)
limits!(ax3, xlims..., ylims...)

hidexdecorations!.((ax1, ax2, ax3); label = false, ticklabels = false, ticks = false, minorticks = true)

for (pos, label) in zip(1:3, ["A", "B", "C"])
    Label(
        fig[1, pos, TopLeft()], label,
        fontsize = 21,
        font = :bold,
        padding = (0, 0, 0, 0),
        halign = :right,
        width = 0 
    )
end

text!.((ax1, ax2, ax3), 0.15, 0.02; text = "(ii)", align = (:left, :bottom), fontsize = 18)
text!.((ax1, ax2, ax3), 2.05, 0.02; text = "(i)", align = (:right, :bottom), fontsize = 18)
        
display(fig)

save("Figure_3.pdf", fig)