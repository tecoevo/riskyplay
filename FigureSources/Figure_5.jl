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

# load the dataset
df = load_data("../Datasets/protected_environment_learning.arrow")

# selecting relevant data from the dataset and plotting
df_dang = @rsubset(df, :ρ == 0.9, :ϕ == 0.6, :Ei_juvenile == 4)
df_safe = @rsubset(df, :ρ == 0.9, :ϕ == 0.6, :Ei_juvenile == 0)
df_safe_2 = @rsubset(df, :ρ == 0.8, :ϕ == 0.2, :Ei_juvenile == 0)
sort!(df_dang, :scaled_developmental_time)
sort!(df_safe, :scaled_developmental_time)
sort!(df_safe_2, :scaled_developmental_time)

ylims = (0, 1.6)
xlims = (0, nothing)
xticks = 0:4
yticks = [0, 1]
ygridcolor = (:black, 0.7)

fig = Figure(size = (1200, 440))

x, y = df_dang.scaled_developmental_time, df_dang.normalised_relearning_time 
ax1 = Axis(fig[1,1]; ylabel = "Re-learning Time", title = "Dangerous juv. env.", xticks, yticks, ygridcolor)
lines!(ax1, x, y; linewidth = 3, color = :darkorange1)
band!(ax1, x, y; color = (:darkorange1, 0.3))
xlims = extrema(x)
limits!(ax1, xlims..., ylims...)

x, y = df_safe.scaled_developmental_time, df_safe.normalised_relearning_time
ax2 = Axis(fig[1,2]; xlabel = "Scaled developmental Time", title = "Protected juv. env.", xticks, yticks, ygridcolor)
lines!(ax2, x, y; linewidth = 3, color = :magenta4)
band!(ax2, x, y; color = (:magenta4, 0.3))
hideydecorations!(ax2; grid = false, minorgrid = false)
xlims = extrema(x)
limits!(ax2, xlims..., ylims...)

x, y = df_safe_2.scaled_developmental_time, df_safe_2.normalised_relearning_time
ax3 = Axis(fig[1,3]; title = "Protected juv. env. 2", xticks, yticks, ygridcolor)
lines!(ax3, x, y; linewidth = 3, color = :magenta4)
band!(ax3, x, y; color = (:magenta4, 0.3))
hideydecorations!(ax3; grid = false, minorgrid = false)
limits!(ax3, xlims..., ylims...)

hidexdecorations!.((ax1, ax2, ax3); label = false, ticklabels = false, ticks = false, minorticks = false)

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

display(fig)

save("Figure_5.pdf", fig)