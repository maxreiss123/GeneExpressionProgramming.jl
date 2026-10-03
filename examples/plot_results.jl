#=
Optional plotting for the examples, which include this file only when Plots.jl is
installed, so they run without it.
=#
using Plots

"""
    plot_results(regressor, pred, y; file=nothing)

Predicted against actual values, and the training and validation loss per epoch.
Displayed, or written to `file` when one is given.
"""
function plot_results(regressor, pred, y; file=nothing)
    parity = scatter(vec(y), vec(pred);
        xlabel="Actual Values", ylabel="Predicted Values", label="Predictions",
        title="Predictions vs Actual - Symbolic Regression")
    plot!(parity, vec(y), vec(y); label="Prediction = Actual", color=:red)

    # epochs after an early stop are not recorded, and a log scale cannot show a zero loss
    history = regressor.fitness_history_
    recorded(v) = [max(v[i][1], 1e-16) for i in eachindex(v) if isassigned(v, i)]
    losses = plot(recorded(history.train_loss);
        label="Training Loss", ylabel="Loss", xlabel="Epoch", linewidth=2, yscale=:log10)
    plot!(losses, recorded(history.val_loss); label="Validation Loss", linewidth=2)

    fig = plot(parity, losses; layout=(1, 2), size=(1000, 400))
    isnothing(file) ? display(fig) : savefig(fig, file)
    return fig
end
