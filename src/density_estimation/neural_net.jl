### Categorical MLP

"""

Creates a sequence of hidden layers.
Note: The last hidden layer has no activation function
"""
function hidden_layers(;input_size=1, hidden_sizes=(32,), activation=relu)
    length(hidden_sizes) == 1 && return Dense(input_size, hidden_sizes[1])
    last = Dense(hidden_sizes[end-1], hidden_sizes[end])
    ins = vcat(input_size, hidden_sizes[1:end-2]...)
    outs = collect(hidden_sizes[1:end-1])
    return Chain((Chain(Dense(in, out, activation), Dropout(0.2)) for (in, out) in zip(ins, outs))..., last)
end

CategoricalMLP(;input_size=1, hidden_sizes=(32, 2)) =
    hidden_layers(input_size=input_size, hidden_sizes=hidden_sizes)

compute_loss(model, x, y::OneHotMatrix) = Flux.logitcrossentropy(model(x), y)

function sample_from(model::Chain, X, labels)
    ps = softmax(model(X))
    indicators = [rand(Distributions.Categorical(collect(p))) for p in eachcol(ps)]
    return categorical([labels[i] for i in indicators], levels=labels)
end

function TMLE.expected_value(model::Chain, X, labels)
    p = softmax(model(X))
    return transpose(labels)*p
end

### Mixture Density Network

"""
Implements a Mixture of Gaussians Density Network.

Attributes:

- p_k_x: conditional probability of k giuven x
- σ_xk: standard deviation of the k-th gaussian as a function of x
- μ_xk: mean of the k-th gaussian as a function of x
"""
struct MixtureDensityNetwork
    p_k_x::Chain
    σ_xk::Chain
    μ_xk::Chain
end

function MixtureDensityNetwork(;input_size=1, hidden_sizes=(20,), K=3)
    last_hidden_size = hidden_sizes[end]
    z_h = Chain(hidden_layers(input_size=input_size, hidden_sizes=hidden_sizes), x -> tanh.(x))
    z_k = Dense(last_hidden_size, K)
    z_σ = Dense(last_hidden_size, K, exp)
    z_μ = Dense(last_hidden_size, K)
    p_k_x = Chain(z_h, z_k, softmax)
    σ_xk = Chain(z_h, z_σ)
    μ_xk = Chain(z_h, z_μ)
    return MixtureDensityNetwork(p_k_x, σ_xk, μ_xk)
end

exponent(y, μ, σ, p_k_x) =
    @. log(p_k_x) - log(sqrt(2π)) - log(σ) - 0.5((y - μ) / σ)^2

function (model::MixtureDensityNetwork)(x, y)
    p_k_x = model.p_k_x(x)
    σ_xk = model.σ_xk(x)
    μ_xk = model.μ_xk(x)
    l = p_k_x.*gaussian_kernel(y, μ_xk, σ_xk)
    return sum(l, dims=1)
end

Flux.@functor MixtureDensityNetwork

gaussian_kernel(y, μ, σ) = @. exp(- log(sqrt(2π)) - log(σ) - 0.5((y - μ) / σ)^2)

logloss(x) = - mean(log.(x))

function compute_loss(model::MixtureDensityNetwork, x, y::Matrix{<:AbstractFloat})
    p_k_x = model.p_k_x(x)
    σ_xk = model.σ_xk(x)
    μ_xk = model.μ_xk(x)
    return - mean(logsumexp(exponent(y, μ_xk, σ_xk, p_k_x), dims=1))
end

function gumbel_sample(x)
    z = rand(Gumbel(), size(x))
    return argmax(log.(x) + z, dims=1)
end

function sample_from(model::MixtureDensityNetwork, X, labels=nothing)
    k_x = gumbel_sample(model.p_k_x(X))
    σ_xk = model.σ_xk(X)
    μ_xk = model.μ_xk(X)
    return rand(Normal(), length(k_x)) .* vec(σ_xk[k_x]) .+ vec(μ_xk[k_x])
end

function TMLE.expected_value(model::MixtureDensityNetwork, X, labels)
    k_x = model.p_k_x(X)
    μ_xk = model.μ_xk(X)
    return sum(k_x .* μ_xk, dims=1)
end

### Neural Network Estimator

early_stopping_message(patience) =  string("Validation Loss stopped decreasing for ", patience, " consecutive epochs. Early Stopping.")

function X_y(dataset, Xcols, ycol)
    outcome_dataset = dropmissing(dataset, Symbol.([Xcols..., ycol]))
    X = outcome_dataset[:, collect(Xcols)]
    y = outcome_dataset[:, ycol]
    return X, y
end

function train_val_dataloaders(estimator, X, y, train_samples, val_samples)
    X = transpose_table(estimator, X)
    y = transpose_target(y, estimator.labels)
    X_train = X[:, train_samples]
    X_val = X[:, val_samples]
    y_train = y[:, train_samples]
    y_val = y[:, val_samples]

    train_loader = Flux.DataLoader((X_train, y_train), batchsize=estimator.batchsize)
    val_loader = Flux.DataLoader((X_val, y_val), batchsize=estimator.batchsize)
    return train_loader, val_loader
end

function training_loop!(estimator, train_loader, val_loader; verbosity=1)
    model = estimator.model
    n_train, n_val = length(train_loader.data[2]), length(val_loader.data[2])
    # Training Loop
    best_validation_loss = Inf
    best_model = Flux.state(model)
    local epoch_validation_loss = Inf
    es = Flux.early_stopping(() -> epoch_validation_loss, estimator.patience; init_score = Inf)
    opt_state = Flux.setup(estimator.optimiser, model)
    early_stopped = false
    for epoch in 1:estimator.max_epochs
        # Training
        epoch_training_loss = 0.
        for (xbatch, ybatch) in train_loader
            training_loss, grads = Flux.withgradient(model) do m
                compute_loss(m, xbatch, ybatch)
            end
            epoch_training_loss += training_loss*length(ybatch)
            Flux.update!(opt_state, model, grads[1])
        end
        # Check for NaNs
        if isnan(epoch_training_loss)
            throw(ErrorException("NaNs were encountered during training."))
        end
        # Compute Validation Loss
        epoch_training_loss /= n_train
        epoch_validation_loss = sum(length(ybatch)*compute_loss(model, xbatch, ybatch) for (xbatch, ybatch) in val_loader) / n_val
        # Update Best Model
        if epoch_validation_loss < best_validation_loss
            best_validation_loss = epoch_validation_loss
            best_model = Flux.state(model)
        end
        # Log Losses
        if verbosity > 0
            @info string("Epoch: ", epoch, ", Training Loss: ",  epoch_training_loss, ", Validation Loss: ", epoch_validation_loss)
        end
        # Check for early stopping
        if epoch_validation_loss == 0 || es()
            early_stopped = true
            break
        end
    end
    early_stopped == false && @warn string("NeuralNetworkEstimator early stopping not triggered (patience:", estimator.patience, "): increase `max_epochs`.")
    Flux.loadmodel!(estimator.model, best_model)
    return estimator, (validation_loss=best_validation_loss,)
end

function train!(estimator, X, y; verbosity=1)    
    # Dataloaders
    train_samples, val_samples = stratified_holdout_train_val_samples(X, y; resampling=estimator.resampling)
    train_loader, val_loader = train_val_dataloaders(estimator, X, y, train_samples, val_samples)

    # Training Loop
    estimator, info = training_loop!(estimator, train_loader, val_loader; verbosity=verbosity)
    return estimator, info
end

mutable struct NeuralNetworkEstimator
    model
    encoder::MLJBase.Machine
    optimiser::Flux.Optimise.AbstractOptimiser
    resampling::MLJBase.ResamplingStrategy
    batchsize::Int
    patience::Int
    max_epochs::Int
    labels::Union{Nothing, Vector}
    function NeuralNetworkEstimator(X, y; 
        hidden_sizes=(20,), 
        K=3, 
        optimiser = Adam(),
        resampling = Holdout(),
        batchsize = 64,
        patience = 5,
        max_epochs = 10,)
        
        encoder = machine(continuous_encoder(), X)
        fit!(encoder, verbosity=0)
        input_size = get_input_size(X)
        labels = nothing
        if y isa CategoricalVector
            labels = levels(y)
            output_size = length(labels)
            hidden_sizes = tuple(hidden_sizes..., output_size)
            model = CategoricalMLP(
                input_size=input_size, 
                hidden_sizes=hidden_sizes,
            )
        else
            model = MixtureDensityNetwork(
                input_size=input_size, 
                hidden_sizes=hidden_sizes,
                K=K
            )
        end

        return new(model, encoder, optimiser, resampling, batchsize, patience, max_epochs, labels)
    end
end

sample_from(estimator::NeuralNetworkEstimator, X::DataFrame) = 
    sample_from(estimator.model, transpose_table(estimator, X), estimator.labels)

function evaluation_metrics(estimator::NeuralNetworkEstimator, X, y)
    return (logloss = compute_loss(estimator.model, transpose_table(estimator, X), transpose_target(y, estimator.labels)), )
end

TMLE.expected_value(estimator::NeuralNetworkEstimator, X) = vec(TMLE.expected_value(estimator.model, transpose_table(estimator, X), estimator.labels))

"""
Procedure:

1. The data is split into train/validation
2. An initial `NeuralNetworkEstimator` is fitted on the train/validation sets using the smallest architecture
3. While validation loss decreases (specified by `sieve_patience`):
    1. Build a larger model specified by the sequence: `hidden_sizes_candidates`
    2. Fit the new `NeuralNetworkEstimator` the train/validation sets
4. Keep the best `NeuralNetworkEstimator`
"""
mutable struct SieveNeuralNetworkEstimator
    neural_net_estimator::NeuralNetworkEstimator
    hidden_sizes_candidates::Vector{Vector}
    sieve_patience::Int
    function SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=[[20,]],
        sieve_patience=2,
        K=3, 
        optimiser = Adam(),
        resampling = Holdout(),
        batchsize = 64,
        patience = 5,
        max_epochs = 10,)
        
        neural_net_estimator = NeuralNetworkEstimator(X, y; 
            hidden_sizes=first(hidden_sizes_candidates),
            K=K, 
            optimiser = optimiser,
            resampling = resampling,
            batchsize = batchsize,
            patience = patience,
            max_epochs = max_epochs,
        )

        return new(neural_net_estimator, [collect(x) for x in hidden_sizes_candidates], sieve_patience)
    end
end

infer_input_size(model::Chain) = size(model.layers[1][1].weight, 2)

infer_input_size(model::MixtureDensityNetwork) = size(model.p_k_x.layers[1][1].weight, 2)

infer_K(model::MixtureDensityNetwork) = size(model.p_k_x[end-1].weight, 1)

function newmodel(model::Chain, hidden_sizes)
    output_size = size(model[end].weight, 1)
    return CategoricalMLP(
        input_size=infer_input_size(model), 
        hidden_sizes=tuple(hidden_sizes..., output_size),
    )
end

function newmodel(model::MixtureDensityNetwork, hidden_sizes) 
    MixtureDensityNetwork(
        input_size=infer_input_size(model), 
        hidden_sizes=hidden_sizes,
        K=infer_K(model)
    )
end

function train!(estimator::SieveNeuralNetworkEstimator, X, y; verbosity=1)
    neural_net_estimator = estimator.neural_net_estimator
    # Dataloaders
    train_samples, val_samples = stratified_holdout_train_val_samples(X, y; resampling=neural_net_estimator.resampling)
    train_loader, val_loader = train_val_dataloaders(neural_net_estimator, X, y, train_samples, val_samples)

    # Initial fit
    neural_net_estimator, info = training_loop!(estimator.neural_net_estimator, train_loader, val_loader; verbosity=verbosity-1)
    best_validation_loss = info.validation_loss
    best_model = neural_net_estimator.model
    verbosity > 0 && @info(string("1-th Neural Network's (hidden-sizes: ", estimator.hidden_sizes_candidates[1], ") validation loss: ", info.validation_loss))

    # Progressively increase network size
    sieve_patience_count = 0
    for (step, hidden_sizes) in enumerate(estimator.hidden_sizes_candidates[2:end])
        # Create new model and fit
        neural_net_estimator.model = newmodel(best_model, hidden_sizes)
        neural_net_estimator, info = training_loop!(estimator.neural_net_estimator, train_loader, val_loader; verbosity=verbosity-1)
        verbosity > 0 && @info(string(step + 1, "-th Neural Network's (hidden-sizes: ", hidden_sizes, ") validation loss: ", info.validation_loss))
        # Update best_model / best_validation_loss / sieve_patience_count
        if info.validation_loss < best_validation_loss
            best_model = neural_net_estimator.model
            best_validation_loss = info.validation_loss
            sieve_patience_count = 0
        else
            sieve_patience_count += 1
        end
        # Stop if patience has been reached
        if sieve_patience_count == estimator.sieve_patience
            break
        end
    end
    sieve_patience_count != estimator.sieve_patience && @warn string("SieveNeuralNetworkEstimator early stopping not triggered (patience:", estimator.sieve_patience, "): increase `hidden_sizes_candidates`.")
    neural_net_estimator.model = best_model
    return estimator, (validation_loss=best_validation_loss,)
end

sample_from(estimator::SieveNeuralNetworkEstimator, X::DataFrame) = 
    sample_from(estimator.neural_net_estimator, X)

evaluation_metrics(estimator::SieveNeuralNetworkEstimator, X, y) = evaluation_metrics(estimator.neural_net_estimator, X, y)

TMLE.expected_value(estimator::SieveNeuralNetworkEstimator, X) = TMLE.expected_value(estimator.neural_net_estimator, X)
