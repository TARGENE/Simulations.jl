function test_density_estimators(X, y; batchsize=16)
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=[(20,), (20, 20)], 
        batchsize=batchsize
    )
    glm = GLMEstimator(X, y)
    return (snne=snne, glm=glm)
end

function study_density_estimators(X, y)
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=[(5,), (10,), (20,), (40,), (60,), (80,), (100,), (120,), (140,)], 
        max_epochs=10_000,
        optimiser=Adam(1e-4),
        resampling=StratifiedCV(nfolds=train_ratio),
        sieve_patience=5,
        batchsize=64,
        patience=5
    )
    glm = GLMEstimator(X, y)
    return (snne=snne, glm=glm)
end

function get_density_estimators(mode, X, y)
    density_estimators = if mode == "test"
        test_density_estimators(X, y)
    else
        study_density_estimators(X, y)
    end
    return density_estimators
end

function read_density_variables(file)
    d = JSON.parsefile(file)
    return Symbol(d["outcome"]), Symbol.(d["parents"])
end

serializable!(estimators::AbstractVector) = [serializable!(estimator) for estimator in estimators]

function density_estimation(
    dataset_file,
    density_file;
    mode="study",
    output=string("density_estimate.hdf5"),
    train_ratio=10,
    verbosity=1
    )
    outcome, parents = read_density_variables(density_file)
    dataset = TargetedEstimation.instantiate_dataset(dataset_file)
    coerce_parents_and_outcome!(dataset, parents, outcome=outcome)

    X, y = X_y(dataset, parents, outcome)
    density_estimators = get_density_estimators(mode, X, y)
    X_train, y_train, X_test, y_test = train_validation_split(X, y; train_ratio=train_ratio)
    metrics = []
    for estimator in density_estimators
        train!(estimator, X_train, y_train, verbosity=verbosity-1)
        train_loss = evaluation_metrics(estimator, X_train, y_train).logloss
        test_loss = evaluation_metrics(estimator, X_test, y_test).logloss
        push!(metrics, (train_loss=train_loss, test_loss=test_loss))
    end
    # Retrain Sieve Neural Network
    snne = get_density_estimators(mode, X, y).snne
    train!(snne, X, y, verbosity=verbosity-1)
    # Save
    if output !== nothing
        jldopen(output, "w") do io
            io["outcome"] = outcome
            io["parents"] = parents
            io["estimators"] = Simulations.serializable!(density_estimators)
            io["metrics"] = metrics
            io["sieve-neural-net"] = Simulations.serializable!(snne)
        end
    end
    return 0
end
