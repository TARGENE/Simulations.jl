mutable struct GLMEstimator
    model
    machine
    function GLMEstimator(X, y; lambda=0.)
        if y isa CategoricalVector
            model = continuous_encoder() |> MLJLinearModels.LogisticClassifier(lambda=lambda)
        else
            model = continuous_encoder() |> MLJGLMInterface.LinearRegressor()
        end
        return new(model)
    end
end

function serializable!(estimator::GLMEstimator)
    estimator.machine = serializable(estimator.machine)
    return estimator
end

function MLJBase.restore!(estimator::GLMEstimator)
    restore!(estimator.machine)
    return estimator
end

function train!(estimator::GLMEstimator, X, y; verbosity=1)
    mach = machine(estimator.model, X, y, cache=false)
    fit!(mach, verbosity=verbosity)
    estimator.machine = mach
    return estimator
end

"""
For each row in X, samples a new y
"""
function sample_from(estimator::GLMEstimator, X)
    ŷ = MLJBase.predict(estimator.machine, X)
    return rand.(ŷ)
end

function evaluation_metrics(estimator::GLMEstimator, X, y)
    ŷ = MLJBase.predict(estimator.machine, X)
    logloss = -mean(logpdf.(ŷ, y))
    return (logloss = logloss,)
end

TMLE.expected_value(estimator::GLMEstimator, X) = 
    TMLE.expected_value(MLJBase.predict(estimator.machine, X))