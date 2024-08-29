"""
The NullSampler keeps the marginal distributions of each variable in the original dataset
intact while disrupting the causal relationships between them. This is done by:
    1. Sampling from (W, C)
    2. Sampling from each T independently
    3. Sampling from each Y independently
"""
struct NullSampler
    confounders_and_covariates
    other_variables
    function NullSampler(estimands)
        # Check confounders and covariates are the same for all estimands
        confounders_and_covariates = confounders_and_covariates_set(first(estimands))
        other_variables = Set{Symbol}([])
        for Ψ in estimands
            @assert confounders_and_covariates_set(Ψ) == confounders_and_covariates "All estimands should share the same confounders and covariates."
            push!(other_variables, get_outcome(Ψ))
            push!(other_variables, get_treatments(Ψ)...)
        end
        return new(confounders_and_covariates, other_variables)
    end
end

function NullSampler(outcome, treatments; 
    confounders=("PC1", "PC2", "PC3", "PC4", "PC5", "PC6"), 
    outcome_extra_covariates=("Age-Assessment", "Genetic-Sex")
    )
    variables = variables_from_args(outcome, treatments, confounders, outcome_extra_covariates)
    return NullSampler(variables)
end

"""
    sample_from(sampler::NullSampler, origin_dataset; 
        n=100,
        min_occurences=10,
        max_attempts=1000,
        verbosity=1
    )
    The procedure tries to:
        1. Sample jointly from (W, C): 
            - The levels of sampled factor variables should match the levels in the original data.
        2. Sample independently for each T and Y: 
            - The levels of sampled factor variables should match the levels in the original data.
            - The lowest populated sampled level of each factor variable should have more than `min_occurences` samples.
    each for a maximum of `max_attempts`.
"""
function sample_from(sampler::NullSampler, origin_dataset; 
    n=100,
    min_occurences=10,
    max_attempts=1000,
    verbosity=1
    )
    sampled_dataset = sample_from(origin_dataset, collect(sampler.confounders_and_covariates); 
        n=n,
        max_attempts=max_attempts,
        verbosity=verbosity
    )
    # Independently sample the rest of variables
    for variable in sampler.other_variables
        sampled_dataset[!, variable] = sample_from(origin_dataset, variable; 
            n=n,
            min_occurences=min_occurences,
            max_attempts=max_attempts,
            )
    end
    return sampled_dataset
end

get_true_effect(sampler::NullSampler, Ψ) = 0

get_true_effect(sampler::NullSampler, Ψ::JointEstimand) = [get_true_effect(sampler, Ψᵢ) for Ψᵢ in Ψ.args]

get_true_effects(sampler::NullSampler, estimands::AbstractVector, origin_dataset; kwargs...) = 
    Dict(Ψ => get_true_effect(sampler, Ψ) for Ψ ∈ estimands)