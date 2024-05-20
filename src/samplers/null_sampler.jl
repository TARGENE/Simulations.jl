"""
The Permutation-Null-Sampler keeps the marginal distributions of each variable in the original dataset
intact while disrupting the causal relationships between them. This is done by:
    1. Sampling from (W, C)
    2. Permuting each T
    3. Permuting Y
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

function sample_from(sampler::NullSampler, origin_dataset; n=100)
    nrows = nrow(origin_dataset)
    sampled_dataset = sample_from(origin_dataset, collect(sampler.confounders_and_covariates); n=n)
    for variable in sampler.other_variables
        sample_mask = rand(1:nrows, n)
        sampled_dataset[!, variable] = origin_dataset[sample_mask, variable]
    end
    return sampled_dataset
end

theoretical_true_effect(Ψ, sampler::NullSampler) = 0.
theoretical_true_effect(Ψ::ComposedEstimand, sampler::NullSampler) = [0. for _ in Ψ.args]

"""
For this generating process, Y is independent of both T and W.

This effect computation:
    - Discards the dependence on W.
    - Uses the "dependence" on T. 
    
Not sure if this is reasonnable.

Mathematically, this is saying:

E[Y|do(T=t)] = E[Y|T=t] (= E[Y], in principle but not used)
"""
function confounders_independent_effect(dataset, Ψ)
    indicator_fns = TMLE.indicator_fns(Ψ)
    effect = 0.
    for (key, group) ∈ pairs(groupby(dataset, TMLE.treatments(Ψ)))
        if haskey(indicator_fns, values(key))
            effect += mean(group[!, Ψ.outcome]) * indicator_fns[values(key)]
        end
    end
    return effect
end

function empirical_true_effect(Ψ, sampler::NullSampler, origin_dataset; n=500_000)
    sampled_dataset = sample_from(sampler, origin_dataset; n=n)
    return confounders_independent_effect(sampled_dataset, Ψ)
end

function empirical_true_effect(Ψ::ComposedEstimand, sampler::NullSampler, origin_dataset; n=500_000)
    sampled_dataset = sample_from(sampler, origin_dataset; n=n)
    effect = zeros(length(Ψ.args))
    for (index, arg) in enumerate(Ψ.args)
        effect[index] = confounders_independent_effect(sampled_dataset, arg)
    end
    return effect
end

true_effect(Ψ, sampler::NullSampler, origin_dataset; n=500_000) = theoretical_true_effect(Ψ, sampler)