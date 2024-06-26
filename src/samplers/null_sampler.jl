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

function sample_from(sampler::NullSampler, origin_dataset; 
    n=100,
    min_occurences=10,
    max_attempts=1000,
    verbosity=1
    )
    sampled_dataset = sample_from(origin_dataset, collect(sampler.confounders_and_covariates); 
        n=n,
        min_occurences=min_occurences,
        max_attempts=max_attempts,
        verbosity=verbosity
    )
    # Independently sample the rest of variables
    variables_to_check = [var for var in sampler.other_variables if isfactor(origin_dataset[!, var])]
    for variable in sampler.other_variables
        sampled_variable_df = sample_from(origin_dataset, [variable]; 
            n=n,
            variables_to_check=variables_to_check,
            min_occurences=min_occurences,
            max_attempts=max_attempts,
            verbosity=verbosity
        )
        sampled_dataset[!, variable] = sampled_variable_df[!, variable]
    end
    return sampled_dataset
end