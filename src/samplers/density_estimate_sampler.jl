
sieve_neural_net_density_estimator(file::AbstractString) = jldopen(io -> restore!(io["sieve-neural-net"]), file)

struct DensityEstimateSampler
    density_mapping::Dict
    all_parents_set::Vector{Symbol}
    variables_required_for_estimation::Vector{Symbol}
end

get_outcomes_set(estimands) = Set(get_outcome(Ψ) for Ψ in estimands)

function DensityEstimateSampler(prefix, estimands)
    # Create density to file map (There could be more files than actually required)
    outcomes_set = get_outcomes_set(estimands)
    all_parents_set = Set{Symbol}()
    density_mapping = Dict()
    for f ∈ files_matching_prefix(prefix)
        jldopen(f) do io
            outcome = Symbol(io["outcome"])
            if outcome ∈ outcomes_set
                parents = Symbol.(io["parents"])
                density_mapping[outcome] = (parents, f)
                union!(all_parents_set, parents)
            end
        end
    end
    variables_required_for_estimation = Set{Symbol}()
    for Ψ ∈ estimands
        pre_outcome_variables = union(
            all_outcome_extra_covariates(Ψ),
            all_treatments(Ψ),
            all_confounders(Ψ),
        )
        union!(
            variables_required_for_estimation,
            pre_outcome_variables
        )
        push!(variables_required_for_estimation, get_outcome(Ψ))
    end

    return DensityEstimateSampler(density_mapping, collect(all_parents_set), collect(variables_required_for_estimation))
end

function safe_sample_from(conditional_density_estimate, sampled_dataset, parents;
    min_occurences=10,
    max_attempts=1000,
    verbosity = 1
    )
    for attempt in 1:max_attempts
        ŷ = sample_from(
            conditional_density_estimate,
            sampled_dataset[!, parents]
        )
        if ŷ isa CategoricalVector
            n_uniques = countmap(ŷ)
            if length(n_uniques) != length(levels(ŷ))
                verbosity > 0 && @info(string("Sampled ŷ after attempt ", attempt, " is missing some levels, retrying."))
                continue
            end
            if minimum(values(n_uniques)) < min_occurences
                verbosity > 0 && @info(string("Some levels in sampled ŷ after attempt ", attempt, " are not frequent enough, retrying."))
                continue
            end
        end
        return ŷ
    end
    msg = string(
        "Could not sample ŷ which wasn't too extreme after: ", max_attempts, 
        " attempts. Possible solutions: increase `sample_size`, change your simulation estimands of increase `max_attempts`."
    )
    throw(ErrorException(msg)) 
end

function sample_from(sampler::DensityEstimateSampler, origin_dataset; 
    n=100, 
    min_occurences=10,
    max_attempts=1000,
    verbosity = 1
    )
    sampled_dataset = sample_from(origin_dataset, sampler.all_parents_set; 
        n=n, 
        min_occurences=min_occurences,
        max_attempts=max_attempts,
        verbosity=verbosity
    )

    for (outcome, (parents, file)) in sampler.density_mapping
        coerce_parents_and_outcome!(sampled_dataset, parents; outcome=nothing)
        conditional_density_estimate = Simulations.sieve_neural_net_density_estimator(file)
        sampled_dataset[!, outcome] = safe_sample_from(conditional_density_estimate, sampled_dataset, parents;
            min_occurences=min_occurences,
            max_attempts=max_attempts,
            verbosity = verbosity
        )
    end
    return sampled_dataset[!, sampler.variables_required_for_estimation]
end
