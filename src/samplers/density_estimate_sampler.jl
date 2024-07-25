
sieve_neural_net_density_estimator(file::AbstractString) = jldopen(io -> restore!(io["sieve-neural-net"]), file)

struct DensityEstimateSampler
    treatments_densities::Dict
    outcomes_densities::Dict
    roots::Vector{Symbol}
    variables_required_for_estimation::Vector{Symbol}
end

function DensityEstimateSampler(prefix, estimands)
    # Parse variables
    variables_required_for_estimation = Set{Symbol}()
    roots_set = Set{Symbol}()
    treatments_set = Set{Symbol}()
    outcomes_set = Set{Symbol}()
    for Ψ ∈ estimands
        # Update outcomes_set
        outcome = get_outcome(Ψ)
        push!(outcomes_set, outcome)
        # Update treatments_set
        treatments = get_treatments(Ψ)
        union!(treatments_set, treatments)
        # Update roots_set
        roots = union(
            get_outcome_extra_covariates(Ψ),
            get_all_confounders(Ψ)
        )
        union!(roots_set, roots)
        # Update variables_required_for_estimation
        Ψ_variables = union(
            Set([outcome]),
            treatments,
            roots,
        )
        union!(
            variables_required_for_estimation,
            Ψ_variables
        )
    end
    # Create density to file maps (There could be more files than required)
    treatments_densities = Dict()
    outcomes_densities = Dict()
    for f ∈ TargeneCore.files_matching_prefix(prefix)
        jldopen(f) do io
            outcome = Symbol(io["outcome"])
            if any(outcome ∈ set for set ∈ (outcomes_set, treatments_set))
                parents = Symbol.(io["parents"])
                if outcome ∈ outcomes_set
                    outcomes_densities[outcome] = (parents, f)
                elseif outcome ∈ treatments_set
                    treatments_densities[outcome] = (parents, f)
                end
                # We add to the roots all parent variables that are not treatments themselves (they have been modeled)
                union!(roots_set, setdiff(parents, treatments_set))
            end
        end
    end
    roots = sort(collect(roots_set))
    variables_required_for_estimation = sort(collect(variables_required_for_estimation))
    return DensityEstimateSampler(treatments_densities, outcomes_densities, roots, variables_required_for_estimation)
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
    # Sample Roots
    sampled_dataset = sample_from(origin_dataset, sampler.roots; 
        n=n, 
        min_occurences=min_occurences,
        variables_to_check=[],
        max_attempts=max_attempts,
        verbosity=verbosity
    )
    coerce_parents_and_outcome!(sampled_dataset, sampler.roots; outcome=nothing)
    # Sample Treatments
    for (treatment, (parents, file)) in sampler.treatments_densities
        conditional_density_estimate = Simulations.sieve_neural_net_density_estimator(file)
        sampled_dataset[!, treatment] = safe_sample_from(conditional_density_estimate, sampled_dataset, parents;
            min_occurences=min_occurences,
            max_attempts=max_attempts,
            verbosity = verbosity
        )
    end
    # Sample Outcomes
    for (outcome, (parents, file)) in sampler.outcomes_densities
        conditional_density_estimate = Simulations.sieve_neural_net_density_estimator(file)
        sampled_dataset[!, outcome] = Simulations.safe_sample_from(conditional_density_estimate, sampled_dataset, parents;
            min_occurences=min_occurences,
            max_attempts=max_attempts,
            verbosity = verbosity
        )
    end
    return sampled_dataset[!, sampler.variables_required_for_estimation]
end
