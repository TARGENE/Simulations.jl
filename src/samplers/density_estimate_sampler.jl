
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

function sample_from(sampler::DensityEstimateSampler, origin_dataset; n=100)

    sampled_dataset = sample_from(origin_dataset, sampler.all_parents_set; n=n)

    for (outcome, (parents, file)) in sampler.density_mapping
        coerce_parents_and_outcome!(sampled_dataset, parents; outcome=nothing)
        conditional_density_estimate = Simulations.sieve_neural_net_density_estimator(file)
        sampled_dataset[!, outcome] = sample_from(
            conditional_density_estimate, 
            sampled_dataset[!, parents]            
        ) 
    end
    return sampled_dataset[!, sampler.variables_required_for_estimation]
end
