
sieve_neural_net_density_estimator(file::AbstractString) = jldopen(io -> restore!(io["sieve-neural-net"]), file)

struct DensityEstimateSampler
    sources::Vector
    treatment_density_mapping::Dict
    outcome_density_mapping::Dict
end

function DensityEstimateSampler(prefix, estimands)
    # Create density to file map (There could be more files than actually required)
    density_mapping = Dict()
    for f ∈ files_matching_prefix(prefix)
        jldopen(f) do io
            density_mapping[(Symbol(io["outcome"]) => Tuple(Symbol.(io["parents"])))] = f
        end
    end
    # Create required density to file map
    required_densities = Dict()
    all_parents = Set([])
    all_outcomes = Set([])
    for Ψ in estimands
        for η in TMLE.nuisance_functions_iterator(Ψ)
            required_densities[η.outcome => η.parents] = density_mapping[η.outcome => η.parents]
            union!(all_parents, η.parents)
            push!(all_outcomes, η.outcome)
        end
    end
    # A source is never an outcome
    sources = [x for x in all_parents if x ∉ all_outcomes]
    # A treatment node is both an outcome and a parent
    treatment_density_mapping = Dict(cd => f for (cd, f) in required_densities if (cd[1] ∈ all_parents && cd[1] ∈ all_outcomes))
    # An outcome node is never a parent
    outcome_density_mapping = Dict(cd => f for (cd, f) in required_densities if cd[1] ∉ all_parents)
    return DensityEstimateSampler(sources, treatment_density_mapping, outcome_density_mapping)
end

function sample_from(sampler::DensityEstimateSampler, origin_dataset; n=100)

    sampled_dataset = sample_from(origin_dataset, sampler.sources; n=n)

    for density_mapping in (sampler.treatment_density_mapping, sampler.outcome_density_mapping)
        for ((outcome, parents), file) in density_mapping
            conditional_density_estimate = Simulations.sieve_neural_net_density_estimator(file)
            sampled_dataset[!, outcome] = sample_from(
                conditional_density_estimate, 
                sampled_dataset[!, collect(parents)]            
            ) 
        end
    end

    return sampled_dataset
end

function counterfactual_aggregate(Ψ, Q, X)
    Ttemplate = TMLE.selectcols(X, TMLE.treatments(Ψ))
    n = nrow(Ttemplate)
    ctf_agg = zeros(n)
    # Loop over Treatment settings
    for (vals, sign) in TMLE.indicator_fns(Ψ)
        # Counterfactual dataset for a given treatment setting
        T_ct = TMLE.counterfactualTreatment(vals, Ttemplate)
        X_ct = merge(X, T_ct)
        # Counterfactual mean
        ctf_agg .+= sign .* TMLE.expected_value(Q, X_ct)
    end
    return ctf_agg
end

function monte_carlo_effect(Ψ, sampler, dataset)
    outcome_mean = TMLE.outcome_mean(Ψ)
    Q = Simulations.sieve_neural_net_density_estimator(sampler.outcome_density_mapping[outcome_mean.outcome => outcome_mean.parents])
    X = TMLE.selectcols(dataset, outcome_mean.parents)
    labels = Simulations.getlabels(dataset[!, outcome_mean.outcome])
    ctf_agg = counterfactual_aggregate(Ψ, Q, X)
    return mean(ctf_agg)
end

function true_effect(Ψ, sampler::DensityEstimateSampler, origin_dataset; n=500_000)
    sampled_dataset = sample_from(sampler, origin_dataset; n=n)
    return monte_carlo_effect(Ψ, sampler, sampled_dataset)
end

function true_effect(Ψ::ComposedEstimand, sampler::DensityEstimateSampler, origin_dataset;n=500_000)
    sampled_dataset = sample_from(sampler, origin_dataset; n=n)
    effect = zeros(length(Ψ.args))
    for (index, arg) in enumerate(Ψ.args)
        effect[index] = monte_carlo_effect(arg, sampler, sampled_dataset)
    end
    return effect
end