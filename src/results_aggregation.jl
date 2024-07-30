
########################################################################
###                                BIAS                              ###
########################################################################

bias(Ψ̂, Ψ₀) = Ψ̂ .- Ψ₀

bias(Ψ̂::TMLE.Estimate, Ψ₀) = bias(TMLE.estimate(Ψ̂), Ψ₀)

mean_bias(estimates, Ψ₀) = mean(bias(TMLE.estimate(Ψ̂), Ψ₀) for Ψ̂ in estimates)

add_mean_bias_col!(results_df) = 
    results_df.MEAN_BIAS = [mean_bias(Ψ̂s, Ψ₀) for (Ψ̂s, Ψ₀) ∈ zip(results_df.ESTIMATES, results_df.TRUE_EFFECT)]

########################################################################
###                              VARIANCE                            ###
########################################################################

variance(Ψ̂) = Ψ̂.std^2 / Ψ̂.n

"""
From: https://johannesjakobmeyer.com/blog/005-multivariate-bias-variance-decomposition/
"""
variance(Ψ̂::TMLE.JointEstimate) = tr(Ψ̂.cov) / Ψ̂.n

mean_variance(Ψ̂s) = mean(variance(Ψ̂) for Ψ̂ ∈ Ψ̂s)

add_mean_variance_col!(results_df) = 
    results_df.MEAN_VARIANCE = [mean_variance(Ψ̂s) for Ψ̂s ∈ results_df.ESTIMATES]

########################################################################
###                             COVERAGE                             ###
########################################################################

function covers(Ψ̂, Ψ₀; alpha=0.05)
    pval = pvalue_or_nan(Ψ̂, Ψ₀)
    return pval === NaN ? NaN : pval > alpha
end

mean_coverage(estimates, Ψ₀) = mean(skipnan(covers(Ψ̂, Ψ₀) for Ψ̂ in estimates))

add_mean_coverage_col!(results_df) = 
    results_df.MEAN_COVERAGE = [mean_coverage(Ψ̂s, Ψ₀) for (Ψ̂s, Ψ₀) in zip(results_df.ESTIMATES, results_df.TRUE_EFFECT)]

########################################################################
###                              MISC                                ###
########################################################################

function add_n_failed!(results_df)
    non_failed_estimates = [filter(x -> !(x isa TargetedEstimation.FailedEstimate), estimates) for estimates in results_df.ESTIMATES]
    n_failed = [length(all) - length(non_failed) for (all, non_failed) ∈ zip(results_df.ESTIMATES, non_failed_estimates)]
    results_df.ESTIMATES = non_failed_estimates
    results_df.N_FAILED = n_failed
end

function add_outcome_col!(results_df)
    results_df.OUTCOME = [TargeneCore.get_outcome(Ψ) for Ψ ∈ results_df.ESTIMAND]
end

function add_true_effect_col!(results_df, density_estimates_prefix, dataset_file; kwargs...)
    origin_dataset = dataset_file !== nothing ? TargetedEstimation.instantiate_dataset(dataset_file) : nothing
    estimands = unique(results_df.ESTIMAND)
    sampler = get_sampler(density_estimates_prefix, estimands)
    true_effects = get_true_effects(sampler, estimands, origin_dataset; kwargs...)
    results_df.TRUE_EFFECT = [true_effects[Ψ] for Ψ in results_df.ESTIMAND]
end

########################################################################
###                               AGGREGATION                        ###
########################################################################

function update_results_dict!(results_dict, estimator, sample_size, Ψ̂)
    key = (estimator, Ψ̂.estimand, sample_size)
    key_results = get!(results_dict, key, [])
    push!(key_results, Ψ̂)
end

function save_aggregated_df_results(
    results_prefix,
    out,
    density_estimates_prefix, 
    dataset_file;
    n=500_000, 
    min_occurences=10,
    max_attempts=10,
    verbosity=0)
    # Aggregate results
    results_dict = Dict()
    for file in TargeneCore.files_matching_prefix(results_prefix)
        jldopen(file) do io
            estimators = io["estimators"]
            file_results = io["results"]
            sample_size = io["sample_size"]
            for estimator in estimators
                for Ψ̂ in file_results[!, estimator]
                    update_results_dict!(results_dict, estimator, sample_size, Ψ̂)
                end
            end
        end
    end
    # Convert results to DataFrame
    results_pairs = collect(results_dict)
    results_df = DataFrame(first.(results_pairs), [:ESTIMATOR, :ESTIMAND, :SAMPLE_SIZE])
    results_df.ESTIMATES = last.(results_pairs)
    # Add various summary columns
    add_n_failed!(results_df)
    add_outcome_col!(results_df)
    add_true_effect_col!(results_df, density_estimates_prefix, dataset_file;
        n=n, 
        min_occurences=min_occurences,
        max_attempts=max_attempts,
        verbosity=verbosity-1)
    add_mean_coverage_col!(results_df)
    add_mean_bias_col!(results_df)
    add_mean_variance_col!(results_df)

    jldsave(out, results=results_df)
end
