get_sampler(::Nothing, estimands) = NullSampler(estimands)

get_sampler(prefix::AbstractString, estimands) =
    DensityEstimateSampler(prefix, estimands)

function estimate_from_simulated_data(
    origin_dataset, 
    estimands_config, 
    estimators_config;
    sample_size=nothing,
    max_sampling_attempts=1000,
    min_occurences=10,
    sampler_config=nothing,
    nrepeats=10,
    out="output.arrow",
    verbosity=1,
    rng_seed=0, 
    chunksize=100
    )
    rng = Random.default_rng()
    Random.seed!(rng, rng_seed)
    origin_dataset = TargetedEstimation.instantiate_dataset(origin_dataset)
    sample_size = sample_size !== nothing ? sample_size : nrow(origin_dataset)
    estimands = TargetedEstimation.instantiate_estimands(estimands_config, origin_dataset)
    estimators_spec = TargetedEstimation.instantiate_estimators(estimators_config, estimands)
    sampler = get_sampler(sampler_config, estimands)
    results = []
    statistics = []
    for repeat_id in 1:nrepeats
        verbosity > 0 && @info(string("Estimation for bootstrap: ", repeat_id))
        sampled_dataset = sample_from(sampler, origin_dataset; 
            n=sample_size,
            max_attempts=max_sampling_attempts,
            min_occurences=min_occurences,
            verbosity=verbosity-1
        )
        push!(statistics, compute_statistics(sampled_dataset, estimands))
        runner = Runner(
            sampled_dataset;
            estimands_config=estimands_config, 
            estimators_spec=estimators_spec, 
            verbosity=verbosity-1, 
            chunksize=chunksize,
            cache_strategy="release-unusable",
            sort_estimands=false
        )
        bootstrap_results = runner(1:size(runner.estimands, 1))
        append!(results, [merge(r, (REPEAT_ID=repeat_id,)) for r in bootstrap_results])
    end

    results_df = DataFrame(results)
    results_df.RNG_SEED .= rng_seed

    jldsave(out, 
        results=results_df, 
        statistics_by_repeat_id=statistics, 
        sample_size=sample_size,
        estimators=keys(estimators_spec)
    )
end
