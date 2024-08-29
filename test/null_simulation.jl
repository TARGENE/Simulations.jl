module TestEstimation

using Test
using Simulations
using TMLE
using Random
using JLD2
using Distributions
using LogExpFunctions
using DataFrames
using JLD2
using TargeneCore

TARGENCORE_PKGDIR = pkgdir(TargeneCore)

TARGENCORE_TESTDIR = joinpath(TARGENCORE_PKGDIR, "test")

PKGDIR = pkgdir(Simulations)

TESTDIR = joinpath(PKGDIR, "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Integration Test Null Simulation" begin
    inputdir = mktempdir()
    estimands_filename = joinpath(inputdir, "estimands.yaml")
    save_configuration_macthing_bgen(estimands_filename)
    # Creating the inputs
    copy!(ARGS, [
        "estimation-inputs",
        estimands_filename,
        string("--traits-file=", joinpath(TARGENCORE_TESTDIR, "data", "traits_1.csv")),
        string("--pcs-file=", joinpath(TARGENCORE_TESTDIR, "data", "pcs.csv")),
        string("--genotypes-prefix=", joinpath(TARGENCORE_TESTDIR, "data", "ukbb", "imputed" ,"ukbb")),
        string("--outprefix=", joinpath(inputdir, "final")), 
        "--call-threshold=0.8",
        "--verbosity=0",
        "--positivity-constraint=0"
    ])
    TargeneCore.julia_main()
    # Estimation Grid
    outdir = mktempdir()
    dataset_file = joinpath(inputdir, "final.data.arrow")
    estimators = ["ose--glm", joinpath(TESTDIR, "assets", "estimators", "vanilla-glmnet.jl")]
    rngs = [1, 2]
    sample_sizes = [100, 200]
    n_bootstraps = 2
    (output_index, (estimator, rng, sample_size)) = first(enumerate(Iterators.product(estimators, rngs, sample_sizes)))
    for (output_index, (estimator, rng, sample_size)) ∈ enumerate(Iterators.product(estimators, rngs, sample_sizes))
        out = joinpath(outdir, string("null_estimation_results_", output_index, ".hdf5"))
        copy!(ARGS, [
            "estimation",
            dataset_file,
            joinpath(inputdir, "final.estimands_1.jls"),
            estimator,
            string("--sample-size=", sample_size),
            "--min-occurences=0",
            "--max-sampling-attempts=1000",
            string("--n-repeats=", n_bootstraps),
            string("--rng=", rng),
            string("--out=", out)
        ])
        Simulations.julia_main()
        jldopen(out) do io
            @test io["sample_size"] == sample_size
            @test length(io["statistics_by_repeat_id"]) == n_bootstraps
            @test io["results"] isa DataFrame
            @test nrow(io["results"]) == 7*2 # 7 estimands * 2 bootstraps samples
        end
    end
    ## Aggregate the results
    results_file = joinpath(outdir, "null_estimation_results.hdf5")
    copy!(ARGS, [
        "aggregate",
        joinpath(outdir, "null_estimation_results_"),
        results_file, 
    ])
    Simulations.julia_main()
    jldopen(results_file) do io
        results = io["results"]
        # 3 estimators, 2 sample sizes, 7 estimands = 42 rows
        @test nrow(results) == 42
        @test Set(results.ESTIMATOR) == Set([:OSE_GLMNET, :wTMLE_GLMNET, :OSE_GLM_GLM])
        @test Set(results.SAMPLE_SIZE) == Set([100, 200])
        @test all(length(x) == 4 - nf for (x, nf) in zip(results.ESTIMATES, results.N_FAILED)) # 2 bootstraps per run * 2 random seeds
        @test Set(names(results)) == Set([
            "ESTIMATOR",
            "ESTIMAND",
            "SAMPLE_SIZE",
            "ESTIMATES",
            "N_FAILED",
            "OUTCOME",
            "TRUE_EFFECT",
            "MEAN_COVERAGE",
            "MEAN_BIAS",
            "MEAN_VARIANCE"]
        )
    end
end

end

true