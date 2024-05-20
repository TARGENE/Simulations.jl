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
    save_integration_test_configuration(estimands_filename)
    # Creating the inputs
    parsed_args = Dict(
        "out-prefix" => joinpath(inputdir, "final"), 
        "batch-size" => 2,
        "positivity-constraint" => 0.,
        "verbosity" => 0,

        "%COMMAND%" => "from-param-file",

        "from-param-file" => Dict{String, Any}(
            "paramfile" => estimands_filename,
            "traits" => joinpath(TARGENCORE_TESTDIR, "data", "traits_1.csv"),
            "pcs" => joinpath(TARGENCORE_TESTDIR, "data", "pcs.csv"),
            "call-threshold" => 0.8, 
            "bgen-prefix" => joinpath(TARGENCORE_TESTDIR, "data", "ukbb", "imputed" ,"ukbb"), 
            ), 
    )
    TargeneCore.tl_inputs_from_param_files(parsed_args)
    # Estimation runs
    outdir = mktempdir()
    nrepeats = 2
    dataset_file = joinpath(inputdir, "final.data.arrow")
    ## Run 1: vanilla-glmnet / ATEs / sample-size=100
    workdir1 = mktempdir()
    out1 = joinpath(outdir, "permutation_results_1.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        joinpath(inputdir, "final.estimands_1.jls"),
        joinpath(TESTDIR, "assets", "estimators", "vanilla-glmnet.jl"),
        "--sample-size=100",
        string("--n-repeats=", nrepeats),
        "--rng=0",
        string("--out=", out1),
        string("--workdir=", workdir1)
    ])
    Simulations.julia_main()
    jldopen(out1) do io
        @test io["sample_size"] == 100
        @test io["estimators"] == (:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)
        @test length(io["statistics_by_repeat_id"]) == 2
        @test names(io["results"]) == ["wTMLE_GLMNET", "TMLE_GLMNET", "OSE_GLMNET", "REPEAT_ID", "RNG_SEED"]
        @test size(io["results"]) == (4, 5)
    end
    ## Run 2: vanilla-xgboost / ATEs / sample-size=200
    workdir2 = mktempdir()
    out2 = joinpath(outdir, "permutation_results_2.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        joinpath(inputdir, "final.estimands_2.jls"),
        joinpath(TESTDIR, "assets", "estimators", "vanilla-xgboost.jl"),
        "--sample-size=200",
        string("--n-repeats=", nrepeats),
        "--rng=1",
        string("--out=", out2),
        string("--workdir=", workdir2)
    ])
    Simulations.julia_main()
    jldopen(out2) do io
        @test io["sample_size"] == 200
        @test io["estimators"] == (:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)
        @test length(io["statistics_by_repeat_id"]) == 2
        @test names(io["results"]) == ["wTMLE_XGBOOST", "TMLE_XGBOOST", "OSE_XGBOOST", "REPEAT_ID", "RNG_SEED"]
        @test size(io["results"]) == (4, 5)
    end
    ## Run 3: vanilla-xgboost / ATEs / sample-size=200 / 4 repeats
    workdir3 = mktempdir()
    out3 = joinpath(outdir, "permutation_results_3.hdf5")
    copy!(ARGS, [
        "estimation",
        dataset_file,
        joinpath(inputdir, "final.estimands_3.jls"),
        joinpath(TESTDIR, "assets", "estimators", "vanilla-xgboost.jl"),
        "--sample-size=200",
        string("--n-repeats=", nrepeats),
        "--rng=2",
        string("--out=", out3),
        string("--workdir=", workdir3)
    ])
    Simulations.julia_main()
    jldopen(out3) do io
        @test io["sample_size"] == 200
        @test io["estimators"] == (:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)
        @test length(io["statistics_by_repeat_id"]) == 2
        @test names(io["results"]) == ["wTMLE_XGBOOST", "TMLE_XGBOOST", "OSE_XGBOOST", "REPEAT_ID", "RNG_SEED"]
        @test size(io["results"]) == (2, 5)
    end
    ## Aggregate the 3 runs
    results_file = joinpath(outdir, "permutation_results.hdf5")
    copy!(ARGS, [
        "aggregate",
        joinpath(outdir, "permutation_results"),
        results_file,
    ])
    Simulations.julia_main()
    jldopen(results_file) do io
        results = io["results"]
        run_1 = results[(:wTMLE_GLMNET, :TMLE_GLMNET, :OSE_GLMNET)][100]
        @test names(run_1) == ["wTMLE_GLMNET", "TMLE_GLMNET", "OSE_GLMNET", "REPEAT_ID", "RNG_SEED"]
        @test size(run_1) == (4, 5)
        @test run_1.REPEAT_ID == [1, 1, 2, 2]
        @test run_1.RNG_SEED ==[0, 0, 0, 0]
        run_2_3 = results[(:wTMLE_XGBOOST, :TMLE_XGBOOST, :OSE_XGBOOST)][200]
        @test names(run_2_3) == ["wTMLE_XGBOOST", "TMLE_XGBOOST", "OSE_XGBOOST", "REPEAT_ID", "RNG_SEED"]
        @test size(run_2_3) == (6, 5)
        @test run_2_3.REPEAT_ID == [1, 1, 2, 2, 1, 2]
        @test run_2_3.RNG_SEED ==[1, 1, 1, 1, 2, 2]
    end
end

end

true