module TestModelSelection

using Test
using Simulations
using Random
using Distributions
using DataFrames
using JLD2
using Flux
using JSON
using CSV
using Serialization
using TMLE

TESTDIR = joinpath(pkgdir(Simulations), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test misc" begin
    density_file = joinpath(TESTDIR, "assets", "conditional_density_x_y.json")
    outcome, features = Simulations.read_density_variables(density_file)
    @test outcome == :x
    @test features == [:y]
end

@testset "Test density_estimation: sinusoidal problem" begin
    # On this dataset, the GLM has no chance to perform best
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    dataset = sinusoidal_dataset(;n_samples=1000)

    outputdir = mktempdir()
    output = joinpath(outputdir, "density_estimate.hdf5")
    datasetfile = joinpath(outputdir, "dataset.csv")

    CSV.write(datasetfile, dataset)

    density_file = joinpath(TESTDIR, "assets", "conditional_density_x_y.json")
    estimators_file = joinpath(TESTDIR, "assets", "density_estimators.jl")
    copy!(ARGS, [
        "density-estimation",
        datasetfile,
        density_file,
        "--mode=test",
        string("--output=", output),
        string("--train-ratio=10"),
        string("--verbosity=0")
    ])
    Simulations.julia_main()

    jldopen(output) do io
        @test io["sieve-neural-net"] isa SieveNeuralNetworkEstimator
        @test io["outcome"] == :x
        @test io["parents"] == [:y]
        @test length(io["estimators"]) == 2
        metrics = io["metrics"]
        @test length(metrics) == 2
    end
end

@testset "Test density_estimation: with categorical variables" begin
    outputdir = mktempdir()
    output = joinpath(outputdir, "density_estimate.hdf5")
    datasetfile = joinpath(TESTDIR, "assets", "dataset.arrow")
    density_file = joinpath(TESTDIR, "assets", "conditional_density_Ybin.json")
    estimators_file = joinpath(TESTDIR, "assets", "density_estimators.jl")
    copy!(ARGS, [
        "density-estimation",
        datasetfile,
        density_file,
        "--mode=test",
        string("--output=", output),
        string("--train-ratio=10"),
        string("--verbosity=0")
    ])
    Simulations.julia_main()

    jldopen(output) do io
        @test io["outcome"] == :Ybin
        @test io["parents"] == [:C, :T₁, :W]
        metrics = io["metrics"]
        @test length(metrics) == 2
        @test length(io["estimators"]) == 2
        @test io["sieve-neural-net"] isa SieveNeuralNetworkEstimator
    end
end



end

true