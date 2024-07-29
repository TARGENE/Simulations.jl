module TestDensityEstimateSampler

using Test
using Simulations
using CategoricalArrays
using DataFrames
using Random
using Arrow
using TMLE
using MLJLinearModels
using Distributions
using LogExpFunctions

TESTDIR = joinpath(pkgdir(Simulations), "test")
@testset "Test DensityEstimateSampler" begin
    density_dir = mktempdir()
    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow") 
    # Learn T₁
    density_file = joinpath(TESTDIR, "assets", "conditional_density_T1.json")
    density_estimation(
        dataset_file,
        density_file;
        mode="test",
        output=joinpath(density_dir, "density_T1.hdf5"),
        train_ratio=10,
        verbosity=0
    )
    # Learn T₂
    density_file = joinpath(TESTDIR, "assets", "conditional_density_T2.json")
    density_estimation(
        dataset_file,
        density_file;
        mode="test",
        output=joinpath(density_dir, "density_T2.hdf5"),
        train_ratio=10,
        verbosity=0
    )
    # Learn Ybin
    density_file = joinpath(TESTDIR, "assets", "conditional_density_Ybin.json")
    density_estimation(
        dataset_file,
        density_file;
        mode="test",
        output=joinpath(density_dir, "density_Ybin.hdf5"),
        train_ratio=10,
        verbosity=0
    )
    # Learn Ycont
    density_file = joinpath(TESTDIR, "assets", "conditional_density_Ycont.json")
    density_estimation(
        dataset_file,
        density_file;
        mode="test",
        output=joinpath(density_dir, "density_Ycont.hdf5"),
        train_ratio=10,
        verbosity=0
    )
    # Scenario 1:
    # only 1 estimand, not all parents of Ybin are in the estimand's variables.
    # This is to make sure variants that are used to generated the 
    # data but not required afterwards for estimation are discarded 
    # in the sampled dataset
    estimands = [ATE(
        outcome=:Ybin,
        treatment_values=(T₁=(case=1, control=0),),
        treatment_confounders=(:W,),
        )]
    prefix = joinpath(density_dir, "density")
    sampler = DensityEstimateSampler(prefix, estimands)
    @test sampler.roots == [:C, :W]
    @test sampler.treatments_densities == Dict(
        :T₁ => ([:W], joinpath(density_dir, "density_T1.hdf5"))
    )
    @test sampler.outcomes_densities == Dict(
        :Ybin => ([:C, :T₁, :W], joinpath(density_dir, "density_Ybin.hdf5"))
    )
    @test sampler.variables_required_for_estimation == [:T₁, :W, :Ybin]
    
    # Sample
    origin_dataset = DataFrame(Arrow.Table(dataset_file))
    sampled_dataset = sample_from(sampler, origin_dataset, n=50, verbosity=0)
    ## Check parents are given by the empirical, only W here is necessary for estimation
    parents = [:W]
    all_origin_rows = collect(eachrow(origin_dataset[!, parents]))
    for row in eachrow(sampled_dataset[!, parents])
        @test row ∈ all_origin_rows
    end
    @test names(sampled_dataset) == string.(sampler.variables_required_for_estimation)
    @test size(sampled_dataset, 1) == 50
    @test sampled_dataset.T₁ isa CategoricalVector
    @test sampled_dataset.Ybin isa CategoricalVector
    # True effects
    Ψ₀ = get_true_effects(sampler, estimands, origin_dataset; n=10_000)
    @test Ψ₀[estimands[1]] isa Float64
    ## Failing sampling
    msg = string(
        "Could not sample ŷ which wasn't too extreme after: ", 10, 
        " attempts. Possible solutions: increase `sample_size`, change your simulation estimands of increase `max_attempts`."
    )
    @test_throws ErrorException(msg) sample_from(sampler, origin_dataset, n=50, min_occurences=50, max_attempts=10, verbosity=0)

    # Scenario 2:
    # Two estimands
    estimands = [ATE(
        outcome=:Ybin,
        treatment_values=(T₁=(case=1, control=0),),
        treatment_confounders=(:W,),
        ),
        factorialEstimand(
            IATE,
            (T₁=[0, 1], T₂=[0, 1]),
            :Ycont,
            confounders=(:W,),
            outcome_extra_covariates=(:C,)
        )
    ]
    sampler = DensityEstimateSampler(prefix, estimands)
    @test sampler.treatments_densities == Dict(
        :T₁ => ([:W], joinpath(density_dir, "density_T1.hdf5")),
        :T₂ => ([:W], joinpath(density_dir, "density_T2.hdf5"))
    )
    @test sampler.roots == [:C, :W]
    @test sampler.variables_required_for_estimation == [:C, :T₁, :T₂, :W, :Ybin, :Ycont]
    @test sampler.outcomes_densities == Dict(
        :Ybin => ([:C, :T₁, :W], joinpath(density_dir, "density_Ybin.hdf5")),
        :Ycont => ([:W, :T₁, :T₂, :C], joinpath(density_dir, "density_Ycont.hdf5"))
    )
    # Sample
    origin_dataset = DataFrame(Arrow.Table(dataset_file))
    sampled_dataset = sample_from(sampler, origin_dataset, n=50, verbosity=0)
    ## Check parents are given by the empirical
    parents = [:C, :W]
    all_origin_rows = collect(eachrow(origin_dataset[!, parents]))
    for row in eachrow(sampled_dataset[!, parents])
        @test row ∈ all_origin_rows
    end
    @test names(sampled_dataset) == string.(sampler.variables_required_for_estimation)
    @test size(sampled_dataset, 1) == 50
    @test sampled_dataset.Ybin isa CategoricalVector
    @test sampled_dataset.T₁ isa CategoricalVector
    @test sampled_dataset.T₂ isa CategoricalVector
    @test sampled_dataset.Ycont isa Vector{Float64}
    # True effects
    Ψ₀ = get_true_effects(sampler, estimands, origin_dataset; n=10_000)
    @test Ψ₀[estimands[1]] isa Float64
    @test Ψ₀[estimands[2]] isa Vector{Float64}
    @test length(Ψ₀[estimands[2]]) == length(estimands[2].args)
end

@testset "Test counterfactual_aggregate" begin
    # Make dataset
    n = 10_000
    W = Float64.(rand(Bernoulli(0.3), n))
    T = Float64.(rand(Uniform(), n) .< logistic.(2W .+ 1))
    Y = 0.3.*T .+ W .+ rand(Normal(), n)
    dataset = DataFrame(W = W, T = T, Y = Y)
    # Save outcome density and estimate
    cde = TMLE.MLConditionalDistributionEstimator(LinearRegressor())(
        TMLE.ConditionalDistribution(:Y, [:T, :W]),
        dataset;
        verbosity=0
    )
    # Check true effect is 0.6
    Ψ = CM(
        outcome = :Y,
        treatment_values = (T = 1,),
        treatment_confounders = [:W]
    )
    dataset.T = categorical(dataset.T)
    X = TMLE.selectcols(dataset, cde.estimand.parents)
    ctf_agg = Simulations.counterfactual_aggregate(Ψ, cde, X)   
    @test 0.5 < mean(ctf_agg) < 0.7
end

end

true
