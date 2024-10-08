module TestNullSampler

using Test
using Simulations
using Random
using DataFrames
using CategoricalArrays
using Statistics
using Distributions
using TMLE
using LogExpFunctions

TESTDIR = joinpath(pkgdir(Simulations), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test NullSampler" begin
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    origin_dataset = linear_interaction_dataset()
    estimands = linear_interaction_dataset_ATEs().estimands
    sampler = NullSampler(estimands)
    @test sampler.confounders_and_covariates == Set([:C, :W])
    @test sampler.other_variables == Set([:Ycont, :Ybin, :Ycount, :T₁, :T₂])

    sampled_dataset = sample_from(sampler, origin_dataset, n=1000)
    @test names(sampled_dataset) == ["W", "C", "Ycont", "Ybin", "T₁", "Ycount", "T₂"]
    @test size(sampled_dataset, 1) == 1000
    # Structure between (W, C) is preserved
    origin_WC = [row for row in eachrow(origin_dataset[!, ["W", "C"]])]
    @test all(row in origin_WC for row in eachrow(sampled_dataset[!, ["W", "C"]]))
    # Basic stats are somewhat preserved
    @test mean(sampled_dataset.Ycont) ≈ mean(origin_dataset.Ycont) atol=0.1
    for T in (:T₁, :T₂)
        p_sampled_T = sort(combine(groupby(sampled_dataset, T), proprow), :proprow)
        p_origin_T = sort(combine(groupby(origin_dataset, T), proprow), :proprow)
        @test p_sampled_T[!, T] == p_origin_T[!, T]
    end
    # True Effect is zero
    true_effects = get_true_effects(sampler, estimands, origin_dataset)
    @test true_effects[estimands[2]] == true_effects[estimands[1]] == 0
    @test true_effects[estimands[3]] == [0, 0]
    # Raise if min_occurences not statisfied
    msg = string(
        "Could not sample variable Ybin because it either did not have enough occurences or some levels were missing after 1 attempts.\nConsider increasing the sample size or changing your estimands."
    )
    @test_throws ErrorException(msg) sample_from(sampler, origin_dataset, n=10, min_occurences = 10, max_attempts=1, verbosity=0)

    # Raises if non compatible estimand is provided
    push!(
        estimands, 
        ATE(
            outcome=:Ycont,
            treatment_values = (T₁ = (case=1, control=0),),
            treatment_confounders = (:W,),
            outcome_extra_covariates = ()
    ))
    @test_throws AssertionError("All estimands should share the same confounders and covariates.") NullSampler(estimands)
end

end

true