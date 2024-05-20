module TestDensityEstimateSampler

using Test
using Simulations
using CategoricalArrays
using DataFrames
using Random
using Arrow
using TMLE

TESTDIR = joinpath(pkgdir(Simulations), "test")

@testset "Test DensityEstimateSampler" begin
    density_dir = mktempdir()
    dataset_file = joinpath(TESTDIR, "assets", "dataset.arrow")
    density_file = joinpath(TESTDIR, "assets", "conditional_density_Ybin.json")
    # Learn Ybin
    density_estimation(
        dataset_file,
        density_file;
        mode="test",
        output=joinpath(density_dir, "density_Ybin.hdf5"),
        train_ratio=10,
        verbosity=0
    )
    # Learn T1
    density_file = joinpath(TESTDIR, "assets", "conditional_density_T1.json")
    density_estimation(
        dataset_file,
        density_file;
        mode="test",
        output=joinpath(density_dir, "density_T1.hdf5"),
        train_ratio=10,
        verbosity=0
    )
    # Create the sampler
    estimands = [ATE(
        outcome=:Ybin,
        treatment_values=(T₁=(case=1, control=0),),
        treatment_confounders=(:W,),
        outcome_extra_covariates=(:C,)
        )]
    prefix = joinpath(density_dir, "density")
    sampler = DensityEstimateSampler(prefix, estimands)
    @test Set(sampler.sources) == Set([:W, :C])
    @test sampler.treatment_density_mapping == Dict((:T₁=>(:W,)) => joinpath(density_dir, "density_T1.hdf5"))
    @test sampler.outcome_density_mapping == Dict((:Ybin=>(:C, :T₁, :W)) => joinpath(density_dir, "density_Ybin.hdf5"))
    # Sample
    origin_dataset = DataFrame(Arrow.Table(dataset_file))
    sampled_dataset = sample_from(sampler, origin_dataset, n=100)

    @test names(sampled_dataset) == ["W", "C", "T₁", "Ybin"]
    @test size(sampled_dataset, 1) == 100

    # True effects
    Ψ = estimands[1]
    effect = true_effect(Ψ, sampler, origin_dataset; n=1000)
end

end

true
