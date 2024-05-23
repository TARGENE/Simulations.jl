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
    @test sort(sampler.all_parents_set) == [:C, :T₁, :W]
    @test sort(sampler.variables_required_for_estimation) == [:T₁, :W, :Ybin]
    @test sampler.density_mapping == Dict(:Ybin => ([:C, :T₁, :W], joinpath(density_dir, "density_Ybin.hdf5")))
    # Sample
    origin_dataset = DataFrame(Arrow.Table(dataset_file))
    sampled_dataset = sample_from(sampler, origin_dataset, n=50)
    ## Check parents are given by the empirical
    parents = [:W, :T₁]
    all_origin_rows = collect(eachrow(origin_dataset[!, parents]))
    for row in eachrow(sampled_dataset[!, parents])
        @test row ∈ all_origin_rows
    end
    @test names(sampled_dataset) == string.(sampler.variables_required_for_estimation)
    @test size(sampled_dataset, 1) == 50
    @test sampled_dataset.Ybin isa CategoricalVector

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
    @test sort(sampler.all_parents_set) == [:C, :T₁, :T₂, :W]
    @test sort(sampler.variables_required_for_estimation) == [:C, :T₁, :T₂, :W, :Ybin, :Ycont]
    @test sampler.density_mapping == Dict(
        :Ybin => ([:C, :T₁, :W], joinpath(density_dir, "density_Ybin.hdf5")),
        :Ycont => ([:W, :T₁, :T₂, :C], joinpath(density_dir, "density_Ycont.hdf5"))
        )
    # Sample
    origin_dataset = DataFrame(Arrow.Table(dataset_file))
    sampled_dataset = sample_from(sampler, origin_dataset, n=50)
    ## All parents are retained
    all_origin_rows = collect(eachrow(origin_dataset[!, sampler.all_parents_set]))
    for row in eachrow(sampled_dataset[!, sampler.all_parents_set])
        @test row ∈ all_origin_rows
    end
    @test names(sampled_dataset) == string.(sampler.variables_required_for_estimation)
    @test size(sampled_dataset, 1) == 50
    @test sampled_dataset.Ybin isa CategoricalVector
end

end

true
