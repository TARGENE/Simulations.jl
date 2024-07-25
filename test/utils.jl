module TestUtils

using Simulations
using Simulations: get_input_size, transpose_table,
    transpose_target, getlabels, train_validation_split, 
    confounders_and_covariates_set
using Test
using CategoricalArrays
using DataFrames
using MLJBase
using TMLE
using DataFrames
using Arrow
using TargeneCore

TARGENCORE_PKGDIR = pkgdir(TargeneCore)

TARGENCORE_TESTDIR = joinpath(TARGENCORE_PKGDIR, "test")

TESTDIR = joinpath(pkgdir(Simulations), "test")

include(joinpath(TESTDIR, "testutils.jl"))

@testset "Test compute_statistics" begin
    dataset = DataFrame(Arrow.Table(joinpath(TESTDIR, "assets", "dataset.arrow")))
    estimands = linear_interaction_dataset_ATEs().estimands
    statistics = Simulations.compute_statistics(dataset, estimands)
    # Continuous outcome, one treatment
    @test only(keys(statistics[1])) == :T₁
    @test names(statistics[1][:T₁]) == ["T₁", "proprow", "nrow"]
    # Count outcome, one treatment
    @test only(keys(statistics[2])) == :T₁
    @test names(statistics[2][:T₁]) == ["T₁", "proprow", "nrow"]
    # Binary outcome, two treatments
    for key ∈ (:Ybin, :T₁, :T₂, (:T₁, :T₂) ,(:Ybin, :T₁, :T₂))
        stats = statistics[3][key]
        @test stats isa DataFrame
        @test hasproperty(stats, "proprow")
        @test hasproperty(stats, "nrow")
    end
end

@testset "Test estimands variables accessors" begin
    Ψcont, Ψcount, composedΨ = linear_interaction_dataset_ATEs().estimands

    @test confounders_and_covariates_set(Ψcont) == Set([:W, :C])

    @test confounders_and_covariates_set(composedΨ) == Set([:W, :C])
    
end

@testset "Test misc" begin
    # Test get_input_size
    ## The categorical variables counts for 2
    X = DataFrame(
        x1 = [1,2, 3], 
        x2 = categorical([1,2, 3]),
        x3 = categorical([1, 2, 3], ordered=true)
    )
    @test get_input_size(X.x1) == 1
    @test get_input_size(X.x2) == 2
    @test get_input_size(X.x3) == 1
    @test get_input_size(X) == 4
    # Test getlabels
    ## Only Categorical Vectors return labels otherwise nothing
    @test getlabels(categorical(["AC", "CC", "CC"])) == ["AC", "CC"]
    @test getlabels([1, 2]) === nothing
    # transpose_table
    X = (
        A = [1, 2, 3],
        B = [4, 5, 6]
    )
    Xt = transpose_table(X)
    @test Xt == [
        1.0 2.0 3.0
        4.0 5.0 6.0
    ]
    @test Xt isa Matrix{Float32}
    @test transpose_target([1, 2, 3], nothing) == [1.0 2.0 3.0]
    y = categorical([1, 2, 1, 2])
    @test transpose_target(y, levels(y)) == [
        1 0 1 0
        0 1 0 1
    ]
    # Test get_bgen_chromosome
    bgen_files = [
        "ukb_53116_chr11.bgen",   
        "ukb_53116_chr12.bgen",
        "ukb_53116_chr12.bgen.bgi",              
        "ukb_53116_chr12.sample",
        "ukb_53116_chr2.bgen",
        "ukb_53116_chr2.bgen.bgi",
        "ukb_53116_chr2.sample"
    ]
    @test Simulations.get_bgen_chromosome(bgen_files, 2) == "ukb_53116_chr2.bgen"
    @test Simulations.get_bgen_chromosome(bgen_files, 12) == "ukb_53116_chr12.bgen"
    @test Simulations.get_bgen_chromosome(bgen_files, 11) == "ukb_53116_chr11.bgen"

    # Test read_bgen_chromosome(bgen_prefix, chr)
    bgen_prefix = joinpath(TARGENCORE_TESTDIR, "data", "ukbb", "imputed", "ukbb")
    b = Simulations.read_bgen_chromosome(bgen_prefix, 12)
    @test basename(b.idx.path) == "ukbb_chr12.bgen.bgi"
    @test !isempty(b.samples)
    @test occursin("ukbb_chr12.bgen", b.io.name)

    # Test keep_only_imputed
    associations = DataFrame(SNP=["RSID_101", "RSID_unkwown", "RSID_198"])
    ## When no bgen prefix is given simply return associations
    @test Simulations.keep_only_imputed(associations, nothing, 12) === associations
    ## Otherwise filter
    @test Simulations.keep_only_imputed(associations, bgen_prefix, 12) == DataFrame(SNP=["RSID_101", "RSID_198"])
end

@testset "Test train_validation_split" begin
    X, y = MLJBase.make_blobs()
    X_train, y_train, X_val, y_val = train_validation_split(X, y)
    @test size(y_train, 1) == 90
    @test size(y_val, 1) == 10
    @test X_train isa NamedTuple
    @test X_val isa NamedTuple
end

@testset "Test sample_from" begin
    origin_dataset = DataFrame(
        A = ["AC", "CC", missing, "AA", "AA", "CC", "AC"],
        B = ["AA", "AC", "CC", "AA", "AA", "AC", "AC"],
        C = ["AA", "AC", "CC", "CC", "AA", "AC", "AC"],
        D = 1:7
    )
    # Dropping missing results in CC not present in the dataset
    error_msg = string(
        "Filtering of missing values resulted in a too extreme dataset. In particular: Missing levels for variable: B.", 
        "\n Consider lowering or setting the `call_threshold` to `nothing`."
    )
    @test_throws ErrorException(error_msg) sample_from(origin_dataset, [:A, :B]; variables_to_check=[:A, :B], n=2, min_occurences=0, verbosity=0)
    # if min_occurences = 10, A won't have enough occurences and will raise first
    error_msg = string(
        "Filtering of missing values resulted in a too extreme dataset. In particular: Not enough occurences for variable: A.", 
        "\n Consider lowering or setting the `call_threshold` to `nothing`."
    )
    @test_throws ErrorException(error_msg) sample_from(origin_dataset, [:A, :B]; variables_to_check=[:A, :B], n=2, min_occurences=10, verbosity=0)
    # This will work
    variables = [:A, :C, :D]
    sampled_dataset = sample_from(origin_dataset, variables; n=4, variables_to_check=[:A, :C], min_occurences=0, verbosity=0)
    all_rows = collect(eachrow(origin_dataset[!, variables]))
    for row in eachrow(sampled_dataset)
        @test row ∈ all_rows
    end
    @test length(unique(sampled_dataset.A)) == 3
    @test length(unique(sampled_dataset.C)) == 3
end

@testset "Test coerce_parents_and_outcome!" begin
    testdataset() = DataFrame(
        A = [0, 1, 0, missing, 0, 1, 1, 0],
        B = [0, 1, 2, missing, 1, 2, 0, 1],
        C = ["AC", "CC", missing, "AA", "AA", "CC", "AC", "AC"],
        D = [0., 1.1, 2.3, missing, 3.2, 0.1, -3.4, -2.2]
    )
    # Binary outcome
    dataset = testdataset()
    Simulations.coerce_parents_and_outcome!(dataset, [:B, :C, :D]; outcome=:A)
    @test dataset.A isa CategoricalVector
    @test dataset.B isa CategoricalVector
    @test dataset.C isa CategoricalVector
    @test eltype(dataset.D) == Union{Missing, Float64}
    # Count outcome: treated as float
    dataset = testdataset()
    Simulations.coerce_parents_and_outcome!(dataset, [:A, :C, :D]; outcome=:B)
    @test dataset.A isa CategoricalVector
    @test eltype(dataset.B) == Union{Missing, Float64}
    @test dataset.C isa CategoricalVector
    @test eltype(dataset.D) == Union{Missing, Float64}
    # String outcome
    dataset = testdataset()
    Simulations.coerce_parents_and_outcome!(dataset, [:B, :A, :D]; outcome=:C)
    @test dataset.A isa CategoricalVector
    @test dataset.B isa CategoricalVector
    @test dataset.C isa CategoricalVector
    @test eltype(dataset.D) == Union{Missing, Float64}
    # Continuous outcome
    dataset = testdataset()
    Simulations.coerce_parents_and_outcome!(dataset, [:B, :C, :A]; outcome=:D)
    @test dataset.A isa CategoricalVector
    @test dataset.B isa CategoricalVector
    @test dataset.C isa CategoricalVector
    @test eltype(dataset.D) == Union{Missing, Float64}
end

end

true