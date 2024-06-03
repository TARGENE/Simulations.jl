module TestUtils

using Simulations
using Simulations: get_input_size, transpose_table,
    transpose_target, getlabels, train_validation_split, 
    get_outcome, confounders_and_covariates_set, get_treatments,
    get_confounders_assert_equal, get_covariates_assert_equal
using Test
using CategoricalArrays
using DataFrames
using MLJBase
using TMLE
using DataFrames
using Arrow

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
    @test get_outcome(Ψcont) == :Ycont
    @test get_treatments(Ψcont) == (:T₁,)

    @test confounders_and_covariates_set(composedΨ) == Set([:W, :C])
    @test get_outcome(composedΨ) == :Ybin
    @test get_treatments(composedΨ) == (:T₁, :T₂)

    @test get_confounders_assert_equal([Ψcont, Ψcount, composedΨ]) == (:W,)
    @test get_covariates_assert_equal([Ψcont, Ψcount, composedΨ]) == (:C,)

    # Estimands with incompatible confounders/covariates
    Ψincompatible = CM(
        outcome="toto", 
        treatment_values=(T1=1, T2=1),
        treatment_confounders=(T1=[:W], T2=[:newW]),
        outcome_extra_covariates=(:newC,)
    )
    @test get_covariates_assert_equal(Ψincompatible) == (:newC,)
    @test_throws AssertionError get_confounders_assert_equal(Ψincompatible)
    
    Ψincompatible = ComposedEstimand(
        TMLE.joint_estimand,
        (
            ATE(
                outcome=:Ybin,
                treatment_values = (T₁ = (case=1, control=0), T₂ = (case=1, control=0)),
                treatment_confounders = (:W1,),
                outcome_extra_covariates = (:C1,)
            ),
            ATE(
                outcome=:Ybin,
                treatment_values = (T₁ = (case=0, control=1), T₂ = (case=0, control=1)),
                treatment_confounders = (:W2,),
                outcome_extra_covariates = (:C2,)
            )
        )
    )
    @test_throws AssertionError get_confounders_assert_equal(Ψincompatible)
    @test_throws AssertionError get_covariates_assert_equal(Ψincompatible)

    @test_throws AssertionError get_confounders_assert_equal([Ψincompatible.args[1], Ψcont])
    @test_throws AssertionError get_covariates_assert_equal([Ψincompatible.args[1], Ψcont])
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
    @test_throws ErrorException(error_msg) sample_from(origin_dataset, [:A, :B]; n=2, min_occurences=0, verbosity=0)
    # if min_occurences = 10, A won't have enough occurences and will raise first
    error_msg = string(
        "Filtering of missing values resulted in a too extreme dataset. In particular: Not enough occurences for variable: A.", 
        "\n Consider lowering or setting the `call_threshold` to `nothing`."
    )
    @test_throws ErrorException(error_msg) sample_from(origin_dataset, [:A, :B]; n=2, min_occurences=10, verbosity=0)
    # This will work
    variables = [:A, :C, :D]
    sampled_dataset = sample_from(origin_dataset, variables; n=4, min_occurences=0, verbosity=0)
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