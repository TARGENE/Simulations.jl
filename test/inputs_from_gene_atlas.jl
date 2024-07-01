module TestInputsFromGeneAtlas

using TMLE
using Test
using Simulations
using Serialization
using JSON
using TargeneCore
using Arrow
using DataFrames

TARGENCORE_PKGDIR = pkgdir(TargeneCore)

TARGENCORE_TESTDIR = joinpath(TARGENCORE_PKGDIR, "test")

PKGDIR = pkgdir(Simulations)

TESTDIR = joinpath(PKGDIR, "test")

include(joinpath(TESTDIR, "testutils.jl"))

function gene_atlas_estimands()
    return [
        factorialEstimand(ATE, (rs502771=["TT", "TC", "CC"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(ATE, (rs184270108=["CC", "CT", "TT"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(IATE, (rs502771=["TT", "TC", "CC"], rs184270108=["CC", "CT", "TT"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(IATE, (rs11868112=["CC", "CT", "TT"], rs6456121=["CC", "CT", "TT"], rs356219=["GG", "GA", "AA"]), "G20 Parkinson's disease";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
    ]
end

function estimands_and_traits_to_variants_matching_bgen()
    estimands = [
        IATE(
            outcome = "BINARY_1",
            treatment_values = (RSID_2 = (case = "AA", control = "GG"), TREAT_1 = (case = 1, control = 0)),
            treatment_confounders = (RSID_2 = [], TREAT_1 = [])
        ),
        ATE(
            outcome = "BINARY_2",
            treatment_values = (RSID_2 = (case = "AA", control = "GG"),),
            treatment_confounders = (RSID_2 = [22001], ),
            outcome_extra_covariates = ["COV_1", 21003]
        ),
        CM(
            outcome = "CONTINUOUS_2",
            treatment_values = (RSID_2 = "AA", ),
            treatment_confounders = (RSID_2 = [22001],),
            outcome_extra_covariates = ["COV_1", 21003]
        ),
        ATE(
            outcome = "CONTINUOUS_2",
            treatment_values = (RSID_2 = (case = "AA", control = "GG"), RSID_198 = (case = "GA", control = "AA")),
            treatment_confounders = (RSID_2 = [], RSID_198 = []),
            outcome_extra_covariates = [22001]
        ),
        JointEstimand(
            CM(
                outcome = "BINARY_1",
                treatment_values = (RSID_2 = "GG", RSID_198 = "GA"),
                treatment_confounders = (RSID_2 = [], RSID_198 = []),
                outcome_extra_covariates = [22001]
            ),
            CM(
                outcome = "BINARY_1",
                treatment_values = (RSID_2 = "AA", RSID_198 = "GA"),
                treatment_confounders = (RSID_2 = [], RSID_198 = []),
                outcome_extra_covariates = [22001]
            )
        )
    ]
    traits_to_variants = Dict(
        "BINARY_1" => ["RSID_2", "RSID_198"],
        "CONTINUOUS_2" => ["RSID_2", "RSID_198"],
        "BINARY_2" => ["RSID_2"],
        )
    return estimands, traits_to_variants
end

@testset "Test check_only_one_set_of_confounders_per_treatment" begin
    estimands = Any[
        CM(outcome=:Y, treatment_values=(T=1,), treatment_confounders=(:W1, :W2)),
        CM(outcome=:Y, treatment_values=(T=0,), treatment_confounders=(:W1, :W2)),
    ]
    Simulations.check_only_one_set_of_confounders_per_treatment(estimands)
    push!(estimands, ATE(outcome=:Y, treatment_values=(T=(case=0,control=1),), treatment_confounders=(:W1,)))
    @test_throws ArgumentError Simulations.check_only_one_set_of_confounders_per_treatment(estimands)
end

@testset "Test get_trait_to_variants_from_estimands" begin
    estimands = linear_interaction_dataset_ATEs().estimands
    push!(estimands, ATE(outcome=:Ycont, treatment_values=(T₃=(case=1, control=0),), treatment_confounders=(:W,)))
    # Empty regex
    trait_to_variants = Simulations.get_trait_to_variants_from_estimands(estimands; regex=r"")
    @test trait_to_variants == Dict(
        "Ycont"  => Set(["T₁", "T₃"]),
        "Ybin"   => Set(["T₁", "T₂"]),
        "Ycount" => Set(["T₁"])
        )
    # T₂ regex
    trait_to_variants = Simulations.get_trait_to_variants_from_estimands(estimands; regex=r"T₂")
    @test trait_to_variants == Dict(
        "Ycont"  => Set(),
        "Ybin"   => Set(["T₂"]),
        "Ycount" => Set()
        )
    # T₁ regex
    trait_to_variants = Simulations.get_trait_to_variants_from_estimands(estimands; regex=r"T₁")
    @test trait_to_variants == Dict(
        "Ycont"  => Set(["T₁"]),
        "Ybin"   => Set(["T₁"]),
        "Ycount" => Set(["T₁"])
        )
end

@testset "Test get_trait_key_map" begin
    # "Vitamin D Level" and "Red-Hair" are not part of the geneATLAS
    traits = ["G35 Multiple sclerosis", "Vitamin D Level", "White blood cell (leukocyte) count", "sarcoidosis", "D86 Sarcoidosis", "G35 Multiple sclerosis", "K90-K93 Other diseases of the digestive system", "H00-H06 Disorders of eyelid, lacrimal system and orbit", "Trunk fat percentage"]
    @test_throws ArgumentError Simulations.get_trait_key_map(traits; trait_table_path=joinpath(PKGDIR, "assets", "Traits_Table_GeneATLAS.csv"))
    # Only valid traits
    traits = ["G35 Multiple sclerosis", "White blood cell (leukocyte) count", "sarcoidosis", "D86 Sarcoidosis", "G35 Multiple sclerosis", "K90-K93 Other diseases of the digestive system", "H00-H06 Disorders of eyelid, lacrimal system and orbit", "Trunk fat percentage"]
    trait_key_map = Simulations.get_trait_key_map(traits; trait_table_path=joinpath(PKGDIR, "assets", "Traits_Table_GeneATLAS.csv"))
    @test trait_key_map == Dict(
        "G35 Multiple sclerosis"                                 => "clinical_c_G35",
        "sarcoidosis"                                            => "selfReported_n_1414",
        "H00-H06 Disorders of eyelid, lacrimal system and orbit" => "clinical_c_Block_H00-H06",
        "Trunk fat percentage"                                   => "23127-0.0",
        "K90-K93 Other diseases of the digestive system"         => "clinical_c_Block_K90-K93",
        "D86 Sarcoidosis"                                        => "clinical_c_D86",
        "White blood cell (leukocyte) count"                     => "30000-0.0"
    )
end

@testset "Test simulation_inputs_from_gene_atlas" begin
    # The function `simulation_inputs_from_gene_atlas` is hard to test end to end due to
    # data limitations. It is split into 3 subfunctions that we here test sequentially but 
    # with different data.
    verbosity = 0
    # Here we use the real geneATLAS data
    tmpdir = mktempdir()
    estimands = gene_atlas_estimands()
    trait_to_variants = Simulations.get_trait_to_variants(
        estimands; 
        verbosity=0, 
        gene_atlas_dir=joinpath(tmpdir, "gene_atlas_data"),
        remove_ga_data=true,
        trait_table_path=joinpath(PKGDIR, "assets", "Traits_Table_GeneATLAS.csv"),
        maf_threshold=0.01,
        pvalue_threshold=1e-5,
        distance_threshold=1e6,
        max_variants=10,
        bgen_prefix=nothing
    )
    @test length(trait_to_variants["sarcoidosis"]) == 10
    @test issubset(("rs502771", "rs184270108"), trait_to_variants["sarcoidosis"])
    @test length(trait_to_variants["G20 Parkinson's disease"]) == 10
    @test issubset(("rs11868112", "rs6456121", "rs356219"), trait_to_variants["G20 Parkinson's disease"])

    # Dataset and validated estimands
    # We change the data to match what is in the BGEN file
    estimands, trait_to_variants = estimands_and_traits_to_variants_matching_bgen()
    bgen_prefix = joinpath(TARGENCORE_TESTDIR, "data", "ukbb", "imputed" ,"ukbb")
    traits_file = joinpath(TARGENCORE_TESTDIR, "data", "traits_1.csv")
    pcs_file = joinpath(TARGENCORE_TESTDIR, "data", "pcs.csv")
    positivity_constraint = 0
    call_threshold = 0.9
    dataset, validated_estimands = Simulations.get_dataset_and_validated_estimands(
        estimands,
        bgen_prefix,
        traits_file,
        pcs_file,
        trait_to_variants;
        call_threshold=call_threshold,
        positivity_constraint=positivity_constraint,
        verbosity=verbosity
    )
    @test names(dataset) == ["SAMPLE_ID", "BINARY_1", "BINARY_2", "CONTINUOUS_1", "CONTINUOUS_2", "COV_1",
        "21003", "22001", "TREAT_1", "PC1", "PC2", "RSID_2", "RSID_198"]
    @test size(dataset, 1) == 490
    @test length(estimands) == length(validated_estimands)
    # Check estimands have been matched to the dataset: GA -> AG
    Ψ = validated_estimands[findfirst(x->x isa JointEstimand, validated_estimands)]
    @test Ψ.args[1].treatment_values.RSID_198 == "AG"
    @test Ψ.args[2].treatment_values.RSID_198 == "AG"

    # Now the writing
    output_prefix = joinpath(tmpdir, "ga.input")
    batchsize = 2
    Simulations.write_ga_simulation_inputs(
        output_prefix,
        dataset,
        validated_estimands,
        trait_to_variants;
        batchsize=batchsize,
        verbosity=verbosity
    )
    # One dataset
    loaded_dataset = Arrow.Table(string(output_prefix, ".data.arrow")) |> DataFrame
    @test size(loaded_dataset) == (490, 13)
    # 5 estimands split in 3 files of size (2, 2, 1)
    loaded_estimands = reduce(
        vcat, 
        deserialize(string(output_prefix, ".estimands_$i.jls")).estimands for i in 1:3
    )
    @test loaded_estimands == validated_estimands
    # 3 densities for the 3 outcomes
    outcome_to_parents = Dict()
    for i in 1:3
        density = JSON.parsefile(string(output_prefix, ".conditional_density_$i.json"))
        outcome_to_parents[density["outcome"]] = sort(density["parents"])
    end
    @test outcome_to_parents == Dict(
        "BINARY_2"     => ["21003", "22001", "COV_1", "PC1", "PC2", "RSID_2"],
        "CONTINUOUS_2" => ["21003", "22001", "COV_1", "PC1", "PC2", "RSID_198", "RSID_2"],
        "BINARY_1"     => ["22001", "PC1", "PC2", "RSID_198", "RSID_2", "TREAT_1"]
    )
end

end

true