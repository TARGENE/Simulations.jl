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
        factorialEstimand(ATE, (rs502771=["TT", "TC", "CC"],), "ALL";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(ATE, (rs184270108=["CC", "CT", "TT"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(AIE, (rs502771=["TT", "TC", "CC"], rs184270108=["CC", "CT", "TT"],), "sarcoidosis";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
        factorialEstimand(AIE, (rs11868112=["CC", "CT", "TT"], rs6456121=["CC", "CT", "TT"], rs356219=["GG", "GA", "AA"]), "G20 Parkinson's disease";
            confounders=[:PC1, :PC2, :PC3], 
            outcome_extra_covariates=["Genetic-Sex", "Age-Assessment"]
        ),
    ]
end

@testset "Test check_only_one_set_of_confounders_per_treatment" begin
    estimands = Any[
        JointEstimand(
            CM(outcome=:Y, treatment_values=(T=1,), treatment_confounders=(:W1, :W2)),
            CM(outcome=:Y, treatment_values=(T=0,), treatment_confounders=(:W1, :W2))
        )
    ]
    Simulations.check_only_one_set_of_confounders_per_treatment(estimands)
    push!(
        estimands,
        ATE(outcome=:Y, treatment_values=(T=(case=0,control=1),), treatment_confounders=(:W1,))
    )
    @test_throws ArgumentError Simulations.check_only_one_set_of_confounders_per_treatment(estimands)
end

@testset "Test initialize_trait_to_variants" begin
    estimands, _ = estimands_and_traits_to_variants_matching_bgen()
    variables = (genetic_variants=Set([:RSID_2]), outcomes=Set([:BINARY_1, :BINARY_2, :CONTINUOUS_1]))
    trait_to_variants = Simulations.initialize_trait_to_variants(estimands, variables)
    @test trait_to_variants == Dict(
        "BINARY_2"     => Set(["RSID_2"]),
        "BINARY_1"     => Set(["RSID_2"]),
        "CONTINUOUS_1" => Set(["RSID_2"])
        )
    variables = (genetic_variants=Set([:RSID_2, :RSID_198]), outcomes=Set([:BINARY_1, :BINARY_2]))
    trait_to_variants = Simulations.initialize_trait_to_variants(estimands, variables)
    @test trait_to_variants == Dict(
        "BINARY_2"     => Set(["RSID_2"]),
        "BINARY_1"     => Set(["RSID_2", "RSID_198"]),
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

@testset "Test realistic_simulation_inputs: sample_gene_atlas_hits=false" begin
    tmpdir = mktempdir()
    estimands_prefix = joinpath(tmpdir, "estimands.jls")
    bgen_prefix = joinpath(TARGENCORE_TESTDIR, "data", "ukbb", "imputed" ,"ukbb")
    traits_file = joinpath(TARGENCORE_TESTDIR, "data", "traits_1.csv")
    pcs_file = joinpath(TARGENCORE_TESTDIR, "data", "pcs.csv")
    output_prefix = joinpath(tmpdir, "realistic_inputs")
    estimands, _ = estimands_and_traits_to_variants_matching_bgen()
    serialize(estimands_prefix, TMLE.Configuration(estimands=estimands))
    
    copy!(ARGS, [
        "realistic-simulation-inputs",
        estimands_prefix,
        bgen_prefix,
        traits_file,
        pcs_file,
        "--sample-gene-atlas-hits=false",
        "--ga-download-dir=gene_atlas_data",
        "--remove-ga-data=true",
        string("--ga-trait-table=", joinpath(PKGDIR, "assets", "Traits_Table_GeneATLAS.csv")),
        "--maf-threshold=0.01",
        "--pvalue-threshold=1e-5",
        "--distance-threshold=1e6",
        "--max-variants=100",
        string("--output-prefix=", output_prefix),
        "--batchsize=10",
        "--positivity-constraint=0",
        "--call-threshold=0.9",
        "--verbosity=0",
        ]
    )
    Simulations.julia_main()
    # Check conditional densities
    conditional_densities = Set([JSON.parsefile(f) for f in TargeneCore.files_matching_prefix(string(output_prefix, ".conditional_density"))])
    @test conditional_densities == Set([
        # Treatment mechanisms
        Dict("parents" => ["PC2", "PC1"], "outcome" => "TREAT_1"),
        Dict("parents" => ["PC2", "PC1"], "outcome" => "RSID_2"),
        Dict("parents" => ["PC2", "PC1"], "outcome" => "RSID_198"),
        # Outcome mechanisms: inferred from ATE with ALL
        Dict("parents" => ["TREAT_1", "RSID_2", "22001", "PC2", "PC1"], "outcome" => "BINARY_2"),
        Dict("parents" => ["TREAT_1", "RSID_2", "22001", "PC2", "PC1"], "outcome" => "COV_1"),
        Dict("parents" => ["TREAT_1", "RSID_2", "22001", "PC2", "PC1"], "outcome" => "CONTINUOUS_2"),
        Dict("parents" => ["TREAT_1", "RSID_2", "22001", "PC2", "PC1"], "outcome" => "21003"),
        Dict("parents" => ["TREAT_1", "RSID_2", "22001", "PC2", "PC1"], "outcome" => "CONTINUOUS_1"),
        # Outcome mechanisms: from both JointEstimand and ATE with ALL, note the presence of TREAT_1 in the parents
        Dict("parents" => ["TREAT_1", "RSID_2", "22001", "PC2", "RSID_198", "PC1"], "outcome" => "BINARY_1")
    ])
    # Check dataset
    dataset = Arrow.Table(string(output_prefix, ".data.arrow")) |> DataFrame
    @test names(dataset) == ["SAMPLE_ID", "BINARY_1", "BINARY_2", "CONTINUOUS_1", "CONTINUOUS_2", "COV_1", "21003", "22001", "TREAT_1", "PC1", "PC2", "RSID_2", "RSID_198"]
    @test size(dataset) == (490, 13)
    # 1 estimand for the JointEstimand and 6 for the ATE
    output_estimands = deserialize(string(output_prefix, ".estimands_1.jls")).estimands
    @test length(output_estimands) == 7
    # Check estimands have been matched to the dataset: GA -> AG
    Ψ = output_estimands[findfirst(x->x isa JointEstimand, output_estimands)]
    @test Ψ.args[1].treatment_values[:RSID_198] == "AG"
    @test Ψ.args[2].treatment_values[:RSID_198] == "AG"
end

@testset "Test realistic_simulation_inputs: sample_gene_atlas_hits=true" begin
    # The function `realistic_simulation_inputs` is hard to test end to end when involving sampling
    # variants from the gene-atlas. Only the first parts of the function are tested here
    tmpdir = mktempdir()
    estimands = gene_atlas_estimands()
    variables = (
        genetic_variants = Set([:rs502771, :rs184270108, :rs11868112, :rs356219, :rs6456121]),
        outcomes= Set([:sarcoidosis, Symbol("G20 Parkinson's disease"), Symbol("J40-J47 Chronic lower respiratory diseases")])
    )
    ## All outcomes inherit rs502771 as a parent from the first estimand
    trait_to_variants = Simulations.initialize_trait_to_variants(estimands, variables)
    @test trait_to_variants == Dict(
        "sarcoidosis"              => Set(["rs502771", "rs184270108"]),
        "J40-J47 Chronic lower respiratory diseases" => Set(["rs502771"]),
        "G20 Parkinson's disease"  => Set(["rs502771", "rs11868112", "rs6456121", "rs356219"])
    )
    Simulations.update_trait_to_variants_from_ga!(
        trait_to_variants; 
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
    @test length(trait_to_variants["J40-J47 Chronic lower respiratory diseases"]) == 10
    @test issubset(("rs502771", ), trait_to_variants["J40-J47 Chronic lower respiratory diseases"])
    @test length(trait_to_variants["G20 Parkinson's disease"]) == 10
    @test issubset(("rs11868112", "rs6456121", "rs356219"), trait_to_variants["G20 Parkinson's disease"])
end

end

true