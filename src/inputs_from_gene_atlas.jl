########################################################################
###       Creation of DensityEstimation Inputs From geneATLAS        ###
########################################################################

imputed_trait_chr_associations_filename(trait, chr) = string("imputed.allWhites.", trait, ".chr", chr, ".csv.gz")

genotyped_trait_chr_associations_filename(trait, chr) = string("genotyped.allWhites.", trait, ".chr", chr, ".csv.gz")

imputed_chr_filename(chr) = string("snps.imputed.chr", chr, ".csv.gz")

genotyped_chr_filename(chr) = string("snps.genotyped.chr", chr, ".csv.gz")

function download_gene_atlas_trait_info(trait; outdir="gene_atlas_data")
    isdir(outdir) || mkdir(outdir)
    # Download association results per chromosome
    for chr in 1:22
        imputed_filename = imputed_trait_chr_associations_filename(trait, chr) 
        imputed_url = string("http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/imputed/data.copy/", imputed_filename)
        download(imputed_url, joinpath(outdir, imputed_filename))

        genotyped_filename = genotyped_trait_chr_associations_filename(trait, chr) 
        genotyped_url = string("http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/genotyped/data/", genotyped_filename)
        download(genotyped_url, joinpath(outdir, genotyped_filename))
    end
end

function download_variants_info(outdir)
    variants_dir = joinpath(outdir, "variants_info")
    isdir(variants_dir) || mkdir(variants_dir)
    for chr in 1:22
        imputed_filename = imputed_chr_filename(chr)
        imputed_url = string("http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/snps/extended/", imputed_filename)
        download(imputed_url, joinpath(variants_dir, imputed_filename))

        genotyped_filename = genotyped_chr_filename(chr)
        genotyped_url = string("http://static.geneatlas.roslin.ed.ac.uk/gwas/allWhites/snps/extended/", genotyped_filename)
        download(genotyped_url, joinpath(variants_dir, genotyped_filename))
    end
end

function get_trait_to_variants_from_estimands(estimands; regex=r"^(rs[0-9]*|Affx)")
    trait_to_variants = Dict()
    for Ψ in estimands
        outcome = string(TargeneCore.get_outcome(Ψ))
        variants = filter(x -> occursin(regex, x), string.(TargeneCore.get_treatments(Ψ)))
        if haskey(trait_to_variants, outcome)
            union!(trait_to_variants[outcome], variants)
        else
            trait_to_variants[outcome] = Set(variants)
        end
    end
    return trait_to_variants
end

function load_associations(trait_dir, trait_key, chr)
    imputed_associations = DataFrames.select(CSV.read(joinpath(trait_dir, imputed_trait_chr_associations_filename(trait_key, chr)), 
        header=["SNP", "ALLELE", "ISCORE", "NBETA", "NSE", "PV"],
        skipto=2,
        DataFrame
    ), Not(:ISCORE))
    genotyped_associations = CSV.read(joinpath(trait_dir, genotyped_trait_chr_associations_filename(trait_key, chr)), 
        header=["SNP", "ALLELE", "NBETA", "NSE", "PV"],
        skipto=2,
        DataFrame
    )
    return vcat(imputed_associations, genotyped_associations)
end

function load_variants_info(gene_atlas_dir, chr)
    imputed_variants = DataFrames.select(CSV.read(joinpath(gene_atlas_dir, "variants_info", imputed_chr_filename(chr)) , DataFrame), Not(:iscore))
    genotyped_variants = CSV.read(joinpath(gene_atlas_dir, "variants_info", genotyped_chr_filename(chr)) , DataFrame)
    return vcat(imputed_variants, genotyped_variants)
end

function update_trait_to_variants_from_gene_atlas!(trait_to_variants, trait_key_map; 
    gene_atlas_dir="gene_atlas_data",
    remove_ga_data=true,
    maf_threshold=0.01,
    pvalue_threshold=1e-5,
    distance_threshold=1e6,
    max_variants=100,
    bgen_prefix=nothing
    )
    isdir(gene_atlas_dir) || mkdir(gene_atlas_dir)
    download_variants_info(gene_atlas_dir)

    for (trait, trait_key) in trait_key_map
        # Download association data
        trait_outdir = joinpath(gene_atlas_dir, trait_key)
        download_gene_atlas_trait_info(trait_key; outdir=trait_outdir)
        estimand_variants = trait_to_variants[trait]
        independent_chr_variants = Dict()
        for chr in 1:22
            # Load association data and SNP info
            associations = load_associations(trait_outdir, trait_key, chr)
            variants_info = load_variants_info(gene_atlas_dir, chr)
            associations = innerjoin(associations, variants_info, on="SNP")
            # Restrict variants to those present in the imputed file
            associations = keep_only_imputed(associations, bgen_prefix, chr)
            associations.NOTIN_ESTIMAND = [v ∉ estimand_variants for v in associations.SNP]
            # Only keep variants in estimands OR bi-allelic SNPs with "sufficient" MAF and p-value
            filter!(
                x -> !(x.NOTIN_ESTIMAND) || (x.PV < pvalue_threshold && x.MAF >= maf_threshold && (length(x.A1) == length(x.A2) == 1)), 
                associations
            )
            # Prioritizing SNPs in estimands and then by p-value
            sort!(associations, [:NOTIN_ESTIMAND, :PV])
            snp_to_pos = []
            for (snp, pos, notin_estimand) ∈ zip(associations.SNP, associations.Position, associations.NOTIN_ESTIMAND)
                # Always push SNPs in estimands
                if !notin_estimand
                    push!(snp_to_pos, snp => pos)
                else
                    # Always push the first SNP
                    if isempty(snp_to_pos)
                        push!(snp_to_pos, snp => pos)
                    else
                        # Only push if the SNP is at least `distance_threshold` away from the closest SNP
                        min_dist = min((abs(prev_pos - pos) for (prev_snp, prev_pos) in snp_to_pos)...)
                        if min_dist > distance_threshold
                            push!(snp_to_pos, snp => pos)
                        end
                    end
                end
            end

            # Update variant set from associated SNPs
            independent_chr_variants[chr] = [x[1] for x in snp_to_pos]
        end
        independent_variants = vcat(values(independent_chr_variants)...)
        # Check all variants in estimands have been found
        notfound = setdiff(estimand_variants, independent_variants)
        isempty(notfound) || throw(ArgumentError(string("Did not find some estimands' variants in geneATLAS or Imputed files: ", notfound)))
        # Limit Number of variants to max_variants
        trait_to_variants[trait] = if length(independent_variants) > max_variants
            non_requested_variants = shuffle(setdiff(independent_variants, estimand_variants))
            non_requested_variants = non_requested_variants[1:max_variants-length(estimand_variants)]
            vcat(collect(estimand_variants), non_requested_variants)
        else
            independent_variants
        end

        # Remove trait geneATLAS dir
        remove_ga_data && rm(trait_outdir, recursive=true)
    end

    # Remove whole geneATLAS dir
    remove_ga_data && rm(gene_atlas_dir, recursive=true)
end

"""
    get_trait_key_map(traits; trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))

Retrieve the geneAtlas key from the Description. This will fail if not all traits are present in the geneAtlas.
"""
function get_trait_key_map(traits; trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"))
    trait_table = CSV.read(trait_table_path, DataFrame)
    trait_subset = filter(x -> x.Description ∈ traits, trait_table)
    not_found_traits = setdiff(traits, trait_subset.Description)
    @assert isempty(not_found_traits) || throw(ArgumentError(string("Did not find the following traits in the geneATLAS: ", not_found_traits)))
    return Dict(row.Description => row.key for row in eachrow(trait_subset))
end

function group_by_outcome(estimands)
    groups = Dict()
    for Ψ ∈ estimands
        outcome = TargeneCore.get_outcome(Ψ)
        if haskey(groups, outcome)
            push!(groups[outcome], Ψ)
        else
            groups[outcome] = [Ψ]
        end
    end
    return groups
end

function get_trait_to_variants(estimands; 
    verbosity=0, 
    gene_atlas_dir="gene_atlas_data",
    remove_ga_data=true,
    trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"),
    maf_threshold=0.01,
    pvalue_threshold=1e-5,
    distance_threshold=1e6,
    max_variants=100,
    bgen_prefix=nothing
    )
    verbosity > 0 && @info("Retrieve significant variants for each outcome.")
    # Retrieve traits and variants from estimands
    trait_to_variants = get_trait_to_variants_from_estimands(estimands)
    # Retrieve Trait to geneAtlas key map
    trait_key_map = get_trait_key_map(keys(trait_to_variants), trait_table_path=trait_table_path)
    # Update variant set for each trait using geneAtlas summary statistics
    update_trait_to_variants_from_gene_atlas!(trait_to_variants, trait_key_map; 
        gene_atlas_dir=gene_atlas_dir,
        remove_ga_data=remove_ga_data,
        maf_threshold=maf_threshold,
        pvalue_threshold=pvalue_threshold,
        distance_threshold=distance_threshold,
        max_variants=max_variants,
        bgen_prefix=bgen_prefix
    )
    return trait_to_variants
end

function get_dataset_and_validated_estimands(
    estimands,
    bgen_prefix,
    traits_file,
    pcs_file,
    trait_to_variants;
    call_threshold=0.9,
    positivity_constraint=0.,
    verbosity=0
    )
    verbosity > 0 && @info("Calling genotypes.")
    variants_set = Set(string.(vcat(values(trait_to_variants)...)))

    genotypes = TargeneCore.call_genotypes(
        bgen_prefix, 
        variants_set, 
        call_threshold
    )
    ## Read PCs and traits
    traits = TargeneCore.read_csv_file(traits_file)
    pcs = TargeneCore.read_csv_file(pcs_file)
    ## Merge all together
    dataset = TargeneCore.merge(traits, pcs, genotypes)

    # Validate Estimand
    verbosity > 0 && @info("Validating estimands.")
    variables = TargeneCore.get_variables(estimands, traits, pcs)
    estimands = TargeneCore.adjusted_estimands(
        estimands, variables, dataset; 
        positivity_constraint=positivity_constraint
    )
    return dataset, estimands
end

function write_densities(output_prefix, trait_to_variants, estimands)
    # Initialise densities from trait_to_variants
    conditional_densities = Dict(outcome => Set(variants) for (outcome, variants) in trait_to_variants)
    for Ψ ∈ estimands
        # Update outcome's parents list
        outcome = TargeneCore.get_outcome(Ψ)
        treatments = TargeneCore.get_treatments(Ψ)
        new_parents = string.(union(
            TargeneCore.get_outcome_extra_covariates(Ψ),
            treatments,
            TargeneCore.get_all_confounders(Ψ)
        ))
        union!(
            conditional_densities[string(outcome)],
            new_parents
        )
        # Add treatment mechanism if not already present
        for treatment in treatments
            if !haskey(conditional_densities, treatment)
                conditional_densities[string.(treatment)] = Set(string.(TargeneCore.get_confounders(Ψ, treatment)))
            end
        end
    end
    density_index = 1
    for (outcome, parents) in conditional_densities
        open(string(output_prefix, ".conditional_density_", density_index, ".json"), "w") do io
            JSON.print(io, Dict("outcome" => outcome, "parents" => collect(parents)), 1)
        end
        density_index += 1
    end
end

function write_ga_simulation_inputs(
    output_prefix,
    dataset,
    estimands,
    trait_to_variants;
    batchsize=10,
    verbosity=0
    )
    verbosity > 0 && @info("Writing outputs.")
    # Writing estimands and dataset
    TargeneCore.write_tl_inputs(output_prefix, dataset, estimands; batch_size=batchsize)
    # Writing densities
    write_densities(output_prefix, trait_to_variants, estimands)
end

"""
For the simulation we require treatment generating processes 
and outcome generating processes. We infer these required generating processes 
from the estimands. To ensure that there is no 
causal gap for any given estimand all confounders need to be 
observed for each treatment. This means that each treatment must have exactly the 
same set of confounders in all estimands.
"""
function check_only_one_set_of_confounders_per_treatment(estimands)
    treatment_to_confounders = Dict()
    for Ψ ∈ estimands
        for treatment ∈ TargeneCore.get_treatments(Ψ)
            confounders = TargeneCore.get_confounders(Ψ, treatment)
            current_confounders = get!(treatment_to_confounders, treatment, confounders)
            if confounders != current_confounders
                throw(ArgumentError(string("Two estimands define two distinct sets of confounders for treatment variables: ", treatment)))
            end
        end
    end
end

function read_and_validate_estimands(estimands_prefix)
    estimands = reduce(
        vcat, 
        TargetedEstimation.read_estimands_config(f).estimands for f ∈ files_matching_prefix(estimands_prefix)
    )
    check_only_one_set_of_confounders_per_treatment(estimands)
    return estimands
end

"""
    simulation_inputs_from_gene_atlas(
        estimands_prefix, 
        bgen_prefix, 
        traits_file, 
        pcs_file;
        gene_atlas_dir="gene_atlas_data",
        remove_ga_data=true,
        trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"),
        maf_threshold=0.01,
        pvalue_threshold=1e-5,
        distance_threshold=1e6,
        max_variants=100,
        output_prefix="ga_sim_input",
        batchsize=10,
        positivity_constraint=0,
        call_threshold=0.9,
        verbosity=0,
        )

This function generates input files for realistic simulations using 
variants identified from the geneATLAS.

## What files are Generated ?

The Generated input files, prefixed by `output_prefix`, are:

- A dataset: Combines `pcs_file`, `traits_file` and called genotypes from `bgen_prefix` at `call_threshold`.
- Validated estimands: Estimands from `estimands_prefix` are validated. We makes sure the estimands match 
the data in the dataset and pass the `positivity_constraint`. They are then written in batches of size `batchsize`.
- Conditional Densities: To be learnt to simulate new data for the estimands of interest.

## How are conditional densities generated ?

For the purpose of simulation, we need to learn a data generating process which is 
represented by a set of:

- conditional densities for the treatments (genetic variations, ...)
- conditional densities for the outcomes

We infer the densities that are necessary for the simulation from the provided estimands.

For outcome variables, the union of all parents across estimands is used as well as 
genetic variants GWAS hits selected from the geneATLAS.

For treatment variables, the set of parents must be equal across estimands 
to make sure there is no causal gap since we learn only one density per treatment.

## How are variants selected from the geneATLAS ?

Association data is downloaded to `gene_atlas_dir` and cleaned afterwards 
if `remove_ga_data`. For each outcome, variants are selected if:

- They are in the impute files identified by the `bgen_prefix`
- They pass a significance threshold: `pvalue_threshold`.
- They are at least `distance_threshold` away from other variants.
- They are frequent: `maf_threshold`.

Finally, a maximum of `max_variants` is retained per outcome.
"""
function simulation_inputs_from_gene_atlas(
    estimands_prefix, 
    bgen_prefix, 
    traits_file, 
    pcs_file;
    gene_atlas_dir="gene_atlas_data",
    remove_ga_data=true,
    trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"),
    maf_threshold=0.01,
    pvalue_threshold=1e-5,
    distance_threshold=1e6,
    max_variants=100,
    output_prefix="ga_sim_input",
    batchsize=10,
    positivity_constraint=0,
    call_threshold=0.9,
    verbosity=0,
    )
    Random.seed!(123)
    estimands = read_and_validate_estimands(estimands_prefix)
    # Trait to variants from geneATLAS
    trait_to_variants = get_trait_to_variants(estimands; 
        verbosity=verbosity, 
        gene_atlas_dir=gene_atlas_dir,
        remove_ga_data=remove_ga_data,
        trait_table_path=trait_table_path,
        maf_threshold=maf_threshold,
        pvalue_threshold=pvalue_threshold,
        distance_threshold=distance_threshold,
        max_variants=max_variants,
        bgen_prefix=bgen_prefix
        )
    # Dataset and validated estimands
    dataset, estimands = get_dataset_and_validated_estimands(
        estimands,
        bgen_prefix,
        traits_file,
        pcs_file,
        trait_to_variants;
        call_threshold=call_threshold,
        positivity_constraint=positivity_constraint,
        verbosity=verbosity
    )
    # Write outputs
    write_ga_simulation_inputs(
        output_prefix,
        dataset,
        estimands,
        trait_to_variants;
        batchsize=batchsize,
        verbosity=verbosity
        )
    verbosity > 0 && @info("Done.")
    return 0
end