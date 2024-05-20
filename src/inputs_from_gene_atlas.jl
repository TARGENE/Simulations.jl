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

function get_trait_to_variants_from_estimands(estimands; regex=r"^rs[0-9]*")
    trait_to_variants = Dict()
    for Ψ in estimands
        outcome = get_outcome(Ψ)
        variants = filter(x -> occursin(regex, x), string.(get_treatments(Ψ)))
        if haskey(trait_to_variants, outcome) && !isempty(variants)
            union!(trait_to_variants[outcome], variants)
        else
            trait_to_variants[string(outcome)] = Set(variants)
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
    max_variants=100
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
            associations.NOTIN_ESTIMAND = [v ∉ estimand_variants for v in associations.SNP]
            # Only keep variants in estimands or bi-allelic SNPs with "sufficient" MAF and p-value
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
        # Check all variants in estimands have been found (i.e genotyped)
        notfound = setdiff(estimand_variants, independent_variants)
        isempty(notfound) || throw(ArgumentError(string("Did not find some estimands' variants in geneATLAS: ", notfound)))
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
        outcome = Simulations.get_outcome(Ψ)
        if haskey(groups, outcome)
            push!(groups[outcome], Ψ)
        else
            groups[outcome] = [Ψ]
        end
    end
    return groups
end

function simulation_inputs_from_gene_atlas(estimands_prefix;
    gene_atlas_dir="gene_atlas_data",
    remove_ga_data=true,
    trait_table_path=joinpath("assets", "Traits_Table_GeneATLAS.csv"),
    maf_threshold=0.01,
    pvalue_threshold=1e-5,
    distance_threshold=1e6,
    max_variants=100,
    output_prefix="ga_sim_input",
    batchsize=10
    )
    estimands = reduce(
        vcat, 
        TargetedEstimation.read_estimands_config(f).estimands for f ∈ files_matching_prefix(estimands_prefix)
    )
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
        max_variants=max_variants
    )

    # Write all variants for dataset extraction
    unique_variants = unique(vcat(values(trait_to_variants)...))
    open(string(output_prefix, "_variants.txt"), "w") do io
        for v in unique_variants
            write(io, string(v, "\n"))
        end
    end
    # Group and Write Estimands
    batch_index = 1
    for (outcome, estimands_group) ∈ group_by_outcome(estimands)
        # Optimize order within a group and write estimands
        estimands_group = groups_ordering(estimands_group)
        for batch in Iterators.partition(estimands_group, batchsize)
            batch_filename = string(output_prefix, "_estimands_", batch_index, ".jls")
            serialize(batch_filename, TMLE.Configuration(estimands=batch))
            batch_index += 1
        end
    end
    # Write densities
    density_index = 1
    confounders = get_confounders_assert_equal(estimands)
    covariates = get_covariates_assert_equal(estimands)
    ## Outcome densities
    for (outcome, variants) in trait_to_variants
        parents = vcat(variants, confounders..., covariates...)
        open(string(output_prefix, "_conditional_density_", density_index, ".json"), "w") do io
            JSON.print(io, Dict("outcome" => outcome, "parents" => parents), 1)
        end
        density_index += 1
    end
    ## Propensity scores
    for variant in unique_variants
        open(string(output_prefix, "_conditional_density_", density_index, ".json"), "w") do io
            JSON.print(io, Dict("outcome" => variant, "parents" => confounders), 1)
        end
        density_index += 1
    end

    return 0
end