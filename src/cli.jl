function cli_settings()
    s = ArgParseSettings(description="PopGenSim CLI.")

    @add_arg_table! s begin
        "estimation"
            action = :command
            help = "Run Estimation from Permutation Null Sampler."

        "aggregate"
            action = :command
            help = "Aggregate multiple results file created by estimation procedures."

        "density-estimation-inputs"
            action = :command
            help = "Generates density estimation inputs."

        "density-estimation"
            action = :command
            help = "Estimate a conditional density."
        
        "simulation-inputs-from-ga"
            action = :command
            help = "Generate simulation inputs from geneATLAS."

        "analyse"
            action = :command
            help = "Run analyses script and generate plots."
    end

    @add_arg_table! s["analyse"] begin
        "results-file"
            arg_type = String
            help = "Aggregated result file output by the `aggregate` command."
        
        "estimands-prefix"
            arg_type = String
            help = "Prefix to estimands files."

        "--out-dir"
            arg_type = String
            default = "analysis_results"
            help = "Output directory."
        
        "--n"
            arg_type = Int
            default = 500_000
            help = "Number of samples used to compute the ground truth value of the estimands."
        
        "--dataset-file"
            arg_type = String
            help = "Dataset file to use to sample data and compute ground truth values (if --density_estimates_prefix is specified)."

        "--density-estimates-prefix"
            arg_type = String
            help = string("If specified, a prefix to density estimates. ",
                    "It is thus assumed that the results-file was generated using these densities. ", 
                    "If left unspecified, the NullSampler is used and all effects are assumed to be 0.")

    end

    @add_arg_table! s["aggregate"] begin
        "input-prefix"
            arg_type = String
            help = "Prefix to all files to be aggregated."
        "out"
            arg_type = String
            help = "Output path."
    end

    @add_arg_table! s["estimation"] begin
        "origin-dataset"
            arg_type = String
            help = "Path to the dataset (either .csv or .arrow)"

        "estimands-config"
            arg_type = String
            help = "A string (`factorialATE`) or a serialized TMLE.Configuration (accepted formats: .json | .yaml | .jls)"

        "estimators-config"
            arg_type = String
            help = "A julia file containing the estimators to use."
        
        "--density-estimates-prefix"
            arg_type = String
            help = "If specified, a prefix to density estimates, otherwise permutation sampling is perfomed."

        "--sample-size"
            arg_type = Int
            help = "Size of simulated dataset."
            default = nothing

        "--n-repeats"
            arg_type = Int
            help = "Number of simulations to run."
            default = 10

        "--out"
            arg_type = String
            default = "permutation_estimation_results.hdf5"
            help = "Output file."
        
        "--verbosity"
            arg_type = Int
            default = 0
            help = "Verbosity level"

        "--chunksize"
            arg_type = Int
            help = "Results are written in batches of size chunksize."
            default = 100

        "--rng"
            arg_type = Int
            help = "Random seed (Only used for estimands ordering at the moment)."
            default = 123
        
        "--workdir"
            arg_type = String
            help = "Working directory"
            default = mktempdir()
    end

    @add_arg_table! s["density-estimation-inputs"] begin
        "dataset"
            arg_type = String
            help = "Path to the dataset (either .csv or .arrow)"

        "estimands-prefix"
            arg_type = String
            help = "A prefix to serialized TMLE.Configuration (accepted formats: .json | .yaml | .jls)"

        "--output-prefix"
            arg_type = String
            default = "de_inputs"
            help = "Output JSON file."
        "--batchsize"
            arg_type = Int
            default = 10
            help = "Estimands are batched to optimize speed by //"
    end

    @add_arg_table! s["density-estimation"] begin
        "dataset"
            arg_type = String
            help = "Path to the dataset (either .csv or .arrow)"

        "density-file"
            arg_type = String
            help = "YAML file with an `outcome` field and a `parents` field"

        "--mode"
            arg_type = String
            default = "study"
            help = "study or test"

        "--output"
            arg_type = String
            default = "conditional_density.hdf5"
            help = "Output JSON file."
        
        "--train-ratio"
            arg_type = Int
            default = 10
            help = "The dataset is split using this ratio."

        "--verbosity"
            arg_type = Int
            default = 0
            help = "Verbosity level."

    end

    @add_arg_table! s["simulation-inputs-from-ga"] begin
        "estimands-prefix"
            arg_type = String
            help = "A prefix to serialized TMLE.Configuration (accepted formats: .json | .yaml | .jls)"

        "--ga-download-dir"
            arg_type = String
            default = "gene_atlas_data"
            help = "Where the geneATLAS data will be downloaded"

        "--ga-trait-table"
            arg_type = String
            default = joinpath("assets", "Traits_Table_GeneATLAS.csv")
            help = "geneATLAS Trait Table."

        "--remove-ga-data"
            arg_type = Bool
            default = true
            help = "Removes geneATLAS downloaded data after execution."

        "--maf-threshold"
            arg_type = Float64
            default = 0.01
            help = "Only variants with at least `maf-threshold` are selected."
        
        "--pvalue-threshold"
            arg_type = Float64
            default = 1e-5
            help = "Only variants with pvalue lower than `pvalue-threhsold` are selected."

        "--distance-threshold"
            arg_type = Float64
            default = 1e6
            help = "Only variants that are at least `distance-threhsold` away from each other are selected."
        
        "--max-variants"
            arg_type = Int
            default = 100
            help = "Maximum variants retrieved per trait."

        "--output-prefix"
            arg_type = String
            default = "ga_sim_input"
            help = "Prefix to outputs."
        
        "--batchsize"
            arg_type = Int
            default = 10
            help = "Estimands are further split in files of `batchsize`"
            
    end

    return s
end

function julia_main()::Cint
    settings = parse_args(ARGS, cli_settings())
    cmd = settings["%COMMAND%"]
    cmd_settings = settings[cmd]

    if cmd == "estimation"
        estimate_from_simulated_data(
            cmd_settings["origin-dataset"],
            cmd_settings["estimands-config"],
            cmd_settings["estimators-config"];
            sample_size=cmd_settings["sample-size"],
            sampler_config=cmd_settings["density-estimates-prefix"],
            nrepeats=cmd_settings["n-repeats"],
            out=cmd_settings["out"],
            verbosity=cmd_settings["verbosity"],
            rng_seed=cmd_settings["rng"], 
            chunksize=cmd_settings["chunksize"],
            workdir=cmd_settings["workdir"]
            )
    elseif cmd == "aggregate"
        save_aggregated_df_results(cmd_settings["input-prefix"], cmd_settings["out"])
    elseif cmd == "simulation-inputs-from-ga"
        simulation_inputs_from_gene_atlas(
            cmd_settings["estimands-prefix"];
            gene_atlas_dir=cmd_settings["ga-download-dir"],
            remove_ga_data=cmd_settings["remove-ga-data"], 
            trait_table_path=cmd_settings["ga-trait-table"],
            maf_threshold=cmd_settings["maf-threshold"],
            pvalue_threshold=cmd_settings["pvalue-threshold"],
            distance_threshold=cmd_settings["distance-threshold"],
            output_prefix=cmd_settings["output-prefix"],
            batchsize=cmd_settings["batchsize"],
            max_variants=cmd_settings["max-variants"]
        )
    elseif cmd == "density-estimation-inputs"
        density_estimation_inputs(
            cmd_settings["dataset"],
            cmd_settings["estimands-prefix"];
            batchsize=cmd_settings["batchsize"],
            output_prefix=cmd_settings["output-prefix"]
        )
    elseif cmd == "density-estimation"
        density_estimation(
            cmd_settings["dataset"],
            cmd_settings["density-file"];
            mode=cmd_settings["mode"],
            output=cmd_settings["output"],
            train_ratio=cmd_settings["train-ratio"],
            verbosity=cmd_settings["verbosity"]
        )
    elseif cmd == "analyse"
        analyse(
            cmd_settings["results-file"],
            cmd_settings["estimands-prefix"];
            out_dir=cmd_settings["out-dir"],
            n=cmd_settings["n"],
            dataset_file=cmd_settings["dataset-file"],
            density_estimates_prefix=cmd_settings["density-estimates-prefix"],
        )
    end

    return 0
end