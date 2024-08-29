########################################################################
###                          Misc Functions                          ###
########################################################################

keep_only_imputed(associations, bgen_prefix::Nothing, chr) = associations

function keep_only_imputed(associations, bgen_prefix, chr)
    imputed_rsids = DataFrame(SNP=rsids(read_bgen_chromosome(bgen_prefix, chr)))
    return innerjoin(associations, imputed_rsids, on=:SNP)
end

get_bgen_chromosome(bgen_files, chr) = only(filter(
    x -> endswith(x, Regex(string("[^0-9]", chr, ".bgen"))), 
    bgen_files
    ))

function read_bgen_chromosome(bgen_prefix, chr)
    bgen_files = TargeneCore.files_matching_prefix(bgen_prefix)
    bgen_file = get_bgen_chromosome(bgen_files, chr)
    return TargeneCore.read_bgen(bgen_file)
end

function coerce_parents_and_outcome!(dataset, parents; outcome=nothing)
    TMLECLI.coerce_types!(dataset, parents)
    if outcome !== nothing
        # Continuous and Counts except Binary outcomes are treated as continuous
        if elscitype(dataset[!, outcome]) <: Union{Infinite, Missing} && !(TMLECLI.isbinary(outcome, dataset))
            TMLECLI.coerce_types!(dataset, [outcome], rules=:discrete_to_continuous)
        else
            TMLECLI.coerce_types!(dataset, [outcome], rules=:few_to_finite)
        end
    end
end

function variables_from_dataset(dataset)
    confounders = Set([])
    outcome_extra_covariates = Set(["Genetic-Sex", "Age-Assessment"])
    outcomes = Set([])
    variants = Set([])
    for colname in names(dataset)
        if colname == "SAMPLE_ID"
            continue
        elseif startswith(colname, r"PC[0-9]*")
            push!(confounders, colname)
        elseif startswith(colname, r"rs[0-9]*")
            push!(variants, colname)
        elseif colname ∈ outcome_extra_covariates
            continue
        else
            push!(outcomes, colname)
        end
    end
    variables = (
        outcomes = collect(outcomes), 
        variants = collect(variants), 
        confounders = collect(confounders), 
        outcome_extra_covariates = collect(outcome_extra_covariates)
    )
    return variables
end

serializable!(estimator) = estimator

MLJBase.restore!(estimator) = estimator

continuous_encoder() = ContinuousEncoder(drop_last=true)

getlabels(col::CategoricalVector) = levels(col)
getlabels(col) = nothing

get_input_size(::Type{Multiclass{N}}) where N = N - 1 
get_input_size(::Type) = 1
get_input_size(x::AbstractVector) = get_input_size(elscitype(x))
get_input_size(X) = sum(get_input_size(x) for x in eachcol(X))

propensity_score_inputs(variables) = collect(variables.confounders)
outcome_model_inputs(variables) = vcat(collect(variables.treatments), collect(variables.confounders), collect(variables.outcome_extra_covariates))
confounders_and_covariates(variables) = vcat(collect(variables.confounders), collect(variables.outcome_extra_covariates))

countuniques(dataset, colname) = DataFrames.combine(groupby(dataset, colname, skipmissing=true), nrow)

levels_missing(sampled_vector, origin_vector) = Set(skipmissing(sampled_vector)) != Set(skipmissing(origin_vector))

"""
Checks that the multiclass variables have all their levels present in the sampled dataset
"""
function check_sampled_levels(sampled_dataset, origin_dataset, variables_to_check)
    for variable in variables_to_check
        if levels_missing(sampled_dataset[!, variable], origin_dataset[!, variable])
            return true, string(variable)
        end
    end
    return false, ""
end

ismulticlass(col) = autotype(col, :few_to_finite) <: Union{Missing, <:Multiclass}

multiclass_variables(origin_dataset, variables) =
    filter(v -> ismulticlass(origin_dataset[!, v]), variables)

"""
    sample_from(origin_dataset::DataFrame, variables; 
        n=100, 
        max_attempts=1000,
        verbosity = 1
    )

Tries to jointly sample non-missing values of `variables` from the `origin_dataset` for a maximum of `max_attempts`.
Each sampled dataset is checked as follows:
    1. The levels of sampled factor variables should match the levels in the original data.
"""
function sample_from(origin_dataset::DataFrame, variables; 
    n=100,
    variables_to_check=multiclass_variables(origin_dataset, variables),
    max_attempts=1000,
    verbosity = 1
    )
    variables = collect(variables)
    variables_to_check = intersect(variables, variables_to_check)
    nomissing = dropmissing(origin_dataset[!, variables])
    levels_are_missing, msg = Simulations.check_sampled_levels(nomissing, origin_dataset, variables_to_check)
    if levels_are_missing
        msg = string(
            "Filtering of missing values resulted in missing levels for variable ", msg, 
            ".\n Consider lowering or setting the `call_threshold` to `nothing`."
        )
        throw(ErrorException(msg))
    end
    for attempt in 1:max_attempts
        sample_rows = StatsBase.sample(1:nrow(nomissing), n, replace=true)
        sampled_dataset = nomissing[sample_rows, variables]
        levels_are_missing, msg = check_sampled_levels(sampled_dataset, nomissing, variables_to_check)
        if !levels_are_missing
            return sampled_dataset
        end
        verbosity > 0 && @info(string("Sampled dataset after attempt ", attempt, " had missing levels. In particular: ", msg, ".\n Retrying."))
    end
    msg = string(
        "Could not sample a dataset with all variables' levels in `variables_to_check` after: ", max_attempts, 
        " attempts. Possible solutions: increase `sample_size`, change your simulation estimands or increase `max_attempts`."
    )
    throw(ErrorException(msg))
end

function sampled_vector_has_enough_occurences(sampled_vector, origin_vector; 
    min_occurences=10
    )
    n_uniques = countmap(sampled_vector)
    if length(n_uniques) != length(levels(origin_vector))
        return false, "missing levels."
    end
    if minimum(values(n_uniques)) < min_occurences
        return false, "not enough occurrences for each level."
    end
    return true, ""
end

"""
    sample_from(origin_vector::AbstractVector; 
        n=100,
        min_occurences=10,
        max_attempts=1000,
    )
    Tries to sample non-missing values from a vector for a maximum of `max_attempts`.
    Each sampled vector is checked as follows:
        1. The levels of sampled factor variables should match the levels in the original data.
        2. The lowest populated sampled level of each factor variable should have more than `min_occurences` samples.
"""
function sample_from(origin_dataset::DataFrame, variable::Union{Symbol, AbstractString}; 
    n=100,
    min_occurences=10,
    max_attempts=1000,
    verbosity = 1
    )
    origin_vector = collect(skipmissing(origin_dataset[!, variable]))
    for attempt in 1:max_attempts
        sampled_vector = StatsBase.sample(origin_vector, n, replace=true)
        # If binary of multiclass: check levels and occurences
        if length(levels(sampled_vector)) == 2 || ismulticlass(sampled_vector)
            has_enough_occurences, msg = Simulations.sampled_vector_has_enough_occurences(sampled_vector, origin_vector; 
                min_occurences=min_occurences
            )
            if !has_enough_occurences
                verbosity > 0 && @info(string("The sampled vector for variable ", variable, " had ", msg, "\nRetrying."))
                continue
            end
        end
        return sampled_vector
    end
    throw(ErrorException(string("Could not sample variable ", variable, " because it either did not have enough occurences or some levels were missing after ", max_attempts, 
        " attempts.\nConsider increasing the sample size or changing your estimands.")))
end

variables_from_args(outcome, treatments, confounders, outcome_extra_covariates) = (
    outcome = Symbol(outcome),
    treatments = Symbol.(Tuple(treatments)),
    confounders = Symbol.(Tuple(confounders)),
    outcome_extra_covariates = Symbol.(Tuple(outcome_extra_covariates))
    )

transpose_target(y, labels) = onehotbatch(y, labels)
transpose_target(y, ::Nothing) = Float32.(reshape(y, 1, length(y)))

transpose_table(X) = Float32.(Tables.matrix(X, transpose=true))
transpose_table(estimator, X) =
    transpose_table(MLJBase.transform(estimator.encoder, X))

function get_conditional_densities_variables(estimands)
    conditional_densities_variables = Set{Pair}([])
    for Ψ in estimands
        for factor in TMLE.nuisance_functions_iterator(Ψ)
            push!(conditional_densities_variables, factor.parents => factor.outcome)
        end
    end
    return [Dict("outcome" => pair[2], "parents" => collect(pair[1])) for pair in conditional_densities_variables]
end

function compute_statistics(dataset, Ψ::TMLE.Estimand)
    outcome = get_outcome(Ψ)
    treatments = get_treatments(Ψ)
    nomissing_dataset = dropmissing(dataset, [outcome, treatments..., confounders_and_covariates_set(Ψ)...])
    categorical_variables = TMLECLI.isbinary(outcome, nomissing_dataset) ? (outcome, treatments...) : treatments

    statistics = Dict()
    # Each Variable
    for variable ∈ categorical_variables
        statistics[variable] = DataFrames.combine(groupby(nomissing_dataset, variable), proprow, nrow)
    end
    # Joint treatment
    if length(treatments) > 1
        statistics[treatments] = DataFrames.combine(groupby(nomissing_dataset, collect(treatments)), proprow, nrow)
    end
    # Joint treatment/outcome
    if length(categorical_variables) > length(treatments)
        statistics[categorical_variables] = DataFrames.combine(groupby(nomissing_dataset, collect(categorical_variables)), proprow, nrow)
    end
    return statistics
end

compute_statistics(dataset, estimands) =
    [compute_statistics(dataset, Ψ) for Ψ in estimands]

########################################################################
###                    Train / Validation Splits                     ###
########################################################################

function stratified_holdout_train_val_samples(X, y;
    resampling=StratifiedCV(nfolds=10)
    )
    first(MLJBase.train_test_pairs(resampling, 1:length(y), X, y))
end

function train_validation_split(X, y; 
    train_ratio=10, 
    resampling=StratifiedCV(nfolds=train_ratio)
    )
    # Get Train/Validation Splits
    train_samples, val_samples = stratified_holdout_train_val_samples(X, y; resampling=resampling)
    # Split Data
    X_train = selectrows(X, train_samples)
    X_val = selectrows(X, val_samples)
    y_train = selectrows(y, train_samples)
    y_val = selectrows(y, val_samples)

    return (X_train, y_train, X_val, y_val)
end

########################################################################
###                    Estimand variables accessors                  ###
########################################################################

function confounders_and_covariates_set(Ψ)
    confounders_and_covariates = Set{Symbol}([])
    push!(
        confounders_and_covariates, 
        Iterators.flatten(values(Ψ.treatment_confounders))..., 
        Ψ.outcome_extra_covariates...
    )
    return confounders_and_covariates
end

confounders_and_covariates_set(Ψ::JointEstimand) = 
    union((confounders_and_covariates_set(arg) for arg in Ψ.args)...)