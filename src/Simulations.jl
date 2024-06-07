module Simulations

using Distributions
using Random
using TargetedEstimation
using DataFrames
using Flux
using MLJBase
using TMLE
using OneHotArrays
using MLJModels
using MLJLinearModels
using MLJGLMInterface
using CategoricalArrays
using StatsBase
using JLD2
using Tables
using Arrow
using ArgParse
using JSON
using Serialization
using CSV
using TargeneCore
using BGEN

include("utils.jl")

include(joinpath("density_estimation", "glm.jl"))
include(joinpath("density_estimation", "neural_net.jl"))
include(joinpath("density_estimation", "density_estimation.jl"))

include(joinpath("samplers", "null_sampler.jl"))
include(joinpath("samplers", "density_estimate_sampler.jl"))

include(joinpath("inputs_from_gene_atlas.jl"))
include("estimation.jl")
include("cli.jl")

export NullSampler, DensityEstimateSampler
export MixtureDensityNetwork, CategoricalMLP
export NeuralNetworkEstimator, SieveNeuralNetworkEstimator
export GLMEstimator
export sample_from, train!, evaluation_metrics
export density_estimation
export density_estimation_inputs_from_gene_atlas
export estimate_from_simulated_data
export save_aggregated_df_results

end
