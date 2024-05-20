using Simulations
using Test

TESTDIR = joinpath(pkgdir(Simulations), "test")

@testset "Simulations.jl" begin
    # Unit Tests
    @test include(joinpath(TESTDIR, "utils.jl"))
    @test include(joinpath(TESTDIR, "inputs_from_gene_atlas.jl"))

    @test include(joinpath(TESTDIR, "density_estimation", "glm.jl"))
    @test include(joinpath(TESTDIR, "density_estimation", "neural_net.jl"))
    @test include(joinpath(TESTDIR, "density_estimation", "density_estimation.jl"))

    
    @test include(joinpath(TESTDIR, "samplers", "null_sampler.jl"))
    @test include(joinpath(TESTDIR, "samplers", "density_estimate_sampler.jl"))

    # Integration Tests
    @test include(joinpath(TESTDIR, "null_simulation.jl"))
    @test include(joinpath(TESTDIR, "gene_atlas_simulation.jl"))
end
