using Simulations
using Test

TESTDIR = joinpath(pkgdir(Simulations), "test")

@testset "Simulations.jl" begin
    @test include(joinpath(TESTDIR, "utils.jl"))
    @test include(joinpath(TESTDIR, "results_aggregation.jl"))
    @test include(joinpath(TESTDIR, "null_simulation.jl"))
    @test include(joinpath(TESTDIR, "realistic_simulation_inputs.jl"))

    @test include(joinpath(TESTDIR, "density_estimation", "glm.jl"))
    @test include(joinpath(TESTDIR, "density_estimation", "neural_net.jl"))
    @test include(joinpath(TESTDIR, "density_estimation", "density_estimation.jl"))
    
    @test include(joinpath(TESTDIR, "samplers", "null_sampler.jl"))
    @test include(joinpath(TESTDIR, "samplers", "density_estimate_sampler.jl"))    
end
