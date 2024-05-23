using Simulations

@info "Running precompilation script."
# Run workload
TEST_DIR = joinpath(pkgdir(Simulations), "test")
push!(LOAD_PATH, TEST_DIR)
include(joinpath(TEST_DIR, "runtests.jl"))