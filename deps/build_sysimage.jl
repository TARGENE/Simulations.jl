using PackageCompiler
PackageCompiler.create_sysimage(
    ["Simulations"],
    cpu_target="generic",
    sysimage_path="Simulations.so", 
    precompile_execution_file="deps/execute.jl", 
)
