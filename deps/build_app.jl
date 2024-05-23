using PackageCompiler
PackageCompiler.create_app(".", "targene-simulation",
    executables = ["targene-simulation" => "julia_main"],
    precompile_execution_file="deps/execute.jl", 
    include_lazy_artifacts=true
)
