using Simulations
using Documenter

DocMeta.setdocmeta!(Simulations, :DocTestSetup, :(using Simulations); recursive=true)

makedocs(;
    modules=[Simulations],
    authors="Olivier Labayle <olabayle@gmail.com> and contributors",
    sitename="Simulations.jl",
    format=Documenter.HTML(;
        canonical="https://TARGENE.github.io/Simulations.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/TARGENE/Simulations.jl",
    devbranch="main",
)
