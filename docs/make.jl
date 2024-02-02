using CalibrationTests
using Documenter

DocMeta.setdocmeta!(
    CalibrationTests, :DocTestSetup, :(using CalibrationTests); recursive=true
)

makedocs(;
    modules=[CalibrationTests],
    authors="David Widmann",
    repo="https://github.com/devmotion/CalibrationTests.jl/blob/{commit}{path}#{line}",
    sitename="CalibrationTests.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://devmotion.github.io/CalibrationTests.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md", "api.md"],
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/devmotion/CalibrationTests.jl", push_preview=true, devbranch="main"
)
