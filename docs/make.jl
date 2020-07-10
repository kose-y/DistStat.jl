using Documenter, DistStat

makedocs(
    format = Documenter.HTML(),
    sitename = "DistStat.jl",
    authors = "Seyoon Ko",
    clean = true,
    debug = true,
    pages = [
        "index.md"
    ]
)

deploydocs(
    repo   = "github.com/kose-y/DistStat.jl.git",
    target = "build",
    deps   = nothing,
    make   = nothing
)


