using Documenter, SeqSpace

makedocs(
    sitename = "SeqSpace.jl",
    authors  = "Nicholas Noll, Madhav Mani, Boris Shraiman",
    modules  = [SeqSpace],
    pages    = [
        "Home"         => "index.md",
        "Science"      => [
            "sci/normalize.md",
            "sci/inference.md",
            "sci/autoencode.md",
        ],
        "Library"      => [
            "lib/distance.md",
            "lib/generate.md",
            "lib/infer.md",
            "lib/io.md",
            "lib/manifold.md",
            "lib/mixtures.md",
            "lib/mle.md",
            "lib/model.md",
            "lib/pointcloud.md",
            "lib/queue.md",
            "lib/rank.md",
            "lib/scrna.md",
            "lib/util.md",
            "lib/voronoi.md",
        ],
        "Command Line" => [
            "cli/normalize.md",
        ],
    ]
)
