using CalibrationTests
using Aqua
using CalibrationErrors
using Distributions
using StableRNGs
using StatsBase

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "CalibrationTests" begin
    @testset "General" begin
        include("aqua.jl")
    end
    @testset "Binary trend" begin
        include("binary_trend.jl")
    end

    @testset "Consistency test" begin
        include("consistency.jl")
    end

    @testset "SKCE" begin
        include("skce/asymptotic.jl")
        include("skce/asymptotic_block.jl")
        include("skce/distribution_free.jl")
    end

    @testset "Asymptotic CME" begin
        include("cme.jl")
    end
end
