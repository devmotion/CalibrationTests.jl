using CalibrationTests
using CalibrationErrors
using Distributions
using StableRNGs
using StatsBase

using Random
using Statistics
using Test

Random.seed!(1234)

@testset "CalibrationTests" begin
    @testset "Binary trend" begin
        include("binary_trend.jl")
    end

    @testset "Consistency test" begin
        include("consistency.jl")
    end

    @testset "SKCE" begin
        @testset "Asymptotic" begin
            include("skce/asymptotic.jl")
        end
        @testset "Asymptotic block" begin
            include("skce/asymptotic_block.jl")
        end
        @testset "Distribution-free" begin
            include("skce/distribution_free.jl")
        end
    end

    @testset "Asymptotic CME" begin
        include("cme.jl")
    end
end
