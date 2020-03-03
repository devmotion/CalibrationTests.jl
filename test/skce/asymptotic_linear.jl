using CalibrationTests
using CalibrationErrors
using Distributions
using StatsBase
using StatsFuns

using Random
using Test

Random.seed!(1234)

@testset "estimate, stderr, and z" begin
    skce = LinearUnbiasedSKCE(transform(ExponentialKernel(), 0.1), WhiteKernel())

    for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
        # sample predictions and targets
        dist = Dirichlet(nclasses, 1)
        predictions = [rand(dist) for _ in 1:nsamples]
        targets_consistent = [rand(Categorical(prediction)) for prediction in predictions]
        targets_onlyone = ones(Int, length(predictions))

        # for both sets of targets
        for targets in (targets_consistent, targets_onlyone)
            test = AsymptoticLinearTest(skce, predictions, targets)

            @test test.n == div(nsamples, 2)
            @test test.estimate ≈ calibrationerror(skce, predictions, targets)
            @test test.z == test.estimate / test.stderr

            @test pvalue(test) == pvalue(test; tail = :right) == normccdf(test.z)
            @test_throws ArgumentError pvalue(test; tail = :left)
            @test_throws ArgumentError pvalue(test; tail = :both)

            for α in 0.55:0.05:0.95
                q = norminvcdf(α)
                @test confint(test; level = α) == confint(test; level = α, tail = :right) ==
                    (test.estimate - q * test.stderr, Inf)
                @test_throws ArgumentError confint(test; level = α, tail = :left)
                @test_throws ArgumentError confint(test; level = α, tail = :both)
            end
        end
    end
end

@testset "consistency" begin
    skce = LinearUnbiasedSKCE(transform(ExponentialKernel(), 0.1), WhiteKernel())
    αs = 0.05:0.1:0.95
    nsamples = 100

    pvalues_consistent = Vector{Float64}(undef, 100)
    pvalues_onlyone = similar(pvalues_consistent)

    for nclasses in (2, 10)
        dist = Dirichlet(nclasses, 1)
        predictions = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
        targets_consistent = Vector{Int}(undef, nsamples)
        targets_onlyone = ones(Int, nsamples)

        for i in eachindex(pvalues_consistent)
            # sample predictions and targets
            for j in 1:nsamples
                rand!(dist, predictions[j])
                targets_consistent[j] = rand(Categorical(predictions[j]))
            end

            # define test
            test_consistent = AsymptoticLinearTest(skce, predictions, targets_consistent)
            test_onlyone = AsymptoticLinearTest(skce, predictions, targets_onlyone)

            # estimate pvalues
            pvalues_consistent[i] = pvalue(test_consistent)
            pvalues_onlyone[i] = pvalue(test_onlyone)
        end

        # compute empirical test errors
        @test ecdf(pvalues_consistent).(αs) ≈ αs atol = 0.16
        @test all(iszero, 1 .- ecdf(pvalues_onlyone).(αs))
    end
end
