using CalibrationTests
using CalibrationErrors
using Distributions
using StatsBase
using StatsFuns

using Random
using Test

Random.seed!(1234)

@testset "estimate, stderr, and z" begin
    for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
        for blocksize in (2, 5, 10, 50)
            # blocksize may no be greater than number of samples
            blocksize < nsamples || continue

            # create estimator
            skce = BlockUnbiasedSKCE(
                transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel(), blocksize,
            )

            # sample predictions and targets
            dist = Dirichlet(nclasses, 1)
            predictions = [rand(dist) for _ in 1:nsamples]
            targets_consistent = [rand(Categorical(prediction)) for prediction in
                                  predictions]
            targets_onlyone = ones(Int, length(predictions))

            # for both sets of targets
            for targets in (targets_consistent, targets_onlyone)
                test = AsymptoticBlockSKCETest(skce, predictions, targets)

                @test test.blocksize == blocksize
                @test test.nblocks == nsamples ÷ blocksize
                @test test.estimate ≈ calibrationerror(skce, predictions, targets)
                @test test.z == test.estimate / test.stderr

                @test pvalue(test) == pvalue(test; tail = :right) == normccdf(test.z)
                @test_throws ArgumentError pvalue(test; tail = :left)
                @test_throws ArgumentError pvalue(test; tail = :both)

                for α in 0.55:0.05:0.95
                    q = norminvcdf(α)
                    @test confint(test; level = α) ==
                        confint(test; level = α, tail = :right) ==
                        (max(0, test.estimate - q * test.stderr), Inf)
                    @test_throws ArgumentError confint(test; level = α, tail = :left)
                    @test_throws ArgumentError confint(test; level = α, tail = :both)
                end
            end
        end
    end
end

@testset "consistency" begin
    αs = 0.05:0.1:0.95
    nsamples = 100

    pvalues_consistent = Vector{Float64}(undef, 100)
    pvalues_onlyone = similar(pvalues_consistent)

    for blocksize in (2, 5, 10)
        # create block estimator
        skce = BlockUnbiasedSKCE(
            transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel(), blocksize,
        )

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
                test_consistent = AsymptoticBlockSKCETest(skce, predictions, targets_consistent)
                test_onlyone = AsymptoticBlockSKCETest(skce, predictions, targets_onlyone)

                # estimate pvalues
                pvalues_consistent[i] = pvalue(test_consistent)
                pvalues_onlyone[i] = pvalue(test_onlyone)
            end

            # compute empirical test errors
            @test all(ecdf(pvalues_consistent)(α) ≤ α + 0.08 for α in αs)
            @test all(iszero, 1 .- ecdf(pvalues_onlyone).(αs))
        end
    end
end
