using CalibrationTests
using CalibrationErrors
using Distributions
using StatsBase

using Random
using Test

Random.seed!(1234)

@testset "estimate, statistic, and kernel matrix" begin
    kernel = transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel()
    biasedskce = BiasedSKCE(kernel)
    unbiasedskce = UnbiasedSKCE(kernel)

    for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
        # sample predictions and targets
        dist = Dirichlet(nclasses, 1)
        predictions = [rand(dist) for _ in 1:nsamples]
        targets_consistent = [rand(Categorical(prediction)) for prediction in predictions]
        targets_onlyone = ones(Int, length(predictions))

        # compute calibration error estimate and test statistic
        for targets in (targets_consistent, targets_onlyone)
            estimate, statistic, kernelmatrix = CalibrationTests.estimate_statistic_kernelmatrix(
                kernel, predictions, targets
            )

            @test estimate ≈ calibrationerror(unbiasedskce, predictions, targets)
            @test statistic ≈
                  nsamples / (nsamples - 1) *
                  calibrationerror(unbiasedskce, predictions, targets) -
                  calibrationerror(biasedskce, predictions, targets)
            @test kernelmatrix ≈
                  CalibrationErrors.unsafe_skce_eval.(
                (kernel,),
                predictions,
                targets,
                permutedims(predictions),
                permutedims(targets),
            )
        end
    end
end

@testset "consistency" begin
    kernel1 = transform(ExponentialKernel(), 0.1)
    kernel2 = WhiteKernel()
    kernel = kernel1 ⊗ kernel2
    skce = UnbiasedSKCE(kernel)
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
            test_consistent = AsymptoticSKCETest(skce, predictions, targets_consistent)
            test_onlyone = AsymptoticSKCETest(skce, predictions, targets_onlyone)

            # estimate pvalues
            pvalues_consistent[i] = pvalue(test_consistent; bootstrap_iters=200)
            pvalues_onlyone[i] = pvalue(test_onlyone; bootstrap_iters=200)
        end

        # compute empirical test errors
        ecdf_consistent = ecdf(pvalues_consistent)
        @test all(ecdf_consistent(α) ≤ α + 0.1 for α in αs)
        @test all(p < 0.05 for p in pvalues_onlyone)
    end
end
