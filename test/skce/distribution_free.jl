using CalibrationTests
using CalibrationErrors
using Distributions
using StatsBase
using StatsFuns

using Random
using Test

Random.seed!(1234)

@testset "bounds" begin
    # default bounds for base kernels
    CalibrationTests.uniformbound(ExponentialKernel()) == 1
    CalibrationTests.uniformbound(SqExponentialKernel()) == 1
    CalibrationTests.uniformbound(TVExponentialKernel()) == 1
    CalibrationTests.uniformbound(WhiteKernel()) == 1

    # default bounds for kernels with input transformations
    CalibrationTests.uniformbound(transform(SqExponentialKernel(), rand())) == 1
    CalibrationTests.uniformbound(transform(TVExponentialKernel(), rand(10))) == 1

    # default bounds for scaled kernels
    CalibrationTests.uniformbound(42 * ExponentialKernel()) == 42

    # default bounds for tensor product kernels
    kernel = (3.2 * SqExponentialKernel()) ⊗ (2.7 * WhiteKernel())
    CalibrationTests.uniformbound(kernel) == 3.2 * 2.7

    # default bounds for kernel terms
    CalibrationTests.uniformbound(BlockUnbiasedSKCE(kernel)) == 2 * 3.2 * 2.7
end

@testset "estimator and estimates" begin
    kernel = transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel()

    for skce in (BiasedSKCE(kernel), UnbiasedSKCE(kernel), BlockUnbiasedSKCE(kernel))
        for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
            # sample predictions and targets
            dist = Dirichlet(nclasses, 1)
            predictions = [rand(dist) for _ in 1:nsamples]
            targets_consistent = [rand(Categorical(prediction)) for prediction in predictions]
            targets_onlyone = ones(Int, length(predictions))

            # for both sets of targets
            for targets in (targets_consistent, targets_onlyone)
                test = DistributionFreeSKCETest(skce, predictions, targets)

                @test test.estimator == skce
                @test test.n == nsamples
                @test test.estimate ≈ calibrationerror(skce, predictions, targets)
                @test test.bound == CalibrationTests.uniformbound(skce)
            end
        end
    end
end

@testset "consistency" begin
    kernel = transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel()
    αs = 0.05:0.1:0.95
    nsamples = 100

    pvalues_consistent = Vector{Float64}(undef, 100)

    for skce in (BiasedSKCE(kernel), UnbiasedSKCE(kernel), BlockUnbiasedSKCE(kernel))
        for nclasses in (2, 10)
            dist = Dirichlet(nclasses, 1)
            predictions = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
            targets_consistent = Vector{Int}(undef, nsamples)

            for i in eachindex(pvalues_consistent)
                # sample predictions and targets
                for j in 1:nsamples
                    rand!(dist, predictions[j])
                    targets_consistent[j] = rand(Categorical(predictions[j]))
                end

                # define test
                test_consistent = DistributionFreeSKCETest(skce, predictions,
                                                           targets_consistent)

                # estimate pvalue
                pvalues_consistent[i] = pvalue(test_consistent)
            end

            # compute empirical test errors
            errors = ecdf(pvalues_consistent).(αs)
            @test all(((α, p),) -> p < α, zip(αs, errors))
        end
    end
end
