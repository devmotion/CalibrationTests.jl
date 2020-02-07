using CalibrationTests
using CalibrationErrors
using Distributions
using StatsBase
using StatsFuns

using Random
using Test

Random.seed!(1234)

@testset "bounds" begin
    # default bounds for scalar kernels
    CalibrationTests.uniformbound(ExponentialKernel()) == 1
    CalibrationTests.uniformbound(SquaredExponentialKernel()) == 1

    # default bounds for matrix-valued kernels
    λ = rand()
    CalibrationTests.uniformbound(UniformScalingKernel(λ, ExponentialKernel())) == λ
    d = rand(10)
    CalibrationTests.uniformbound(DiagonalKernel(d, SquaredExponentialKernel())) == maximum(d)

    # default bounds for kernel terms
    CalibrationTests.uniformbound(LinearUnbiasedSKCE(UniformScalingKernel(4, ExponentialKernel()))) == 8
end

@testset "estimator and estimates" begin
    kernel = UniformScalingKernel(ExponentialKernel(0.1))

    for skce in (BiasedSKCE(kernel), QuadraticUnbiasedSKCE(kernel), LinearUnbiasedSKCE(kernel))
        for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
            # sample predictions and targets
            dist = Dirichlet(nclasses, 1)
            predictions = [rand(dist) for _ in 1:nsamples]
            targets_consistent = [rand(Categorical(prediction)) for prediction in predictions]
            targets_onlyone = ones(Int, length(predictions))
        
            # for both sets of targets
            for targets in (targets_consistent, targets_onlyone)
                test = DistributionFreeTest(skce, predictions, targets)
            
                @test test.estimator == skce
                @test test.n == nsamples
                @test test.estimate ≈ calibrationerror(skce, predictions, targets)
                @test test.bound == CalibrationTests.uniformbound(skce)
            end
        end
    end
end

@testset "consistency" begin
    kernel = UniformScalingKernel(ExponentialKernel(0.1))
    αs = 0.05:0.1:0.95
    nsamples = 100

    pvalues_consistent = Vector{Float64}(undef, 100)
    
    for skce in (BiasedSKCE(kernel), QuadraticUnbiasedSKCE(kernel), LinearUnbiasedSKCE(kernel))
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
                test_consistent = DistributionFreeTest(skce, predictions, targets_consistent)

                # estimate pvalue
                pvalues_consistent[i] = pvalue(test_consistent)
            end

            # compute empirical test errors
            errors = ecdf(pvalues_consistent).(αs)
            @test all(((α, p),) -> p < α, zip(αs, errors))
        end
    end
end