using CalibrationTests
using CalibrationErrors
using Distributions

using Random
using Statistics
using Test

resample_direct(rng, test) =
    CalibrationTests.consistency_resampling_ccdf_direct(rng, test, 1_000)
resample_alias(rng, test) =
    CalibrationTests.consistency_resampling_ccdf_alias(rng, test, 1_000)

@testset "ECE" begin
    ce = ECE(UniformBinning(10))
    N = 1_000

    for nclasses in (2, 5, 10)
        println("Consistency test with ECE ($nclasses classes)")

        # sample predictions and targets
        Random.seed!(1234)
        dist = Dirichlet(nclasses, 1)
        predictions = [rand(dist) for _ in 1:10]
        targets_consistent = [rand(Categorical(prediction)) for prediction in predictions]
        targets_onlyone = ones(Int, length(predictions))

        # define consistency resampling tests
        test_consistent = ConsistencyTest(ce, predictions, targets_consistent)
        test_onlyone = ConsistencyTest(ce, predictions, targets_onlyone)

        # compute pvalues with both resampling methods
        Random.seed!(1234)
        pvalues = [pvalue(test_consistent; rng = Random.GLOBAL_RNG) for _ in 1:N]

        Random.seed!(1111)
        pvalues_direct = [resample_direct(Random.GLOBAL_RNG, test_consistent) for _ in 1:N]

        Random.seed!(5678)
        pvalues_alias = [resample_alias(Random.GLOBAL_RNG, test_consistent) for _ in 1:N]

        if nclasses == 2
            @test mean(pvalues) ≈ 0.35 atol = 1e-2
        elseif nclasses == 5
            @test mean(pvalues) ≈ 0.02 atol = 1e-2
        elseif nclasses == 10
            @test mean(pvalues) ≈ 0.02 atol = 1e-2
        end
        @test mean(pvalues_direct) ≈ mean(pvalues) atol = 1e-2
        @test mean(pvalues_alias) ≈ mean(pvalues) atol = 1e-2

        Random.seed!(1234)
        pvalues = [pvalue(test_onlyone; rng = Random.GLOBAL_RNG) for _ in 1:N]

        Random.seed!(1111)
        pvalues_direct = [resample_direct(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

        Random.seed!(5678)
        pvalues_alias = [resample_alias(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

        @test mean(pvalues) < 1e-3
        @test mean(pvalues_direct) ≈ mean(pvalues) atol = 1e-3
        @test mean(pvalues_alias) ≈ mean(pvalues) atol = 1e-3
    end
end

@testset "Linear SKCE" begin
    ce = LinearUnbiasedSKCE(transform(ExponentialKernel(), 0.1), WhiteKernel())
    N = 1_000

    for nclasses in (2, 5, 10)
        println("Consistency test with the linear SKCE estimator ($nclasses classes)")

        # sample predictions and targets
        Random.seed!(1234)
        dist = Dirichlet(nclasses, 1)
        predictions = [rand(dist) for _ in 1:10]
        targets_consistent = [rand(Categorical(prediction)) for prediction in predictions]
        targets_onlyone = ones(Int, length(predictions))

        # define consistency resampling tests
        test_consistent = ConsistencyTest(ce, predictions, targets_consistent)
        test_onlyone = ConsistencyTest(ce, predictions, targets_onlyone)

        # compute pvalues with both resampling methods
        Random.seed!(1234)
        pvalues = [pvalue(test_consistent; rng = Random.GLOBAL_RNG) for _ in 1:N]

        Random.seed!(1111)
        pvalues_direct = [resample_direct(Random.GLOBAL_RNG, test_consistent) for _ in 1:N]

        Random.seed!(5678)
        pvalues_alias = [resample_alias(Random.GLOBAL_RNG, test_consistent) for _ in 1:N]

        if nclasses == 2
            @test mean(pvalues) ≈ 0.12 atol = 1e-2
        elseif nclasses == 5
            @test mean(pvalues) ≈ 0.62 atol = 1e-2
        elseif nclasses == 10
            @test mean(pvalues) ≈ 0.44 atol = 1e-2
        end
        @test mean(pvalues_direct) ≈ mean(pvalues) atol = 1e-2
        @test mean(pvalues_alias) ≈ mean(pvalues) atol = 1e-2

        Random.seed!(1234)
        pvalues = [pvalue(test_onlyone; rng = Random.GLOBAL_RNG) for _ in 1:N]

        Random.seed!(1111)
        pvalues_direct = [resample_direct(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

        Random.seed!(5678)
        pvalues_alias = [resample_alias(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

        @test mean(pvalues) < 1e-2
        @test mean(pvalues_direct) ≈ mean(pvalues) atol = 1e-2
        @test mean(pvalues_alias) ≈ mean(pvalues) atol = 1e-2
    end
end
