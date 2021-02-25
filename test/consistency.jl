using CalibrationTests
using CalibrationErrors
using Distributions

using Random
using Statistics
using Test

function resample_direct(rng, test)
    return CalibrationTests.consistency_resampling_ccdf_direct(rng, test, 1_000)
end
function resample_alias(rng, test)
    return CalibrationTests.consistency_resampling_ccdf_alias(rng, test, 1_000)
end

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
        pvalues = [pvalue(Random.GLOBAL_RNG, test_consistent) for _ in 1:N]

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
        pvalues = [pvalue(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

        Random.seed!(1111)
        pvalues_direct = [resample_direct(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

        Random.seed!(5678)
        pvalues_alias = [resample_alias(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

        @test mean(pvalues) < 1e-3
        @test mean(pvalues_direct) ≈ mean(pvalues) atol = 1e-3
        @test mean(pvalues_alias) ≈ mean(pvalues) atol = 1e-3
    end
end

@testset "Block SKCE" begin
    nsamples = 10
    N = 1_000

    for blocksize in (2, 5)
        ce = BlockUnbiasedSKCE(
            transform(ExponentialKernel(), 0.1) ⊗ WhiteKernel(), blocksize
        )

        for nclasses in (2, 5, 10)
            println("Consistency test with the linear SKCE estimator ($nclasses classes)")

            # sample predictions and targets
            Random.seed!(1234)
            dist = Dirichlet(nclasses, 1)
            predictions = [rand(dist) for _ in 1:10]
            targets_consistent = [
                rand(Categorical(prediction)) for prediction in predictions
            ]
            targets_onlyone = ones(Int, length(predictions))

            # define consistency resampling tests
            test_consistent = ConsistencyTest(ce, predictions, targets_consistent)
            test_onlyone = ConsistencyTest(ce, predictions, targets_onlyone)

            # compute pvalues with both resampling methods
            Random.seed!(1234)
            pvalues = [pvalue(Random.GLOBAL_RNG, test_consistent) for _ in 1:N]

            Random.seed!(1111)
            pvalues_direct = [
                resample_direct(Random.GLOBAL_RNG, test_consistent) for _ in 1:N
            ]

            Random.seed!(5678)
            pvalues_alias = [resample_alias(Random.GLOBAL_RNG, test_consistent) for _ in 1:N
]

            if blocksize == 2
                if nclasses == 2
                    @test mean(pvalues) ≈ 0.12 atol = 1e-2
                elseif nclasses == 5
                    @test mean(pvalues) ≈ 0.62 atol = 1e-2
                elseif nclasses == 10
                    @test mean(pvalues) ≈ 0.44 atol = 1e-2
                end
            else
                if nclasses == 2
                    @test mean(pvalues) ≈ 0.01 atol = 1e-2
                elseif nclasses == 5
                    @test mean(pvalues) ≈ 0.63 atol = 1e-2
                elseif nclasses == 10
                    @test mean(pvalues) ≈ 0.21 atol = 1e-2
                end
            end

            @test mean(pvalues_direct) ≈ mean(pvalues) atol = 1e-2
            @test mean(pvalues_alias) ≈ mean(pvalues) atol = 1e-2

            Random.seed!(1234)
            pvalues = [pvalue(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

            Random.seed!(1111)
            pvalues_direct = [resample_direct(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

            Random.seed!(5678)
            pvalues_alias = [resample_alias(Random.GLOBAL_RNG, test_onlyone) for _ in 1:N]

            @test mean(pvalues) < 1e-2
            @test mean(pvalues_direct) ≈ mean(pvalues) atol = 1e-2
            @test mean(pvalues_alias) ≈ mean(pvalues) atol = 1e-2
        end
    end
end
