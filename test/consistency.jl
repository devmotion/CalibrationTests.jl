@testset "consistency.jl" begin
    @testset "ECE" begin
        ce = ECE(UniformBinning(10))
        N = 1_000

        for nclasses in (2, 5, 10)
            println("Consistency test with ECE ($nclasses classes)")

            # sample predictions and targets
            rng = StableRNG(3881)
            dist = Dirichlet(nclasses, 1)
            predictions = [rand(rng, dist) for _ in 1:10]
            targets_consistent = [
                rand(rng, Categorical(prediction)) for prediction in predictions
            ]
            targets_onlyone = ones(Int, length(predictions))

            # define consistency resampling tests
            test_consistent = ConsistencyTest(ce, predictions, targets_consistent)
            test_onlyone = ConsistencyTest(ce, predictions, targets_onlyone)

            # compute pvalues with both resampling methods
            pvalues = [pvalue(test_consistent) for _ in 1:N]

            if nclasses == 2
                @test mean(pvalues) ≈ 0.02 atol = 1e-2
            elseif nclasses == 5
                @test mean(pvalues) ≈ 0.03 atol = 1e-2
            elseif nclasses == 10
                @test mean(pvalues) ≈ 0.03 atol = 1e-2
            end

            pvalues = [pvalue(test_onlyone) for _ in 1:N]
            @test mean(pvalues) < 1e-3
        end
    end

    @testset "Block SKCE" begin
        nsamples = 10
        N = 1_000

        for blocksize in (2, 5)
            ce = SKCE(
                (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel();
                blocksize=blocksize,
            )

            for nclasses in (2, 5, 10)
                println(
                    "Consistency test with the linear SKCE estimator ($nclasses classes)"
                )

                # sample predictions and targets
                rng = StableRNG(8339)
                dist = Dirichlet(nclasses, 1)
                predictions = [rand(rng, dist) for _ in 1:10]
                targets_consistent = [
                    rand(rng, Categorical(prediction)) for prediction in predictions
                ]
                targets_onlyone = ones(Int, length(predictions))

                # define consistency resampling tests
                test_consistent = ConsistencyTest(ce, predictions, targets_consistent)
                test_onlyone = ConsistencyTest(ce, predictions, targets_onlyone)

                # compute pvalues with both resampling methods
                pvalues = [pvalue(test_consistent) for _ in 1:N]

                if blocksize == 2
                    if nclasses == 2
                        @test mean(pvalues) ≈ 0.18 atol = 1e-2
                    elseif nclasses == 5
                        @test mean(pvalues) ≈ 0.79 atol = 1e-2
                    elseif nclasses == 10
                        @test mean(pvalues) ≈ 0.32 atol = 1e-2
                    end
                else
                    if nclasses == 2
                        @test mean(pvalues) ≈ 0.41 atol = 1e-2
                    elseif nclasses == 5
                        @test mean(pvalues) ≈ 0.35 atol = 1e-2
                    elseif nclasses == 10
                        @test mean(pvalues) ≈ 0.11 atol = 1e-2
                    end
                end

                pvalues = [pvalue(test_onlyone) for _ in 1:N]
                @test mean(pvalues) < 1e-2
            end
        end
    end
end
