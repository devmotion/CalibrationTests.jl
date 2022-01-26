@testset "asymptotic_block.jl" begin
    @testset "estimate, stderr, and z" begin
        kernel = (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel()
        for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
            for blocksize in (2, 5, 10, 50)
                # blocksize may no be greater than number of samples
                blocksize < nsamples || continue

                # sample predictions and targets
                dist = Dirichlet(nclasses, 1)
                predictions = [rand(dist) for _ in 1:nsamples]
                targets_consistent = [
                    rand(Categorical(prediction)) for prediction in predictions
                ]
                targets_onlyone = ones(Int, length(predictions))

                skce = BlockUnbiasedSKCE(kernel, blocksize)

                # for both sets of targets
                for targets in (targets_consistent, targets_onlyone)
                    test = AsymptoticBlockSKCETest(kernel, blocksize, predictions, targets)

                    @test test.blocksize == blocksize
                    @test test.nblocks == nsamples ÷ blocksize
                    @test test.estimate ≈ skce(predictions, targets)
                    @test test.z == test.estimate / test.stderr

                    @test pvalue(test) ==
                          pvalue(test; tail=:right) ==
                          ccdf(Normal(), test.z)
                    @test_throws ArgumentError pvalue(test; tail=:left)
                    @test_throws ArgumentError pvalue(test; tail=:both)

                    for α in 0.55:0.05:0.95
                        q = quantile(Normal(), α)
                        @test confint(test; level=α) ==
                              confint(test; level=α, tail=:right) ==
                              (max(0, test.estimate - q * test.stderr), Inf)
                        @test_throws ArgumentError confint(test; level=α, tail=:left)
                        @test_throws ArgumentError confint(test; level=α, tail=:both)
                    end
                end
            end
        end
    end

    @testset "consistency" begin
        kernel1 = ExponentialKernel() ∘ ScaleTransform(0.1)
        kernel2 = WhiteKernel()
        kernel = kernel1 ⊗ kernel2
        αs = 0.05:0.1:0.95
        nsamples = 100

        pvalues_consistent = Vector{Float64}(undef, 100)
        pvalues_onlyone = similar(pvalues_consistent)

        for blocksize in (2, 5, 10)
            for nclasses in (2, 10)
                rng = StableRNG(6144)
                dist = Dirichlet(nclasses, 1)
                predictions = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
                targets_consistent = Vector{Int}(undef, nsamples)
                targets_onlyone = ones(Int, nsamples)

                for i in eachindex(pvalues_consistent)
                    # sample predictions and targets
                    for j in 1:nsamples
                        rand!(rng, dist, predictions[j])
                        targets_consistent[j] = rand(rng, Categorical(predictions[j]))
                    end

                    # define test
                    test_consistent = AsymptoticBlockSKCETest(
                        kernel, blocksize, predictions, targets_consistent
                    )
                    test_onlyone = AsymptoticBlockSKCETest(
                        kernel, blocksize, predictions, targets_onlyone
                    )

                    # estimate pvalues
                    pvalues_consistent[i] = pvalue(test_consistent)
                    pvalues_onlyone[i] = pvalue(test_onlyone)
                end

                # compute empirical test errors
                ecdf_consistent = ecdf(pvalues_consistent)
                @test maximum(ecdf_consistent.(αs) .- αs) < 0.1
                @test maximum(pvalues_onlyone) < 0.01
            end
        end
    end
end
