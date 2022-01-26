@testset "asymptotic.jl" begin
    @testset "estimate, statistic, and kernel matrix" begin
        kernel = (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel()
        biasedskce = BiasedSKCE(kernel)
        unbiasedskce = UnbiasedSKCE(kernel)

        for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
            # sample predictions and targets
            dist = Dirichlet(nclasses, 1)
            predictions = [rand(dist) for _ in 1:nsamples]
            targets_consistent = [
                rand(Categorical(prediction)) for prediction in predictions
            ]
            targets_onlyone = ones(Int, length(predictions))

            # compute calibration error estimate and test statistic
            for targets in (targets_consistent, targets_onlyone)
                estimate, statistic, kernelmatrix = CalibrationTests.estimate_statistic_kernelmatrix(
                    kernel, predictions, targets
                )

                @test estimate ≈ unbiasedskce(predictions, targets)
                @test statistic ≈
                      nsamples / (nsamples - 1) * unbiasedskce(predictions, targets) -
                      biasedskce(predictions, targets)
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
        kernel1 = ExponentialKernel() ∘ ScaleTransform(0.1)
        kernel2 = WhiteKernel()
        kernel = kernel1 ⊗ kernel2
        αs = 0.05:0.1:0.95
        nsamples = 100

        pvalues_consistent = Vector{Float64}(undef, 100)
        pvalues_onlyone = similar(pvalues_consistent)

        for nclasses in (2, 10)
            rng = StableRNG(1523)
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
                test_consistent = AsymptoticSKCETest(
                    kernel, predictions, targets_consistent
                )
                test_onlyone = AsymptoticSKCETest(kernel, predictions, targets_onlyone)

                # estimate pvalues
                pvalues_consistent[i] = pvalue(test_consistent; bootstrap_iters=500)
                pvalues_onlyone[i] = pvalue(test_onlyone; bootstrap_iters=500)
            end

            # compute empirical test errors
            ecdf_consistent = ecdf(pvalues_consistent)
            @test maximum(ecdf_consistent.(αs) .- αs) < 0.15
            @test maximum(pvalues_onlyone) < 0.01
        end
    end
end
