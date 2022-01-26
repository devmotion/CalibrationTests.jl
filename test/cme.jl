@testset "cme.jl" begin
    @testset "estimate and statistic" begin
        kernel = (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel()

        function deviation(testprediction, testtarget, prediction, target)
            testlocation = (testprediction, testtarget)
            return mapreduce(+, prediction, 1:length(prediction)) do p, t
                ((t == target) - p) * kernel(testlocation, (prediction, t))
            end
        end

        for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
            # define estimator (sample test locations uniformly)
            dist = Dirichlet(nclasses, 1)
            testpredictions = [rand(dist) for _ in 1:(nsamples ÷ 10)]
            testtargets = rand(1:nclasses, nsamples ÷ 10)
            estimator = UCME(kernel, testpredictions, testtargets)

            # sample predictions and targets
            predictions = [rand(dist) for _ in 1:nsamples]
            targets_consistent = [
                rand(Categorical(prediction)) for prediction in predictions
            ]
            targets_onlyone = ones(Int, length(predictions))

            # compute calibration error estimate and test statistic
            for targets in (targets_consistent, targets_onlyone)
                test = AsymptoticCMETest(estimator, predictions, targets)

                @test test.kernel == kernel
                @test test.nsamples == nsamples
                @test test.ntestsamples == nsamples ÷ 10
                @test test.estimate ≈ estimator(predictions, targets)

                deviations =
                    deviation.(testpredictions', testtargets', predictions, targets)
                mean_deviations = vec(mean(deviations; dims=1))
                @test test.mean_deviations ≈ mean_deviations

                # use of `inv` can lead to slightly different results
                S = cov(deviations; dims=1)
                statistic = nsamples * mean_deviations' * inv(S) * mean_deviations
                @test test.statistic ≈ statistic rtol = 1e-4
            end
        end
    end

    @testset "consistency" begin
        kernel = (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel()
        αs = 0.05:0.1:0.95
        nsamples = 100

        pvalues_consistent = Vector{Float64}(undef, 100)
        pvalues_onlyone = similar(pvalues_consistent)

        for nclasses in (2, 10)
            # create estimator (sample test locations uniformly)
            rng = StableRNG(7434)
            dist = Dirichlet(nclasses, 1)
            testpredictions = [rand(rng, dist) for _ in 1:5]
            testtargets = rand(rng, 1:nclasses, 5)
            estimator = UCME(kernel, testpredictions, testtargets)

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
                test_consistent = AsymptoticCMETest(
                    estimator, predictions, targets_consistent
                )
                test_onlyone = AsymptoticCMETest(estimator, predictions, targets_onlyone)

                # estimate pvalues
                pvalues_consistent[i] = pvalue(test_consistent)
                pvalues_onlyone[i] = pvalue(test_onlyone)
            end

            # compute empirical test errors
            ecdf_consistent = ecdf(pvalues_consistent)
            @test all(ecdf_consistent(α) < α + 0.15 for α in αs)
            @test all(p < 0.05 for p in pvalues_onlyone)
        end
    end
end
