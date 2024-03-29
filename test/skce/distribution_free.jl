@testset "distribution_free.jl" begin
    @testset "bounds" begin
        # default bounds for base kernels
        CalibrationTests.uniformbound(ExponentialKernel()) == 1
        CalibrationTests.uniformbound(SqExponentialKernel()) == 1
        CalibrationTests.uniformbound(WhiteKernel()) == 1

        # default bounds for kernels with input transformations
        CalibrationTests.uniformbound(SqExponentialKernel() ∘ ScaleTransform(rand())) == 1
        CalibrationTests.uniformbound(ExponentialKernel() ∘ ScaleTransform(rand(10))) == 1

        # default bounds for scaled kernels
        CalibrationTests.uniformbound(42 * ExponentialKernel()) == 42

        # default bounds for tensor product kernels
        kernel = (3.2 * SqExponentialKernel()) ⊗ (2.7 * WhiteKernel())
        CalibrationTests.uniformbound(kernel) == 3.2 * 2.7

        # default bounds for kernel terms
        CalibrationTests.uniformbound(SKCE(kernel; blocksize=2)) == 2 * 3.2 * 2.7
    end

    @testset "estimator and estimates" begin
        kernel = (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel()

        for skce in (SKCE(kernel), SKCE(kernel; unbiased=false), SKCE(kernel; blocksize=2))
            for nclasses in (2, 10, 100), nsamples in (10, 50, 100)
                # sample predictions and targets
                dist = Dirichlet(nclasses, 1)
                predictions = [rand(dist) for _ in 1:nsamples]
                targets_consistent = [
                    rand(Categorical(prediction)) for prediction in predictions
                ]
                targets_onlyone = ones(Int, length(predictions))

                # for both sets of targets
                for targets in (targets_consistent, targets_onlyone)
                    test = DistributionFreeSKCETest(skce, predictions, targets)

                    @test test.estimator == skce
                    @test test.n == nsamples
                    @test test.estimate ≈ skce(predictions, targets)
                    @test test.bound == CalibrationTests.uniformbound(skce)
                end
            end
        end
    end

    @testset "consistency" begin
        kernel = (ExponentialKernel() ∘ ScaleTransform(0.1)) ⊗ WhiteKernel()
        αs = 0.05:0.1:0.95
        nsamples = 100

        pvalues_consistent = Vector{Float64}(undef, 100)

        for skce in (SKCE(kernel), SKCE(kernel; unbiased=false), SKCE(kernel; blocksize=2))
            for nclasses in (2, 10)
                rng = StableRNG(5921)
                dist = Dirichlet(nclasses, 1)
                predictions = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
                targets_consistent = Vector{Int}(undef, nsamples)

                for i in eachindex(pvalues_consistent)
                    # sample predictions and targets
                    for j in 1:nsamples
                        rand!(rng, dist, predictions[j])
                        targets_consistent[j] = rand(rng, Categorical(predictions[j]))
                    end

                    # define test
                    test_consistent = DistributionFreeSKCETest(
                        skce, predictions, targets_consistent
                    )

                    # estimate pvalue
                    pvalues_consistent[i] = pvalue(test_consistent)
                end

                # compute empirical test errors
                @test all(ecdf(pvalues_consistent).(αs) .< αs)
            end
        end
    end
end
