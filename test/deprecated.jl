@testset "deprecated.jl" begin
    kernel = SqExponentialKernel() ⊗ WhiteKernel()
    predictions = map(1:100) do _
        x = rand(5)
        x ./= sum(x)
        return x
    end
    targets = rand(1:5, 100)
    deprecated_data = (
        ((predictions, targets),),
        (reduce(hcat, predictions), targets),
        (map(tuple, predictions, targets),),
    )
    all_data = ((predictions, targets), deprecated_data...)

    @testset "ConsistencyTest" begin
        for estimator in (ECE(UniformBinning(10)), UnbiasedSKCE(kernel))
            for data in deprecated_data
                test = @test_deprecated ConsistencyTest(estimator, data...)
                @test test isa ConsistencyTest
                @test test.estimator === estimator
                @test test.predictions == predictions
                @test test.targets == targets
                @test test.estimate == estimator(predictions, targets)
            end
        end
    end

    @testset "DistributionFreeSKCETest" begin
        for estimator in (UnbiasedSKCE(kernel), BiasedSKCE(kernel))
            for data in deprecated_data
                test = @test_deprecated DistributionFreeSKCETest(estimator, data...)
                @test test isa DistributionFreeSKCETest
                @test test.estimator === estimator
                @test test.bound == CalibrationTests.uniformbound(estimator)
                @test test.n === length(predictions)
                @test test.estimate == estimator(predictions, targets)

                bound = rand()
                test = @test_deprecated DistributionFreeSKCETest(
                    estimator, data...; bound=bound
                )
                @test test isa DistributionFreeSKCETest
                @test test.estimator === estimator
                @test test.bound == bound
                @test test.n === length(predictions)
                @test test.estimate == estimator(predictions, targets)
            end
        end
    end

    @testset "AsymptoticSKCETest" begin
        skce = UnbiasedSKCE(kernel)
        for arg in (kernel, skce)
            for data in deprecated_data
                test = @test_deprecated AsymptoticSKCETest(arg, data...)
                @test test isa AsymptoticSKCETest
                @test test.kernel === kernel
                @test test.estimate ≈ skce(predictions, targets)
            end
        end

        test = @test_deprecated AsymptoticSKCETest(skce, predictions, targets)
        @test test isa AsymptoticSKCETest
        @test test.kernel === kernel
        @test test.estimate ≈ skce(predictions, targets)
    end

    @testset "AsymptoticBlockSKCETest" begin
        blocksize = 10
        skce = BlockUnbiasedSKCE(kernel, blocksize)
        for args in ((kernel, blocksize), (skce,))
            for data in deprecated_data
                test = @test_deprecated AsymptoticBlockSKCETest(args..., data...)
                @test test isa AsymptoticBlockSKCETest
                @test test.kernel === kernel
                @test test.blocksize == blocksize
                @test test.nblocks == length(predictions) ÷ blocksize
                @test test.estimate ≈ skce(predictions, targets)
            end
        end

        test = @test_deprecated AsymptoticBlockSKCETest(skce, predictions, targets)
        @test test isa AsymptoticBlockSKCETest
        @test test.kernel === kernel
        @test test.blocksize == blocksize
        @test test.nblocks == length(predictions) ÷ blocksize
        @test test.estimate ≈ skce(predictions, targets)
    end

    @testset "AsymptoticCMETest" begin
        nsamples = length(predictions)
        ntestsamples = 10
        testpredictions = map(1:ntestsamples) do _
            x = rand(5)
            x ./= sum(x)
            return x
        end
        testtargets = rand(1:5, ntestsamples)
        ucme = UCME(kernel, testpredictions, testtargets)

        for data in deprecated_data
            test = @test_deprecated AsymptoticCMETest(ucme, data...)
            @test test isa AsymptoticCMETest
            @test test.kernel === kernel
            @test test.nsamples == nsamples
            @test test.ntestsamples == ntestsamples
            @test test.estimate == ucme(predictions, targets)
        end
    end
end
