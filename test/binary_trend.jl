using CalibrationTests
using CalibrationErrors, Distributions

using Random
using Statistics
using Test

Random.seed!(1234)

# sample data
function generate_binary_data(nsamples)
    # generate predictions
    dist = Dirichlet(2, 1)
    predictions = [rand(dist) for _ in 1:nsamples]

    # generate targets
    targets_consistent = [rand() < predictions[i][1] ? 1 : 2 for i in 1:nsamples]
    targets_onlytwo = fill(2, nsamples)

    return (predictions, targets_consistent), (predictions, targets_onlytwo)
end
const data_consistent, data_only_two = generate_binary_data(500)

# define tensor product kernel (using the mean total variation distance as bandwidth)
const kernel = transform(ExponentialKernel(), 3) ⊗ WhiteKernel()

@testset "Consistency test" begin
    # define estimators
    estimators = (
        BiasedSKCE(kernel),
        UnbiasedSKCE(kernel),
        (BlockUnbiasedSKCE(kernel, b) for b in (2, 10, 50, 100))...,
    )

    for estimator in estimators
        test_consistent = ConsistencyTest(estimator, data_consistent)
        @test pvalue(test_consistent) > 0.6
        print(test_consistent)

        test_only_two = ConsistencyTest(estimator, data_only_two)
        @test pvalue(test_only_two) < 1e-6
        print(test_only_two)
    end
end

@testset "Distribution-free tests" begin
    # define estimators
    estimators = (
        BiasedSKCE(kernel),
        UnbiasedSKCE(kernel),
        (BlockUnbiasedSKCE(kernel, b) for b in (2, 10, 50, 100))...,
    )

    for estimator in estimators
        test_consistent = DistributionFreeSKCETest(estimator, data_consistent)
        @test pvalue(test_consistent) > 0.7
        println(test_consistent)

        test_only_two = DistributionFreeSKCETest(estimator, data_only_two)
        if estimator isa Union{UnbiasedSKCE,BlockUnbiasedSKCE}
            @test pvalue(test_only_two) < 0.4
        else
            @test pvalue(test_only_two) < 1e-6
        end
        println(test_only_two)
    end
end

@testset "Asymptotic block SKCE test" begin
    for blocksize in (2, 10, 50, 100)
        test_consistent = AsymptoticBlockSKCETest(kernel, blocksize, data_consistent)
        @test pvalue(test_consistent) > 0.7
        println(test_consistent)

        test_only_two = AsymptoticBlockSKCETest(kernel, blocksize, data_only_two)
        @test pvalue(test_only_two) < 1e-6
        println(test_only_two)
    end
end

@testset "Asymptotic SKCE test" begin
    test_consistent = AsymptoticSKCETest(kernel, data_consistent)
    @test pvalue(test_consistent) > 0.7
    println(test_consistent)

    test_only_two = AsymptoticSKCETest(kernel, data_only_two)
    @test pvalue(test_only_two) < 1e-6
    println(test_only_two)
end

@testset "Asymptotic CME test" begin
    # define estimator (uniformly distributed test locations)
    testpredictions = [rand(Dirichlet(2, 1)) for _ in 1:5]
    testtargets = rand(1:2, 5)
    estimator = UCME(kernel, testpredictions, testtargets)

    test_consistent = AsymptoticCMETest(estimator, data_consistent)
    @test pvalue(test_consistent) > 0.8
    println(test_consistent)

    test_only_two = AsymptoticCMETest(estimator, data_only_two)
    @test pvalue(test_only_two) < 1e-6
    println(test_only_two)
end
