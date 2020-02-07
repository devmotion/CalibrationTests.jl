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

    (predictions, targets_consistent), (predictions, targets_onlytwo)
end
const data_consistent, data_only_two = generate_binary_data(500)

# define matrix-valued kernel (using the mean total variation distance as bandwidth)
const kernel = UniformScalingKernel(ExponentialKernel(3))

@testset "Consistency test" begin
    # define estimators
    estimators = (ECE(UniformBinning(10)), BiasedSKCE(kernel),
                  LinearUnbiasedSKCE(kernel), QuadraticUnbiasedSKCE(kernel))

    for estimator in estimators
        test_consistent = ConsistencyTest(estimator, data_consistent)
        @test pvalue(test_consistent) > 0.7
        print(test_consistent)

        test_only_two = ConsistencyTest(estimator, data_only_two)
        @test pvalue(test_only_two) < 1e-6
        print(test_only_two)
    end
end

@testset "Distribution-free tests" begin
    # define estimators
    estimators = (BiasedSKCE(kernel), LinearUnbiasedSKCE(kernel),
                  QuadraticUnbiasedSKCE(kernel))

    for estimator in estimators
        test_consistent = DistributionFreeTest(estimator, data_consistent)
        @test pvalue(test_consistent) > 0.7
        println(test_consistent)

        test_only_two = DistributionFreeTest(estimator, data_only_two)
        if estimator isa Union{LinearUnbiasedSKCE,QuadraticUnbiasedSKCE}
            @test pvalue(test_only_two) < 0.4
        else
            @test pvalue(test_only_two) < 1e-6
        end
        println(test_only_two)
    end
end

@testset "Asymptotic linear test" begin
    test_consistent = AsymptoticLinearTest(kernel, data_consistent)
    @test pvalue(test_consistent) > 0.7
    println(test_consistent)

    test_only_two = AsymptoticLinearTest(kernel, data_only_two)
    @test pvalue(test_only_two) < 1e-6
    println(test_only_two)
end

@testset "Asymptotic quadratic test" begin
    test_consistent = AsymptoticQuadraticTest(kernel, data_consistent)
    @test pvalue(test_consistent) > 0.7
    println(test_consistent)

    test_only_two = AsymptoticQuadraticTest(kernel, data_only_two)
    @test pvalue(test_only_two) < 1e-6
    println(test_only_two)
end
