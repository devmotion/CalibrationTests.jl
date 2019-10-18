using CalibrationTests
using CalibrationErrors, Distributions
using Random, Test, Statistics

Random.seed!(1234)

# sample data
function generate_binary_data(nsamples)
    # generate predictions
    predictions = rand(Dirichlet(2, 1), nsamples)

    # generate labels
    labels_consistent = [rand() < predictions[1, i] ? 1 : 2 for i in 1:nsamples]
    labels_only_two = fill(2, nsamples)

    (predictions, labels_consistent), (predictions, labels_only_two)
end
const data_consistent, data_only_two = generate_binary_data(500)

# define matrix-valued kernel (using the mean total variation distance as bandwidth)
const kernel = UniformScalingKernel(ExponentialKernel(3))

@testset "Consistency test" begin
    println("consistency tests")

    # define estimators
    estimators = (ECE(UniformBinning(10)), BiasedSKCE(kernel),
                  LinearUnbiasedSKCE(kernel), QuadraticUnbiasedSKCE(kernel))

    for estimator in estimators
        test_consistent = ConsistencyTest(estimator, data_consistent)
        @test pvalue(test_consistent) > 0.7

        test_only_two = ConsistencyTest(estimator, data_only_two)
        @test pvalue(test_only_two) < 1e-6
    end
end

@testset "Distribution-free tests" begin
    println("distribution-free tests")

    # define estimators
    estimators = (BiasedSKCE(kernel), LinearUnbiasedSKCE(kernel),
                  QuadraticUnbiasedSKCE(kernel))

    for estimator in estimators
        test_consistent = DistributionFreeTest(estimator, data_consistent)
        @test pvalue(test_consistent) > 0.7

        test_only_two = ConsistencyTest(estimator, data_only_two)
        @test pvalue(test_only_two) < 1e-6
    end
end

@testset "Asymptotic linear test" begin
    println("asymptotic linear test")

    test_consistent = AsymptoticLinearTest(kernel, data_consistent)
    @test pvalue(test_consistent) > 0.7

    test_only_two = AsymptoticLinearTest(kernel, data_only_two)
    @test pvalue(test_only_two) < 1e-6
end

@testset "Asymptotic quadratic test" begin
    println("asymptotic quadratic test")

    test_consistent = AsymptoticQuadraticTest(kernel, data_consistent)
    @test pvalue(test_consistent) > 0.7

    test_only_two = AsymptoticQuadraticTest(kernel, data_only_two)
    @test pvalue(test_only_two) < 1e-6
end
