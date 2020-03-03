struct DistributionFreeTest{E<:SKCE,B,V} <: HypothesisTests.HypothesisTest
    """Calibration estimator."""
    estimator::E
    """Uniform upper bound of the terms of the estimator."""
    bound::B
    """Number of observations."""
    n::Int
    """Calibration error estimate."""
    estimate::V
end

function DistributionFreeTest(estimator::SKCE, data...; bound = uniformbound(estimator))
    # obtain the predictions and targets
    predictions, targets = CalibrationErrors.predictions_targets(data...)

    # compute the calibration error estimate
    estimate = calibrationerror(estimator, predictions, targets)

    DistributionFreeTest(estimator, bound, length(predictions), estimate)
end

# HypothesisTests interface

HypothesisTests.default_tail(::DistributionFreeTest) = :right

function HypothesisTests.pvalue(test::DistributionFreeTest{<:BiasedSKCE})
    @unpack bound, n, estimate = test

    s = sqrt(n * estimate / bound) - 1

    # if estimate is below threshold B₂
    s < zero(s) && (s = zero(s))

    exp(- s^2 / 2)
end

function HypothesisTests.pvalue(test::DistributionFreeTest{<:Union{QuadraticUnbiasedSKCE,
                                                                   LinearUnbiasedSKCE}})
    @unpack bound, n, estimate = test

    exp(- div(n, 2) * estimate^2 / (2 * bound ^ 2))
end

HypothesisTests.testname(::DistributionFreeTest) = "Distribution-free calibration test"

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::DistributionFreeTest)
    nameof(typeof(test.estimator)), zero(test.estimate), test.estimate
end

function HypothesisTests.show_params(io::IO, test::DistributionFreeTest, ident = "")
    println(io, ident, "number of observations: $(test.n)")
    println(io, ident, "uniform bound of the terms of the estimator: $(test.bound)")
end

# uniform bound `B_{p;q}` of the absolute value of the terms in the estimators
uniformbound(kce::SKCE) = 2 * uniformbound(kce.kernel)

# uniform bounds of the norm of base kernels
uniformbound(kernel::ExponentialKernel) = 1
uniformbound(kernel::SqExponentialKernel) = 1
uniformbound(kernel::TVExponentialKernel) = 1
uniformbound(kernel::WhiteKernel) = 1

# uniform bound of the norm of a scaled kernel
uniformbound(kernel::ScaledKernel) = first(kernel.σ) * uniformbound(kernel.kernel)

# uniform bound of the norm of a kernel with input transformations
# assume transform is bijective (i.e., transform does not affect the bound) as default
uniformbound(kernel::TransformedKernel) = uniformbound(kernel.kernel)

# uniform bounds of the norm of tensor product kernels
uniformbound(kernel::TensorProductKernel) =
    uniformbound(kernel.kernel1) * uniformbound(kernel.kernel2)
