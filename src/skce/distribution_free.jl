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
# by default consider we assume `p = q`
uniformbound(kce::SKCE) = 2 * uniformbound(CalibrationErrors.kernel(kce))

# uniform bounds of the norm of scalar-valued kernels
uniformbound(kernel::CalibrationErrors.ExponentialKernel) = exp(zero(kernel.γ))
uniformbound(kernel::CalibrationErrors.SquaredExponentialKernel) = exp(zero(kernel.γ))

# uniform bounds `K_{p;q}` of the norm of matrix-valued kernels for `p = q`
uniformbound(kernel::CalibrationErrors.UniformScalingKernel) =
    kernel.λ * uniformbound(kernel.kernel)
uniformbound(kernel::CalibrationErrors.DiagonalKernel) =
    maximum(kernel.diag) * uniformbound(kernel.kernel)