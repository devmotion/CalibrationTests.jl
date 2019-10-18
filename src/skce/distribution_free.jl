struct DistributionFreeTest{E<:SKCE,B,V} <: CalibrationTest
    """Calibration estimator."""
    estimator::E
    """Upper bound of estimator."""
    bound::B
    """Number of data points."""
    n::Int
    """SKCE estimate."""
    estimate::V
end

function DistributionFreeTest(estimator::SKCE,
                              data::Tuple{<:AbstractMatrix{<:Real},<:AbstractVector{<:Integer}};
                              bound = maximum(estimator))
    # check if the number of predictions and labels is equal
    predictions, labels = CalibrationErrors.get_predictions_labels(data)

    DistributionFreeTest(estimator, bound, size(predictions, 2),
                         calibrationerror(estimator, data))
end

HypothesisTests.testname(::DistributionFreeTest) = "Distribution-free calibration test"

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
