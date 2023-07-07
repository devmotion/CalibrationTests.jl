struct ConsistencyTest{E<:CalibrationErrorEstimator,P,T,V} <: HypothesisTests.HypothesisTest
    """Calibration estimator."""
    estimator::E
    """Predictions."""
    predictions::P
    """Targets."""
    targets::T
    """Calibration estimate."""
    estimate::V
end

"""
    ConsistencyTest(
        estimator::CalibrationErrorEstimator,
        predictions::AbstractVector,
        targets::AbstractVector,
    )

Construct an hypothesis test of calibration based on consistency resampling with a
calibration `estimator` as test statistic and `predictions` and `targets` of a model of
interest.

Consistency resampling is a parametric bootstrap method for calibrated models.
    
## References

Bröcker, J., & Smith, L. A. (2007).
[Increasing the reliability of reliability diagrams](https://doi.org/10.1175/WAF993.1).
Weather and forecasting, 22(3), 651-661.
"""
function ConsistencyTest(
    estimator::CalibrationErrorEstimator,
    predictions::AbstractVector,
    targets::AbstractVector,
)
    estimate = estimator(predictions, targets)
    return ConsistencyTest(estimator, predictions, targets, estimate)
end

# HypothesisTests interface

HypothesisTests.default_tail(::ConsistencyTest) = :right

HypothesisTests.testname(::ConsistencyTest) = "Consistency resampling test"

function HypothesisTests.pvalue(test::ConsistencyTest; kwargs...)
    return pvalue(Random.GLOBAL_RNG, test; kwargs...)
end
function HypothesisTests.pvalue(
    rng::Random.AbstractRNG, test::ConsistencyTest; bootstrap_iters::Int=1_000
)
    bootstrap_iters > 0 || error("number of bootstrap samples must be positive")

    predictions = test.predictions
    sampledpredictions = similar(predictions)
    sampledtargets = similar(test.targets)
    samples = StructArrays.StructArray((sampledpredictions, sampledtargets))

    estimate = test.estimate
    estimator = test.estimator
    sampler = Random.Sampler(rng, ConsistencyResampling.Consistent(predictions))
    n = 0
    for _ in 1:bootstrap_iters
        # perform consistency resampling
        Random.rand!(rng, samples, sampler)

        # evaluate the calibration error
        sampledestimate = estimator(sampledpredictions, sampledtargets)

        # check if the estimate for the resampled data is ≥ the original estimate
        if sampledestimate ≥ estimate
            n += 1
        end
    end

    return n / bootstrap_iters
end

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::ConsistencyTest)
    return nameof(typeof(test.estimator)), zero(test.estimate), test.estimate
end

HypothesisTests.show_params(io::IO, test::ConsistencyTest, ident="") = nothing
