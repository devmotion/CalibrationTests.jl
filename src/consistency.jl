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

function ConsistencyTest(estimator::CalibrationErrorEstimator, data...)
    # obtain the predictions and targets
    predictions, targets = CalibrationErrors.predictions_targets(data...)

    # compute the calibration error estimate
    estimate = calibrationerror(estimator, predictions, targets)

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
    return consistency_resampling_ccdf(rng, test, bootstrap_iters)
end

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::ConsistencyTest)
    return nameof(typeof(test.estimator)), zero(test.estimate), test.estimate
end

HypothesisTests.show_params(io::IO, test::ConsistencyTest, ident="") = nothing

# estimate ccdf using bootstrapping - move upstream?

function consistency_resampling_ccdf(
    rng::AbstractRNG, test::ConsistencyTest, bootstrap_iters::Int
)
    @unpack predictions = test
    bootstrap_iters > 0 || error("number of bootstrap samples must be positive")

    # use same heuristic as StatsBase to decide whether to sample predictions
    # directly or to build an alias table
    # TODO: needs proper benchmarking
    nsamples = length(predictions)
    if nsamples < 40
        p = consistency_resampling_ccdf_direct(rng, test, bootstrap_iters)
    else
        t = nsamples < 500 ? 64 : 32
        if length(predictions[1]) < t
            p = consistency_resampling_ccdf_direct(rng, test, bootstrap_iters)
        else
            p = consistency_resampling_ccdf_alias(rng, test, bootstrap_iters)
        end
    end

    return p
end

# sample targets directly (without alias table)
function consistency_resampling_ccdf_direct(
    rng::AbstractRNG, test::ConsistencyTest, bootstrap_iters::Int
)
    @unpack estimator, predictions, targets, estimate = test

    # create caches
    nsamples = length(predictions)
    resampledpredictions = similar(predictions)
    resampledtargets = similar(targets)
    resampledidxs = Vector{Int}(undef, nsamples)

    # for each resampling step
    n = 0
    sampler = Random.Sampler(rng, 1:nsamples)
    @inbounds for _ in 1:bootstrap_iters
        # resample data
        rand!(rng, resampledidxs, sampler)
        for j in 1:nsamples
            # resample predictions
            idx = resampledidxs[j]
            resampledpredictions[j] = prediction = predictions[idx]

            # resample targets
            p = rand(rng)
            cw = prediction[1]
            target = 1
            while cw < p && target < length(prediction)
                target += 1
                cw += prediction[target]
            end
            resampledtargets[j] = target
        end

        # evaluate the calibration error
        resampledestimate = calibrationerror(
            estimator, resampledpredictions, resampledtargets
        )

        # check if the estimate for the resampled data is ≥ the original estimate
        if resampledestimate ≥ estimate
            n += 1
        end
    end

    return n / bootstrap_iters
end

# sample targets with an alias table
function consistency_resampling_ccdf_alias(
    rng::AbstractRNG, test::ConsistencyTest, bootstrap_iters::Int
)
    @unpack estimator, predictions, targets, estimate = test

    # create alias table
    nsamples = length(predictions)
    nclasses = length(predictions[1])
    accept = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
    alias = [Vector{Int}(undef, nclasses) for _ in 1:nsamples]
    @inbounds for i in 1:nsamples
        StatsBase.make_alias_table!(predictions[i], 1.0, accept[i], alias[i])
    end

    # create caches
    resampledpredictions = similar(predictions)
    resampledtargets = similar(targets)
    resampledidxs = Vector{Int}(undef, nsamples)

    # for each resampling step
    n = 0
    sampler_predictions = Random.Sampler(rng, 1:nsamples)
    sampler_targets = Random.Sampler(rng, 1:nclasses)
    @inbounds for _ in 1:bootstrap_iters
        # resample data
        rand!(rng, resampledidxs, sampler_predictions)
        for j in 1:nsamples
            # resample predictions
            idx = resampledidxs[j]
            resampledpredictions[j] = predictions[idx]

            # resample targets
            target = rand(rng, sampler_targets)
            resampledtargets[j] =
                rand(rng) < accept[idx][target] ? target : alias[idx][target]
        end

        # evaluate the calibration error
        resampledestimate = calibrationerror(
            estimator, resampledpredictions, resampledtargets
        )

        # check if the estimate for the resampled data is ≥ the original estimate
        if resampledestimate ≥ estimate
            n += 1
        end
    end

    return n / bootstrap_iters
end
