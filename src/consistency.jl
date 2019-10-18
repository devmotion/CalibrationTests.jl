struct ConsistencyTest{E<:CalibrationErrorEstimator,V} <: CalibrationTest
    """Calibration estimator."""
    estimator::E
    """Calibration estimate."""
    estimate::V
    """Bootstrap samples using consistency resampling."""
    straps::Vector{V}
end

function ConsistencyTest(estimator::CalibrationErrorEstimator,
                         data::Tuple{<:AbstractMatrix{<:Real},<:AbstractVector{<:Integer}};
                         nruns::Int = 1_000,
                         rng::AbstractRNG = Random.GLOBAL_RNG)
    # check if the number of predictions and labels is equal
    predictions, labels = CalibrationErrors.get_predictions_labels(data)

    # perform consistency resampling and evaluate calibration error
    estimate, straps = consistency_resampling(rng, estimator, predictions, labels, nruns)

    ConsistencyTest(estimator, estimate, straps)
end

HypothesisTests.testname(::ConsistencyTest) =
    "Calibration test based on consistency resampling"

function HypothesisTests.pvalue(test::ConsistencyTest)
    # count number of sampled estimates greater or equal to the observed estimate
    estimate = test.estimate
    n = 0
    for sample in test.straps
        sample ≥ estimate && (n += 1)
    end

    n / length(test.straps)
end

# TODO: move the following code upstream
function consistency_resampling(rng, estimator, predictions, labels, nruns)
    nclasses, nsamples = size(predictions)

    # use same heuristic as StatsBase to decide whether to sample predictions
    # directly or to build an alias table
    # TODO: needs proper benchmarking
    if nsamples < 40
        consistency_resampling_direct(rng, estimator, predictions, labels, nruns)
    else
        t = nsamples < 500 ? 64 : 32
        if nclasses < t
            consistency_resampling_direct(rng, estimator, predictions, labels, nruns)
        else
            consistency_resampling_alias(rng, estimator, predictions, labels, nruns)
        end
    end
end


# sample labels directly (without alias table)
function consistency_resampling_direct(rng::AbstractRNG, estimator, predictions, labels,
                                       nruns)
    nclasses, nsamples = size(predictions)

    # evaluate statistic
    estimate = calibrationerror(estimator, (predictions, labels))

    # create caches
    resampled_predictions = similar(predictions)
    resampled_labels = similar(labels)
    resampled_data = (resampled_predictions, resampled_labels)

    # create sampler
    sp = Random.Sampler(rng, Base.OneTo(nsamples))

    # create output
    straps = Vector{typeof(estimate)}(undef, nruns)

    # for each resampling step
    @inbounds for i in 1:nruns
        # resample data
        for j in 1:nsamples
            # resample predictions
            idx = rand(rng, sp)
            for k in axes(predictions, 1)
                resampled_predictions[k, j] = predictions[k, idx]
            end

            # resample labels
            p = rand(rng)
            cw = resampled_predictions[1, j]
            label = 1
            while cw < p && label < nclasses
                label += 1
                cw += resampled_predictions[label, j]
            end
            resampled_labels[j] = label
        end

        # evaluate calibration error
        straps[i] = calibrationerror(estimator, resampled_data)
    end

    estimate, straps
end

# sample labels with alias table
function consistency_resampling_alias(rng::AbstractRNG, estimator, predictions, labels,
                                      nruns)
    nclasses, nsamples = size(predictions)

    # evaluate statistic
    estimate = calibrationerror(estimator, (predictions, labels))

    # create alias table
    accept = [Vector{Float64}(undef, nclasses) for _ in 1:nsamples]
    alias = [Vector{Int}(undef, nclasses) for _ in 1:nsamples]
    @inbounds for i in axes(predictions, 2)
        StatsBase.make_alias_table!(view(predictions, :, i), 1.0, accept[i], alias[i])
    end

    # create sampler of labels
    splabels = Random.Sampler(rng, Base.OneTo(nclasses))

    # create caches
    resampled_predictions = similar(predictions)
    resampled_labels = similar(labels)
    resampled_data = (resampled_predictions, resampled_labels)

    # create sampler
    sp = Random.Sampler(rng, Base.OneTo(nsamples))

    # create output
    straps = Vector{typeof(estimate)}(undef, nruns)

    # for each resampling step
    @inbounds for i in 1:nruns
        # resample data
        for j in 1:nsamples
            # resample predictions
            idx = rand(rng, sp)
            for k in axes(predictions, 1)
                resampled_predictions[k, j] = predictions[k, idx]
            end

            # resample labels
            l = rand(rng, splabels)
            resampled_labels[j] = rand(rng) < accept[j][l] ? l : alias[j][l]
        end

        # evaluate calibration error
        straps[i] = calibrationerror(estimator, resampled_data)
    end

    estimate, straps
end
