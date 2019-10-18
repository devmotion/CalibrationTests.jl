struct AsymptoticQuadraticTest{K<:MatrixKernel,V} <: CalibrationTest
    """Matrix-valued kernel."""
    kernel::K
    """Test statistic."""
    statistic::V
    """Samples of bootstrap statistic assuming strong calibration."""
    straps::Vector{V}
end

AsymptoticQuadraticTest(skce::QuadraticUnbiasedSKCE, data; kwargs...) =
    AsymptoticQuadraticTest(skce.kernel, data; kwargs...)

function AsymptoticQuadraticTest(kernel::MatrixKernel,
                                 data::Tuple{<:AbstractMatrix{<:Real},<:AbstractVector{<:Integer}};
                                 kwargs...)
    # check if the number of predictions and labels is equal
    predictions, labels = CalibrationErrors.get_predictions_labels(data)

    # compute statistic and bootstrap samples
    statistic, straps = _statistic_straps(kernel, predictions, labels; kwargs...)

    AsymptoticQuadraticTest(kernel, statistic, straps)
end

HypothesisTests.testname(::AsymptoticQuadraticTest) =
    "Asymptotic calibration test based on the quadratic unbiased SKCE estimator"

function HypothesisTests.pvalue(test::AsymptoticQuadraticTest)
    @unpack statistic, straps = test

    n = 0
    for strap in straps
        strap ≥ statistic && (n += 1)
    end

    n / length(straps)
end

function _statistic_straps(kernel::MatrixKernel,
                           predictions::AbstractMatrix{<:Real},
                           labels::AbstractVector{<:Integer};
                           nruns::Int = 1_000,
                           rng::AbstractRNG = Random.GLOBAL_RNG)
    nsamples = size(predictions, 2)

    # compute the test statistic and a constant term in the following computation
    statistic = zero(skce_result_type(QuadraticUnbiasedSKCE(kernel), predictions))
    constant = statistic
    for j in 1:nsamples
        # obtain jth data pair
        predictions_j = view(predictions, :, j)
        labels_j = labels[j]

        for i in 1:(j-1)
            val = skce_kernel(kernel, view(predictions, :, i), labels[i], predictions_j,
                        labels_j)
            statistic += val
            constant += 2 * val
        end

        # add diagonal terms to constant
        constant += skce_kernel(kernel, predictions_j, labels_j, predictions_j, labels_j)
    end

    # apply scaling
    statistic *= 2 * inv(nsamples - 1)
    invnsamples = inv(nsamples)
    oneminusinvnsamples = 1 - inv(nsamples)
    twoinvnsamples = 2 * invnsamples
    constant *= oneminusinvnsamples * invnsamples

    # initialize output array of bootstrap statistic
    straps = Vector{typeof(constant)}(undef, nruns)

    # create sampler and initialize array of resampled indices
    resampler = Random.Sampler(rng, Base.OneTo(nsamples))
    resampled_idxs = Vector{Int}(undef, nsamples)

    # for each bootstrap sample
    @inbounds for k in eachindex(straps)
        # resample data set
        rand!(rng, resampled_idxs, resampler)

        # evaluate bootstrap statistic
        s = zero(constant)
        @inbounds for j in 1:nsamples
            # obtain jth resampled data pair
            resampled_idxs_j = resampled_idxs[j]
            resampled_predictions_j = view(predictions, :, resampled_idxs_j)
            resampled_labels_j = labels[resampled_idxs_j]

            # evaluate combinations of bootstrapped samples
            @inbounds for i in 1:(j-1)
                # obtain ith resampled data pair
                resampled_idxs_i = resampled_idxs[i]
                s += skce_kernel(kernel, view(predictions, :, resampled_idxs_i),
                                 labels[resampled_idxs_i], resampled_predictions_j,
                                 resampled_labels_j)
            end

            # substract combinations of bootstrapped samples and original samples
            t = zero(s)
            @inbounds for i in 1:nsamples
                t += skce_kernel(kernel, view(predictions, :, i), labels[i],
                                 resampled_predictions_j, resampled_labels_j)
            end

            # subtract terms
            s -= oneminusinvnsamples * t
        end

        # combine result
        straps[k] = constant + twoinvnsamples * s
    end

    statistic, straps
end
