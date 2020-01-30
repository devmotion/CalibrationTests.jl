struct AsymptoticQuadraticTest{K<:MatrixKernel,P,T,E,V} <: HypothesisTests.HypothesisTest
    """Matrix-valued kernel."""
    kernel::K
    """Predictions."""
    predictions::P
    """Targets."""
    targets::T
    """Calibration error estimate."""
    estimate::E
    """Test statistic."""
    statistic::V
end

AsymptoticQuadraticTest(skce::QuadraticUnbiasedSKCE, data...; kwargs...) =
    AsymptoticQuadraticTest(skce.kernel, data...; kwargs...)

function AsymptoticQuadraticTest(kernel::MatrixKernel, data...)
    # obtain the predictions and targets
    predictions, targets = CalibrationErrors.predictions_targets(data...)

    # compute the calibration error estimate and the test statistic
    estimate, statistic = estimate_statistic(kernel, predictions, targets)

    AsymptoticQuadraticTest(kernel, predictions, targets, estimate, statistic)
end

# HypothesisTests interface

HypothesisTests.default_tail(::AsymptoticQuadraticTest) = :right

function HypothesisTests.pvalue(test::AsymptoticQuadraticTest;
                                rng::AbstractRNG = Random.GLOBAL_RNG,
                                bootstrap_iters::Int = 1_000)
    bootstrap_ccdf(rng, test, bootstrap_iters)
end

HypothesisTests.testname(::AsymptoticQuadraticTest) =
    "Asymptotic calibration test based on the quadratic unbiased SKCE estimator"

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::AsymptoticQuadraticTest)
    "SKCE", zero(test.estimate), test.estimate
end

function HypothesisTests.show_params(io::IO, test::AsymptoticQuadraticTest, ident = "")
    println(io, ident, "test statistic: $(test.statistic)")
end

# compute the unbiased estimate of the SKCE and the test statistic
# `nsamples / (nsamples - 1) * SKCEuq - SKCEb`.
function estimate_statistic(kernel::MatrixKernel,
                            predictions::AbstractVector{<:AbstractVector{<:Real}},
                            targets::AbstractVector{<:Integer})
    # obtain number of samples
    nsamples = length(predictions)
    nsamples > 1 || error("there must be at least two samples")

    # pre-computations
    α = (2 * nsamples  - 1) / (nsamples - 1)^2

    @inbounds begin
        # evaluate kernel function for the first sample
        prediction = predictions[1]
        target = targets[1]

        # initialize the test statistic and the unbiased estimate of the SKCE
        statistic = -skce_kernel(kernel, prediction, target, prediction, target) / 1
        estimate = zero(statistic)

        # add evaluations of all other pairs of samples
        nstatistic = 1
        nestimate = 0 
        for i in 2:nsamples
            predictioni = predictions[i]
            targeti = targets[i]

            for j in 1:(i - 1)
                predictionj = predictions[j]
                targetj = targets[j]

                # evaluate the kernel function
                result = skce_kernel(kernel, predictioni, targeti, predictionj, targetj)

                # update the estimate and the test statistic
                nstatistic += 2
                statistic += 2 * (α * result - statistic) / nstatistic
                nestimate += 1
                estimate += (result - estimate) / nestimate
            end

            # evaluate the kernel function for the `i`th sample
            nstatistic += 1
            result = skce_kernel(kernel, predictioni, targeti, predictioni, targeti)
            statistic -= (statistic + result) / nstatistic
        end
    end

    estimate, statistic
end

# estimate the ccdf using bootstrapping
function bootstrap_ccdf(rng::AbstractRNG, test::AsymptoticQuadraticTest,
                        bootstrap_iters::Int)
    @unpack kernel, predictions, targets, statistic = test

    # initialize array of resampled indices
    nsamples = length(predictions)
    resampledidxs = Vector{Int}(undef, nsamples)

    # for each bootstrap sample
    extreme_count = 0
    @inbounds for _ in 1:bootstrap_iters
        # resample data set
        rand!(rng, resampledidxs, 1:nsamples)

        # evaluate the bootstrap statistic
        meanij = meanik = zero(statistic)
        nij = nik = 0
        @inbounds for i in 1:nsamples
            # obtain the `i`th resampled data pair
            idxi = resampledidxs[i]
            predictioni = predictions[idxi]
            targeti = targets[idxi]

            # evaluate combinations of bootstrapped samples
            @inbounds for j in 1:(i - 1)
                # obtain ith resampled data pair
                idxj = resampledidxs[j]

                # evaluate the kernel function
                result = skce_kernel(kernel, predictioni, targeti, predictions[idxj],
                                     targets[idxj])

                # update the running mean
                nij += 1
                meanij += (result - meanij) / nij
            end

            # evaluate combinations of bootstrapped samples and original samples
            @inbounds for k in 1:nsamples
                # evaluate the kernel function
                result = skce_kernel(kernel, predictioni, targeti, predictions[k],
                                     targets[k])

                # update the running mean
                nik += 1
                meanik += (result - meanik) / nik
            end
        end

        # check if the bootstrap statistic is ≥ the original statistic
        if meanij ≥ statistic + 2 * meanik
            extreme_count += 1
        end
    end

    extreme_count / bootstrap_iters
end
