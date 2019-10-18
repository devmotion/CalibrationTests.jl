struct AsymptoticLinearTest{K<:MatrixKernel,V,S} <: CalibrationTest
    """Matrix-valued kernel."""
    kernel::K
    """Calibration error estimate."""
    estimate::V
    """Standard deviation of asymptotic normal distribution."""
    std::S
end

AsymptoticLinearTest(skce::LinearUnbiasedSKCE, data) =
    AsymptoticLinearTest(skce.kernel, data)

function AsymptoticLinearTest(kernel::MatrixKernel,
                              data::Tuple{<:AbstractMatrix{<:Real},<:AbstractVector{<:Integer}})
    # check if the number of predictions and labels is equal
    predictions, labels = CalibrationErrors.get_predictions_labels(data)

    estimate, std = _calibration_std(kernel, predictions, labels)

    AsymptoticLinearTest(kernel, estimate, std)
end

HypothesisTests.testname(::AsymptoticLinearTest) =
    "Asymptotic calibration test based on the linear unbiased SKCE estimator"

HypothesisTests.pvalue(test::AsymptoticLinearTest) = normccdf(0, test.std, test.estimate)

"""
    _calibration_std(kernel, predictions, labels)

Evaluate the linear unbiased SKCE estimator with kernel `kernel` and estimate the standard
devation of its asymptotic normal distribution around the true value for the given
`predictions` and `labels`.
"""
function _calibration_std(kernel::MatrixKernel, predictions::AbstractMatrix{<:Real},
                          labels::AbstractVector{<:Integer})
    # precalculations
    nsamples = size(predictions, 2)

    # initialize estimate and sum of squares
    m = zero(skce_result_type(LinearUnbiasedSKCE(kernel), predictions))
    S = m^2

    n, i = 0, 1
    @inbounds while i < nsamples
        # update number of summands
        n += 1

        # evaluate kernel of next two samples
        j = i + 1
        s = skce_kernel(kernel, view(predictions, :, j), labels[j], view(predictions, :, i),
                        labels[i])

        # update mean
        Δm = s - m
        m += inv(n) * Δm

        # update sum of squares
        S += Δm * (s - m)

        # update index of next sample
        i += 2
    end

    m, inv(n) * sqrt(S)
end
