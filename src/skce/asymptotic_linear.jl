struct AsymptoticLinearTest{K<:Kernel,E,S,Z} <: HypothesisTests.ZTest
    """Kernel."""
    kernel::K
    """Number of pairs of observations."""
    n::Int
    """Calibration error estimate (average of evaluations of pairs of observations)."""
    estimate::E
    """Standard error of evaluations of pairs of observations."""
    stderr::S
    """z-statistic."""
    z::Z
end

AsymptoticLinearTest(skce::LinearUnbiasedSKCE, data...) =
    AsymptoticLinearTest(skce.kernel, data...)

AsymptoticLinearTest(kernel1::Kernel, kernel2::Kernel, data...) =
    AsymptoticLinearTest(TensorProductKernel(kernel1, kernel2), data...)

function AsymptoticLinearTest(kernel::Kernel, data...)
    # obtain predictions and targets
    predictions, targets = CalibrationErrors.predictions_targets(data...)

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ 2 || error("there must be at least two samples")

    @inbounds begin
        # evaluate the kernel function for the first pair of samples
        hij = unsafe_skce_eval(kernel, predictions[1], targets[1], predictions[2], targets[2])

        # initialize the estimate and the sum of squares
        estimate = hij / 1
        S = zero(hij)

        # add evaluations of all subsequent pairs of samples
        n = 1
        for i in 3:2:(nsamples - 1)
            j = i + 1

            # evaluate the kernel function
            hij = unsafe_skce_eval(kernel, predictions[i], targets[i], predictions[j],
                                   targets[j])

            # update the estimate and the sum of squares
            n += 1
            Δestimate = hij - estimate
            estimate += Δestimate / n
            S += Δestimate * (hij - estimate)
        end
    end

    # compute standard error and z-statistic
    stderr = sqrt(S) / n
    z = estimate / stderr

    AsymptoticLinearTest(kernel, n, estimate, stderr, z)
end

# HypothesisTests interface

HypothesisTests.default_tail(::AsymptoticLinearTest) = :right

## have to specify and check keyword arguments in `pvalue` and `confint` to
## force `tail = :right` due to the default implementation in HypothesisTests

function HypothesisTests.pvalue(test::AsymptoticLinearTest; tail = :right)
    if tail === :right
        normccdf(test.z)
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

# confidence interval by inversion
function StatsBase.confint(test::AsymptoticLinearTest; level = 0.95, tail = :right)
    HypothesisTests.check_level(level)

    if tail === :right
        q = norminvcdf(level)
        lowerbound = test.estimate - q * test.stderr
        (lowerbound, oftype(lowerbound, Inf))
    else
        throw(ArgumentError("tail = $(tail) is invalid"))
    end
end

HypothesisTests.testname(::AsymptoticLinearTest) =
    "Asymptotic calibration test based on the linear unbiased SKCE estimator"

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::AsymptoticLinearTest)
    "SKCE", zero(test.estimate), test.estimate
end

function HypothesisTests.show_params(io::IO, test::AsymptoticLinearTest, ident = "")
    println(io, ident, "number of pairs of observations: $(test.n)")
    println(io, ident, "z-statistic: $(test.z)")
    println(io, ident, "standard error of evaluations of pairs of observations: $(test.stderr)")
end
