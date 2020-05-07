struct AsymptoticBlockSKCETest{K<:Kernel,E,S,Z} <: HypothesisTests.ZTest
    """Kernel."""
    kernel::K
    """Number of observations per block."""
    blocksize::Int
    """Number of blocks of observations."""
    nblocks::Int
    """Calibration error estimate (average of evaluations of blocks of observations)."""
    estimate::E
    """Standard error of evaluations of blocks of observations."""
    stderr::S
    """z-statistic."""
    z::Z
end

function AsymptoticBlockSKCETest(skce::BlockUnbiasedSKCE, data...)
    return AsymptoticBlockSKCETest(skce.kernel, skce.blocksize, data...)
end

function AsymptoticBlockSKCETest(
    κpredictions::Kernel,
    κtargets::Kernel,
    args...
)
    return AsymptoticBlockSKCETest(TensorProduct(κpredictions, κtargets), args...)
end

function AsymptoticBlockSKCETest(kernel::Kernel, blocksize::Int, data...)
    # obtain predictions and targets
    predictions, targets = CalibrationErrors.predictions_targets(data...)

    # obtain number of samples
    nsamples = length(predictions)
    nsamples ≥ blocksize || error("there must be at least ", blocksize, " samples")

    # compute number of blocks
    nblocks = nsamples ÷ blocksize

    # evaluate U-statistic of the first block
    istart = 1
    iend = blocksize
    x = CalibrationErrors.unsafe_unbiasedskce(kernel, predictions, targets, istart, iend)

    # initialize the estimate and the sum of squares
    estimate = x / 1
    S = zero(x)

    # for all other blocks
    for b in 2:nblocks
        # evaluate U-statistic
        istart += blocksize
        iend += blocksize
        x = CalibrationErrors.unsafe_unbiasedskce(kernel, predictions, targets, istart,
                                                  iend)

        # update the estimate
        Δestimate = x - estimate
        estimate += Δestimate / b
        S += Δestimate * (x - estimate)
    end

    # compute standard error and z-statistic
    stderr = sqrt(S) / nblocks
    z = estimate / stderr

    return AsymptoticBlockSKCETest(kernel, blocksize, nblocks, estimate, stderr, z)
end

# HypothesisTests interface

HypothesisTests.default_tail(::AsymptoticBlockSKCETest) = :right

## have to specify and check keyword arguments in `pvalue` and `confint` to
## force `tail = :right` due to the default implementation in HypothesisTests

function HypothesisTests.pvalue(test::AsymptoticBlockSKCETest; tail = :right)
    if tail === :right
        normccdf(test.z)
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

# confidence interval by inversion
function StatsBase.confint(test::AsymptoticBlockSKCETest; level = 0.95, tail = :right)
    HypothesisTests.check_level(level)

    if tail === :right
        q = norminvcdf(level)
        lowerbound = test.estimate - q * test.stderr
        (lowerbound, oftype(lowerbound, Inf))
    else
        throw(ArgumentError("tail = $(tail) is invalid"))
    end
end

HypothesisTests.testname(test::AsymptoticBlockSKCETest) = "Asymptotic block SKCE test"

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::AsymptoticBlockSKCETest)
    "SKCE", zero(test.estimate), test.estimate
end

function HypothesisTests.show_params(io::IO, test::AsymptoticBlockSKCETest, ident = "")
    println(io, ident, "number of observations per block: ", test.blocksize)
    println(io, ident, "number of blocks of observations: ", test.nblocks)
    println(io, ident, "z-statistic: ", test.z)
    println(io, ident, "standard error of evaluations of pairs of observations: ",
            test.stderr)
end
