struct AsymptoticCMETest{K<:Kernel,V,M,S} <: HypothesisTests.HypothesisTest
    """Kernel."""
    kernel::K
    """Number of observations."""
    nsamples::Int
    """Number of test locations."""
    ntestsamples::Int
    """UCME estimate."""
    estimate::V
    """Mean deviation for each test location."""
    mean_deviations::M
    """Test statistic."""
    statistic::S
end

function AsymptoticCMETest(estimator::UCME, data...)
    @unpack kernel, testpredictions, testtargets = estimator

    # obtain predictions and targets
    predictions, targets = CalibrationErrors.predictions_targets(data...)

    # determine number of observations and test locations
    nsamples = length(predictions)
    ntestsamples = length(testpredictions)

    # create matrix of deviations (observations × test locations)
    deviations =
        CalibrationErrors.unsafe_ucme_eval.(
            (kernel,),
            predictions,
            targets,
            permutedims(testpredictions),
            permutedims(testtargets),
        )

    # compute UCME estimate
    mean_deviations = mean(deviations; dims=1)
    estimate = sum(abs2, mean_deviations) / ntestsamples

    # compute test statistic
    C = Symmetric(Statistics.covm(deviations, mean_deviations))
    x = vec(mean_deviations)
    statistic = nsamples * dot(x, C \ x)

    return AsymptoticCMETest(kernel, nsamples, ntestsamples, estimate, x, statistic)
end

# HypothesisTests interface

HypothesisTests.default_tail(::AsymptoticCMETest) = :right

## have to specify and check keyword arguments in `pvalue` and `confint` to
## force `tail = :right` due to the default implementation in HypothesisTests

function HypothesisTests.pvalue(test::AsymptoticCMETest; tail=:right)
    if tail === :right
        chisqccdf(test.ntestsamples, test.statistic)
    else
        throw(ArgumentError("tail=$(tail) is invalid"))
    end
end

HypothesisTests.testname(::AsymptoticCMETest) = "Asymptotic CME test"

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::AsymptoticCMETest)
    return "Mean vector", zero(test.mean_deviations), test.mean_deviations
end

function HypothesisTests.show_params(io::IO, test::AsymptoticCMETest, ident="")
    println(io, ident, "number of observations: ", test.nsamples)
    println(io, ident, "number of test locations: ", test.ntestsamples)
    println(io, ident, "UCME estimate: ", test.estimate)
    return println(io, ident, "test statistic: ", test.statistic)
end


# confidence interval by inversion
#=
function StatsBase.confint(test::AsymptoticCMETest; level=0.95, tail=:right)
    HypothesisTests.check_level(level)

    if tail === :right
        q = chisqinvcdf(test.ntestsamples, level)
        lowerbound = test.statistic - q * test.stderr
        (lowerbound, oftype(lowerbound, Inf))
    else
        throw(ArgumentError("tail = $(tail) is invalid"))
    end
end
=#