@doc raw"""
    AsymptoticSKCETest(kernel::Kernel, data...)
    AsymptoticSKCETest(skce::UnbiasedSKCE, data...)

Calibration hypothesis test based on the unbiased estimator of the squared kernel
calibration error (SKCE) with quadratic sample complexity.

# Details

Let ``\mathcal{D} = (P_{X_i}, Y_i)_{i=1,\ldots,n}`` be a data set of predictions and
corresponding targets. Denote the null hypothesis "the predictive probabilistic model is
calibrated" with ``H_0``.

The hypothesis test approximates the p-value ``ℙ(\mathrm{SKCE}_{uq} > c \,|\, H_0)``, where
``\mathrm{SKCE}_{uq}`` is the unbiased estimator of the SKCE, defined as
```math
\frac{2}{n(n-1)} \sum_{1 \leq i < j \leq n} h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big),
```
where
```math
\begin{aligned}
h_k\big((μ, y), (μ', y')\big) ={}&   k\big((μ, y), (μ', y')\big)
                                   - 𝔼_{Z ∼ μ} k\big((μ, Z), (μ', y')\big) \\
                                 & - 𝔼_{Z' ∼ μ'} k\big((μ, y), (μ', Z')\big)
                                   + 𝔼_{Z ∼ μ, Z' ∼ μ'} k\big((μ, Z), (μ', Z')\big).
\end{aligned}
```

The p-value is estimated based on the asymptotically valid approximation
```math
ℙ(n\mathrm{SKCE}_{uq} > c \,|\, H_0) \approx ℙ(T > c \,|\, \mathcal{D}),
```
where ``T`` is the bootstrap statistic
```math
T = \frac{2}{n} \sum_{1 \leq i < j \leq n} \bigg(h_k\big((P^*_{X_i}, Y^*_i), (P^*_{X_j}, Y^*_j)\big)
- \frac{1}{n} \sum_{r = 1}^n h_k\big((P^*_{X_i}, Y^*_i), (P_{X_r}, Y_r)\big)
- \frac{1}{n} \sum_{r = 1}^n h_k\big((P_{X_r}, Y_r), (P^*_{X_j}, Y^*_j)\big)
+ \frac{1}{n^2} \sum_{r, s = 1}^n h_k\big((P_{X_r}, Y_r), (P_{X_s}, Y_s)\big)\bigg)
```
for bootstrap samples ``(P^*_{X_i}, Y^*_i)_{i=1,\ldots,n}`` of ``\mathcal{D}``.
This can be reformulated to the approximation
```math
ℙ(n\mathrm{SKCE}_{uq}/(n - 1) - \mathrm{SKCE}_b > c \,|\, H_0) \approx ℙ(T' > c \,|\, \mathcal{D}),
```
where
```math
\mathrm{SKCE}_b = \frac{1}{n^2} \sum_{i, j = 1}^n h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big)
```
and
```math
T' = \frac{2}{n(n - 1)} \sum_{1 \leq i < j \leq n} h_k\big((P^*_{X_i}, Y^*_i), (P^*_{X_j}, Y^*_j)\big)
- \frac{2}{n^2} \sum_{i, r=1}^n h_k\big((P^*_{X_i}, Y^*_i), (P_{X_r}, Y_r)\big).
```

# References

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html).
In: Advances in Neural Information Processing Systems (NeurIPS 2019) (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021). [Calibration tests beyond
classification](https://openreview.net/forum?id=-bxf89v3Nx).
"""
struct AsymptoticSKCETest{K<:Kernel,E,V,M} <: HypothesisTests.HypothesisTest
    """Kernel."""
    kernel::K
    """Calibration error estimate."""
    estimate::E
    """Test statistic."""
    statistic::V
    """Symmetric kernel matrix, consisting of pairwise evaluations of ``h_{ij}``."""
    kernelmatrix::M
end

AsymptoticSKCETest(skce::UnbiasedSKCE, data...) = AsymptoticSKCETest(skce.kernel, data...)

function AsymptoticSKCETest(kernel::Kernel, data...)
    # obtain the predictions and targets
    predictions, targets = CalibrationErrors.predictions_targets(data...)

    # compute the calibration error estimate, the test statistic, and the kernel matrix
    estimate, statistic, kernelmatrix = estimate_statistic_kernelmatrix(
        kernel, predictions, targets
    )

    return AsymptoticSKCETest(kernel, estimate, statistic, kernelmatrix)
end

# HypothesisTests interface

HypothesisTests.default_tail(::AsymptoticSKCETest) = :right

function HypothesisTests.pvalue(test::AsymptoticSKCETest; kwargs...)
    return pvalue(Random.GLOBAL_RNG, test; kwargs...)
end
function HypothesisTests.pvalue(
    rng::AbstractRNG, test::AsymptoticSKCETest; bootstrap_iters::Int=1_000
)
    return bootstrap_ccdf(rng, test.statistic, test.kernelmatrix, bootstrap_iters)
end

HypothesisTests.testname(::AsymptoticSKCETest) = "Asymptotic SKCE test"

# parameter of interest: name, value under H0, point estimate
function HypothesisTests.population_param_of_interest(test::AsymptoticSKCETest)
    return "SKCE", zero(test.estimate), test.estimate
end

function HypothesisTests.show_params(io::IO, test::AsymptoticSKCETest, ident="")
    return println(io, ident, "test statistic: $(test.statistic)")
end

@doc raw"""
    estimate_statistic_kernelmatrix(kernel, predictions, targets)

Compute the estimate of the SKCE, the test statistic, and the matrix of the evaluations of
the kernel function.

# Details

Let ``\mathcal{D} = (P_{X_i}, Y_i)_{i=1,\ldots,n}`` be a data set of predictions and
corresponding targets.

The unbiased estimator ``\mathrm{SKCE}_{uq}`` of the SKCE is defined as
```math
\frac{2}{n(n-1)} \sum_{1 \leq i < j \leq n} h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big),
```
where
```math
\begin{aligned}
h_k\big((μ, y), (μ', y')\big) ={}&   k\big((μ, y), (μ', y')\big)
                                   - 𝔼_{Z ∼ μ} k\big((μ, Z), (μ', y')\big) \\
                                 & - 𝔼_{Z' ∼ μ'} k\big((μ, y), (μ', Z')\big)
                                   + 𝔼_{Z ∼ μ, Z' ∼ μ'} k\big((μ, Z), (μ', Z')\big).
\end{aligned}
```

The test statistic is defined as
```math
\frac{n}{n-1} \mathrm{SKCE}_{uq} - \mathrm{SKCE}_b,
```
where
```math
\mathrm{SKCE}_b = \frac{1}{n^2} \sum_{i, j = 1}^n h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big)
```
(see [`AsymptoticSKCETest`](@ref)). This is equivalent to
```math
\frac{1}{n^2} \sum_{i, j = 1} h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big) \bigg(\frac{n^2}{(n - 1)^2} 1(i \neq j) - \bigg).
```

The kernelmatrix ``K \in \mathbb{R}^{n \times n}`` is defined as 
```math
K_{ij} = h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big)
```
for ``i, j \in \{1, \ldots, n\}``.
"""
function estimate_statistic_kernelmatrix(kernel, predictions, targets)
    # obtain number of samples
    nsamples = length(predictions)
    nsamples > 1 || error("there must be at least two samples")

    # pre-computations
    α = (2 * nsamples - 1) / (nsamples - 1)^2

    @inbounds begin
        # evaluate the kernel function for the first pair of samples
        prediction = predictions[1]
        target = targets[1]

        # initialize the kernel matrix
        hij = unsafe_skce_eval(kernel, prediction, target, prediction, target)
        kernelmatrix = Matrix{typeof(hij)}(undef, nsamples, nsamples)
        kernelmatrix[1, 1] = hij

        # initialize the test statistic and the unbiased estimate of the SKCE
        statistic = -hij / 1
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
                hij = unsafe_skce_eval(kernel, predictioni, targeti, predictionj, targetj)

                # update the kernel matrix
                kernelmatrix[j, i] = hij

                # update the estimate and the test statistic
                nstatistic += 2
                statistic += 2 * (α * hij - statistic) / nstatistic
                nestimate += 1
                estimate += (hij - estimate) / nestimate
            end

            # evaluate the kernel function for the `i`th sample
            hij = unsafe_skce_eval(kernel, predictioni, targeti, predictioni, targeti)

            # update the kernel matrix
            kernelmatrix[i, i] = hij

            # update the test statistic
            nstatistic += 1
            statistic -= (statistic + hij) / nstatistic
        end
    end

    # add lower triangle of the kernel matrix
    LinearAlgebra.copytri!(kernelmatrix, 'U')

    return estimate, statistic, kernelmatrix
end

@doc raw"""
    bootstrap_ccdf(rng::AbstractRNG, statistic, kernelmatrix, bootstrap_iters::Int)

Estimate the value of the inverse CDF of the test statistic under the calibration null
hypothesis by bootstrapping.

# Details

Let ``\mathcal{D} = (P_{X_i}, Y_i)_{i=1,\ldots,n}`` be a data set of predictions and
corresponding targets. Denote the null hypothesis "the predictive probabilistic model is
calibrated" with ``H_0``, and the test statistic with ``T``.

The value of the inverse CDF under the null hypothesis is estimated based on the
asymptotically valid approximation
```math
ℙ(T > c \,|\, H_0) \approx ℙ(T' > c \,|\, \mathcal{D}),
```
where the bootstrap statistic ``T'`` is defined as
```math
T' = \frac{2}{n(n - 1)} \sum_{1 \leq i < j \leq n} h_k\big((P^*_{X_i}, Y^*_i), (P^*_{X_j}, Y^*_j)\big)
- \frac{2}{n^2} \sum_{i, r=1}^n h_k\big((P^*_{X_i}, Y^*_i), (P_{X_r}, Y_r)\big)
```
for bootstrap samples ``(P^*_{X_i}, Y^*_i)_{i=1,\ldots,n}`` of ``\mathcal{D}``
(see [`AsymptoticSKCETest`](@ref)).

Let ``C_i`` be the number of times that data pair ``(P_{X_i}, Y_i)`` was resampled.
Then we obtain
```math
T' = \frac{1}{n^2} \sum_{i=1}^n C_i \sum_{j=1}^n \bigg(\frac{n}{n-1} (C_j - \delta_{i,j}) - 2\bigg) h_k\big((P_{X_i}, Y_i), (P_{X_j}, Y_j)\big).
```
"""
function bootstrap_ccdf(rng::AbstractRNG, statistic, kernelmatrix, bootstrap_iters::Int)
    # initialize array of counts of resampled indices
    nsamples = LinearAlgebra.checksquare(kernelmatrix)
    resampling_counts = Vector{Int}(undef, nsamples)

    # for each bootstrap sample
    α = nsamples / (nsamples - 1)
    extreme_count = 0
    sampler = Random.Sampler(rng, 1:nsamples)
    for _ in 1:bootstrap_iters
        # resample data set
        fill!(resampling_counts, 0)
        for _ in 1:nsamples
            idx = rand(rng, sampler)
            @inbounds resampling_counts[idx] += 1
        end

        # evaluate the bootstrap statistic
        z = zero(statistic)
        n = 0
        for i in 1:nsamples
            # check if the `i`th data pair was sampled
            ci = resampling_counts[i]
            iszero(ci) && continue

            zi = mean(enumerate(resampling_counts)) do (j, cj)
                # obtain evaluation of the kernel function
                @inbounds hij = kernelmatrix[j, i]
                return ((cj - (i == j)) * α - 2) * hij
            end

            # update bootstrap statistic
            n += ci
            z += ci * (zi - z) / n
        end

        # check if the bootstrap statistic is ≥ the original statistic
        if z ≥ statistic
            extreme_count += 1
        end
    end

    return extreme_count / bootstrap_iters
end
