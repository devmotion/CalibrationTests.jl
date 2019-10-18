module CalibrationTests

using CalibrationErrors
using HypothesisTests
using Parameters
using StatsBase
using StatsFuns

using LinearAlgebra
using Random

using CalibrationErrors: CalibrationErrorEstimator, SKCE, skce_kernel, skce_result_type,
    MatrixKernel

export ConsistencyTest
export DistributionFreeTest, AsymptoticLinearTest, AsymptoticQuadraticTest

# re-export
export pvalue

# defaults
abstract type CalibrationTest <: HypothesisTests.HypothesisTest end

estimator(test::CalibrationTest) = test.estimator
HypothesisTests.default_tail(test::CalibrationTest) = :right

include("consistency.jl")

include("skce/utils.jl")
include("skce/distribution_free.jl")
include("skce/asymptotic_linear.jl")
include("skce/asymptotic_quadratic.jl")

end # module
