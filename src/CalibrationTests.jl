module CalibrationTests

using CalibrationErrors
using HypothesisTests
using Parameters
using StatsBase
using StatsFuns

using LinearAlgebra
using Random

using CalibrationErrors: CalibrationErrorEstimator, SKCE, skce_kernel,
    MatrixKernel

export ConsistencyTest
export DistributionFreeTest, AsymptoticLinearTest, AsymptoticQuadraticTest

# re-export
export pvalue, confint

include("consistency.jl")

include("skce/distribution_free.jl")
include("skce/asymptotic_linear.jl")
include("skce/asymptotic_quadratic.jl")

end # module
