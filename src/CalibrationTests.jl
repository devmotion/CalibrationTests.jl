module CalibrationTests

using Reexport

@reexport using CalibrationErrors
using HypothesisTests
@reexport using KernelFunctions
using UnPack
using StatsBase
using StatsFuns

using LinearAlgebra
using Random
using Statistics

using CalibrationErrors: CalibrationErrorEstimator, SKCE, unsafe_skce_eval

export ConsistencyTest
export DistributionFreeSKCETest, AsymptoticBlockSKCETest, AsymptoticSKCETest
export AsymptoticCMETest

# re-export
export pvalue, confint

include("consistency.jl")

include("skce/asymptotic.jl")
include("skce/asymptotic_block.jl")
include("skce/distribution_free.jl")

include("cme.jl")

end # module
