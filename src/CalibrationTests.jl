module CalibrationTests

using CalibrationErrors
using HypothesisTests
using UnPack
using StatsBase
using StatsFuns

using LinearAlgebra
using Random

using CalibrationErrors: CalibrationErrorEstimator, SKCE, unsafe_skce_eval

export ConsistencyTest
export DistributionFreeSKCETest, AsymptoticBlockSKCETest, AsymptoticSKCETest

# re-export
export pvalue, confint

include("consistency.jl")

include("skce/asymptotic.jl")
include("skce/asymptotic_block.jl")
include("skce/distribution_free.jl")

include("deprecated.jl")

end # module
