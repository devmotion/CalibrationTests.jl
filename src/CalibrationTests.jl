module CalibrationTests

using Reexport

@reexport using CalibrationErrors
using ConsistencyResampling: ConsistencyResampling
using HypothesisTests: HypothesisTests
@reexport using KernelFunctions
using LinearAlgebra: LinearAlgebra
using Random: Random
using Statistics: Statistics
using StatsFuns: StatsFuns
using StructArrays: StructArrays

using CalibrationErrors: CalibrationErrorEstimator, SKCE, unsafe_skce_eval

export ConsistencyTest
export DistributionFreeSKCETest, AsymptoticBlockSKCETest, AsymptoticSKCETest
export AsymptoticCMETest

# re-export
using HypothesisTests: pvalue, confint
export pvalue, confint

include("consistency.jl")

include("skce/asymptotic.jl")
include("skce/asymptotic_block.jl")
include("skce/distribution_free.jl")

include("cme.jl")

include("deprecated.jl")

end # module
