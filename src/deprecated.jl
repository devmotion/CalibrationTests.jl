@deprecate ConsistencyTest(estimator::CalibrationErrorEstimator, data...) ConsistencyTest(
    estimator, CalibrationErrors.predictions_targets(data...)...
)

@deprecate DistributionFreeSKCETest(estimator::SKCE, data...; kwargs...) DistributionFreeSKCETest(
    estimator, CalibrationErrors.predictions_targets(data...)...; kwargs...
)

@deprecate AsymptoticSKCETest(skce::UnbiasedSKCE, data...) AsymptoticSKCETest(
    skce.kernel, data...
)
@deprecate AsymptoticSKCETest(kernel::Kernel, data...) AsymptoticSKCETest(
    kernel, CalibrationErrors.predictions_targets(data...)...
)

@deprecate AsymptoticBlockSKCETest(skce::BlockUnbiasedSKCE, data...) AsymptoticBlockSKCETest(
    skce.kernel, skce.blocksize, data...
)
@deprecate AsymptoticBlockSKCETest(kernel::Kernel, blocksize::Int, data...) AsymptoticBlockSKCETest(
    kernel, blocksize, CalibrationErrors.predictions_targets(data...)...
)

@deprecate AsymptoticCMETest(estimator::UCME, data...) AsymptoticCMETest(
    estimator, CalibrationErrors.predictions_targets(data...)...
)
