# scalar-valued kernel
Base.maximum(kernel::CalibrationErrors.ExponentialKernel) = exp(zero(kernel.γ))
Base.maximum(kernel::CalibrationErrors.SquaredExponentialKernel) = exp(zero(kernel.γ))

# matrix-valued kernels
Base.maximum(kernel::CalibrationErrors.UniformScalingKernel) =
    kernel.λ * maximum(kernel.kernel)
Base.maximum(kernel::CalibrationErrors.DiagonalKernel) =
    maximum(kernel.diag) * maximum(kernel.kernel)

# by default assume p = q
Base.maximum(kce::SKCE) = 2 * maximum(CalibrationErrors.kernel(kce))
