@deprecate AsymptoticSKCETest(kernel1::Kernel, kernel2::Kernel, data...; kwargs...) AsymptoticSKCETest(
    kernel1 ⊗ kernel2, data...; kwargs...
)

@deprecate AsymptoticBlockSKCETest(kernel1::Kernel, kernel2::Kernel, args...) AsymptoticBlockSKCETest(
    kernel1 ⊗ kernel2, args...
)
