using SafeTestsets

@safetestset "Binary trend tests" begin include("binary_trend.jl") end
@safetestset "Consistency test" begin include("consistency.jl") end
@safetestset "Distribution-free tests" begin include("skce/distribution_free.jl") end
@safetestset "Asymptotic linear test" begin include("skce/asymptotic_linear.jl") end
@safetestset "Asymptotic quadratic test" begin include("skce/asymptotic_quadratic.jl") end
