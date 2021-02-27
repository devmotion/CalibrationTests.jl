using SafeTestsets

@safetestset "Binary trend tests" begin
    include("binary_trend.jl")
end
@safetestset "Consistency test" begin
    include("consistency.jl")
end

@safetestset "Asymptotic test" begin
    include("skce/asymptotic.jl")
end
@safetestset "Asymptotic block test" begin
    include("skce/asymptotic_block.jl")
end
@safetestset "Distribution-free tests" begin
    include("skce/distribution_free.jl")
end

@safetestset "Asymptotic CME test" begin include("cme.jl") end
