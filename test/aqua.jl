@testset "Aqua" begin
    # Test ambiguities separately without Base and Core
    # Ref: https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_all(CalibrationTests; ambiguities=false)
    Aqua.test_ambiguities([CalibrationTests])
end
