module TestResultsAggregation

using Test
using Simulations
using TMLE
using Statistics

@testset "Test quality measures" begin
    Ψ = JointEstimand(
        ATE(
            outcome = "Y",
            treatment_values = (rs3129716 = (case = "TC", control = "TT"),), 
            treatment_confounders = (:PC1, :PC2, :PC3, :PC4, :PC5, :PC6),
            outcome_extra_covariates = ("Age-Assessment", "Genetic-Sex")
        ),
        ATE(
            outcome = "Y",
            treatment_values = (rs3129716 = (case = "CC", control = "TC"),), 
            treatment_confounders = (:PC1, :PC2, :PC3, :PC4, :PC5, :PC6),
            outcome_extra_covariates = ("Age-Assessment", "Genetic-Sex")
        )
    )
    # First composed estimate
    estimates_1 = (TMLE.TMLEstimate(
            estimand = Ψ.args[1], 
            estimate = 0.012631690665702362, 
            std = 0.5366203374335045, 
            n = 100000, 
            IC = Float64[]
        ), 
        TMLE.TMLEstimate(
            estimand = Ψ.args[2], 
            estimate = 0.015820922887731858, 
            std = 1.9251498985147422, 
            n = 100000, 
            IC = Float64[]
        )
    )
    cov_1 = [0.287961  -0.227985
            -0.227985   3.7062]
    Ψ̂₁ = TMLE.JointEstimate(
        estimand=Ψ,
        estimates=estimates_1,
        cov=cov_1,
        n=estimates_1[1].n
    )
    # Second composed estimate
    estimates_2 = (TMLE.TMLEstimate(
            estimand = Ψ.args[1], 
            estimate = 0.002631690665702362, 
            std = 0.4366203374335045, 
            n = 100000, 
            IC = Float64[]
        ), 
        TMLE.TMLEstimate(
            estimand = Ψ.args[2], 
            estimate = 0.000820922887731858, 
            std = 1.9251498985147422, 
            n = 100000, 
            IC = Float64[]
        )
    )
    cov_2 = [0.287961  -0.227985
            -0.227985   3.7062]
    Ψ̂₂ = TMLE.JointEstimate(
        estimand=Ψ,
        estimates=estimates_2,
        cov=cov_2,
        n=estimates_2[1].n
    )
    estimates_1D = [Ψ̂₁.estimates[1], Ψ̂₂.estimates[1]]
    estimates_nD = [Ψ̂₁, Ψ̂₂]

    # Coverage
    ## Coverage 1D
    @test confint(significance_test(estimates_1D[1])) == (0.00930570421499154, 0.015957677116413185)
    @test confint(significance_test(estimates_1D[2])) == (-7.449325075900744e-5, 0.0053378745821637315)
    @test Simulations.covers(Ψ̂₁.estimates[1], 0) === false
    @test Simulations.covers(Ψ̂₁.estimates[1], 0.01) === true
    @test Simulations.covers.(estimates_1D, [0, -1e-6]) == [false, true]
    @test Simulations.covers.(estimates_1D, 0.) == [false, true]
    @test Simulations.covers.(estimates_1D, 0.01) == [true, false]
    @test Simulations.mean_coverage(estimates_1D, 0.01) == 0.5
    @test Simulations.mean_coverage(estimates_1D, -0.01) == 0.
    ## Coverage nD
    ### Ψ̂₁ does NOT cover [0, 0]
    @test pvalue(significance_test(Ψ̂₁, [0., 0.])) < 1e-15
    @test Simulations.covers(Ψ̂₁, [0., 0.]) === false
    ### Ψ̂₁ covers [0.012, 0.015]
    @test pvalue(significance_test(Ψ̂₁, [0.012, 0.015])) > 0.9
    @test Simulations.covers(Ψ̂₁, [0.012, 0.015]) === true
    ### Ψ̂₂ does NOT cover [-1., 2.]
    @test pvalue(significance_test(Ψ̂₂, [-1., 2.])) < 1e-15
    @test Simulations.covers(Ψ̂₂, [-1., 2.]) === false
    ### Ψ̂₂ covers [0, 0]
    @test pvalue(significance_test(Ψ̂₂, [0, 0])) > 0.2
    @test Simulations.covers(Ψ̂₂, [0, 0]) === true
    ### Vector
    @test Simulations.covers.(estimates_nD, [[0, 0], [-1., 2.]]) == [false, false]
    @test Simulations.covers.(estimates_nD, [[0.012, 0.015], [-1., 2.]]) == [true, false]
    @test Simulations.mean_coverage(estimates_nD, [0, 0]) == 0.5
    @test Simulations.mean_coverage(estimates_nD, [-1., 2.]) == 0.

    # Bias
    ## Bias 1D
    truth = -1.
    @test Simulations.bias(estimates_1D[1], truth) == estimates_1D[1].estimate - truth
    @test Simulations.bias(estimates_1D[1], estimates_1D[1].estimate) == 0
    @test Simulations.bias.(estimates_1D, truth) == [estimates_1D[1].estimate - truth, estimates_1D[2].estimate - truth]
    @test Simulations.mean_bias(estimates_1D, truth) == (estimates_1D[1].estimate + estimates_1D[2].estimate -2truth)/2
    ## Bias nD
    @test Simulations.bias(Ψ̂₁, [0, -1]) == [TMLE.estimate(Ψ̂₁)[1], TMLE.estimate(Ψ̂₁)[2] + 1]
    @test Simulations.bias(Ψ̂₁, TMLE.estimate(Ψ̂₁)) == [0, 0]
    @test Simulations.mean_bias(estimates_nD, [0, 0]) == mean(Simulations.bias.(estimates_nD, [[0, 0]]))

    # Variance
    @test Simulations.variance(Ψ̂₁) == (Ψ̂₁.cov[1, 1] + Ψ̂₁.cov[2, 2]) / Ψ̂₁.n
    @test Simulations.variance(Ψ̂₁.estimates[1]) == Ψ̂₁.estimates[1].std^2 / Ψ̂₁.estimates[1].n

    @test Simulations.mean_variance(estimates_nD) == (Simulations.variance(Ψ̂₁) + Simulations.variance(Ψ̂₂)) / 2
    @test Simulations.mean_variance(estimates_1D) == (Simulations.variance(Ψ̂₁.estimates[1]) + Simulations.variance(Ψ̂₂.estimates[1])) / 2

end

end

true