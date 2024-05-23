function get_density_estimators(X, y;batchsize=16)
    snne = SieveNeuralNetworkEstimator(X, y; 
        hidden_sizes_candidates=[(20,), (20, 20)], 
        batchsize=batchsize
    )
    glm = GLMEstimator(X, y)
    return [snne, glm]
end