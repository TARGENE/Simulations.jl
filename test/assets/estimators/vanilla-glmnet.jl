resampling = JointStratifiedCV(patterns=[r"^rs[0-9]+"], resampling=StratifiedCV(nfolds=3))

default_models = TMLE.default_models(
  # For the estimation of E[Y|W, T]: continuous outcome
  Q_continuous = Pipeline(
    RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
    GLMNetRegressor(resampling=resampling),
    cache = false
  ),
  # For the estimation of E[Y|W, T]: binary outcome
  Q_binary = Pipeline(
    RestrictedInteractionTransformer(order=2, primary_variables_patterns=[r"^rs[0-9]+"]),
    GLMNetClassifier(resampling=resampling),
    cache = false
  ),
  # For the estimation of p(T| W)
  G = GLMNetClassifier(resampling=resampling),
)

ESTIMATORS = (
  wTMLE_GLMNET = TMLEE(models=default_models, weighted=true),
  TMLE_GLMNET = TMLEE(models=default_models, weighted=false),
  OSE_GLMNET = OSE(models=default_models)
)