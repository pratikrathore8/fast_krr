## ----setup, include = FALSE---------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----pcg-regression-----------------------------------------------------------
library(fasterKRR)

set.seed(123)
n <- 100
d <- 5

# Generate synthetic data: y = sin(x1) + noise
x_train <- matrix(rnorm(n * d), n, d)
y_train <- sin(x_train[, 1]) + 0.1 * rnorm(n)

x_test <- matrix(rnorm(30 * d), 30, d)
y_test <- sin(x_test[, 1]) + 0.1 * rnorm(30)

# Fit with unpreconditioned PCG (30 iterations)
fit <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = 0.01,
  optimizer = "pcg",
  iters    = 30,
  verbose  = TRUE
)

# Inspect convergence metrics stored on the fitted object
cat("Relative residual:", round(fit$metrics$rel_residual, 6), "\n")

## ----predict-test-------------------------------------------------------------
y_pred <- predict(fit, x_test)

cat("First 5 predictions:", round(y_pred[1:5], 4), "\n")
cat("First 5 actual:     ", round(y_test[1:5], 4), "\n")
cat("Correlation:        ", round(cor(y_pred, y_test), 4), "\n")
cat("Test RMSE:          ", round(sqrt(mean((y_pred - y_test)^2)), 4), "\n")

## ----predict-new--------------------------------------------------------------
x_new <- matrix(rnorm(10 * d), 10, d)
y_new <- predict(fit, x_new)
cat("Predictions for 10 new points:", round(y_new, 4), "\n")

## ----predict-inducing---------------------------------------------------------
m_ind <- 15

fit_ind <- CreateInducingKRR(
  x            = x_train,
  y            = y_train,
  inducing_pts = 1:m_ind,
  kernel       = "rbf",
  sigma        = 1.0,
  lambda       = 0.01,
  optimizer    = "pcg",
  iters        = 30
)

y_pred_ind <- predict(fit_ind, x_test)
cat("Inducing KRR predictions (first 5):", round(y_pred_ind[1:5], 4), "\n")
cat("Correlation with truth:            ", round(cor(y_pred_ind, y_test), 4), "\n")

## ----precond-newton-----------------------------------------------------------
fit_newton <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = 0.01,
  optimizer = "pcg",
  preconditioner = list(type = "newton", rho = "regularization"),
  iters    = 5,
  verbose  = TRUE
)

cat("\nNewton PCG (5 iters) - rel_residual:",
    round(fit_newton$metrics$rel_residual, 8), "\n")

## ----precond-nystrom----------------------------------------------------------
fit_nystrom <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = 0.01,
  optimizer = "pcg",
  preconditioner = list(type = "nystrom", r = 20, rho = "regularization"),
  iters    = 10,
  verbose  = TRUE
)

cat("\nNystrom PCG (10 iters) - rel_residual:",
    round(fit_nystrom$metrics$rel_residual, 6), "\n")

## ----eigenpro2----------------------------------------------------------------
fit_ep2 <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = 0.01,
  optimizer = "eigenpro2",
  iters    = 200,
  block_sz = 80,
  r        = 10,
  verbose  = FALSE
)

cat("EigenPro2 (200 iters) - rel_residual:",
    round(fit_ep2$metrics$rel_residual, 6), "\n")

## ----askotch------------------------------------------------------------------
fit_ask <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = 0.01,
  optimizer = "askotch",
  iters    = 100,
  B        = 5,
  verbose  = FALSE
)

cat("ASkotch (100 iters, B = 5) - rel_residual:",
    round(fit_ask$metrics$rel_residual, 6), "\n")

## ----classification-----------------------------------------------------------
y_cls_train <- sign(x_train[, 1])
y_cls_test  <- sign(x_test[, 1])

fit_cls <- CreateFullKRR(
  x        = x_train,
  y        = y_cls_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = 0.01,
  task     = "classification",
  optimizer = "pcg",
  iters    = 30,
  verbose  = FALSE
)

cls_pred <- sign(predict(fit_cls, x_test))
cat("Classification accuracy:", mean(cls_pred == y_cls_test), "\n")

## ----comparison---------------------------------------------------------------
configs <- list(
  "PCG"        = list(),
  "Newton PCG" = list(preconditioner = list(type = "newton",
                                            rho = "regularization")),
  "Nystrom PCG" = list(preconditioner = list(type = "nystrom", r = 20,
                                             rho = "regularization"))
)

n_iters <- 10
results <- matrix(NA, n_iters, length(configs),
                  dimnames = list(NULL, names(configs)))

for (cfg_name in names(configs)) {
  cfg <- configs[[cfg_name]]
  for (it in 1:n_iters) {
    args <- list(
      x        = x_train,
      y        = y_train,
      kernel   = "rbf",
      sigma    = 1.0,
      lambda   = 0.01,
      optimizer = "pcg",
      iters    = it,
      verbose  = FALSE
    )
    if (!is.null(cfg$preconditioner)) {
      args$preconditioner <- cfg$preconditioner
    }
    fit_tmp <- do.call(CreateFullKRR, args)
    results[it, cfg_name] <- fit_tmp$metrics$rel_residual
  }
}

cat("Relative residual by iteration:\n\n")
cat(sprintf("%-5s  %-14s %-14s %-14s\n",
            "Iter", "PCG", "Newton PCG", "Nystrom PCG"))
for (i in c(1, 2, 3, 5, 8, 10)) {
  cat(sprintf("%-5d  %-14.8f %-14.8f %-14.8f\n",
              i, results[i, "PCG"],
              results[i, "Newton PCG"],
              results[i, "Nystrom PCG"]))
}

## ----auto-lambda--------------------------------------------------------------
fit_auto <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = NULL,
  optimizer = "pcg",
  iters    = 30,
  verbose  = TRUE
)

cat("Selected lambda:", fit_auto$lambda, "\n")
cat("Relative residual:", round(fit_auto$metrics$rel_residual, 6), "\n")

## ----auto-lambda-custom-------------------------------------------------------
fit_custom <- CreateFullKRR(
  x           = x_train,
  y           = y_train,
  kernel      = "rbf",
  sigma       = 1.0,
  lambda      = NULL,
  optimizer   = "pcg",
  iters       = 30,
  fast_cv     = FALSE,
  cv_folds    = 3L,
  lambda_grid = c(1e-4, 1e-3, 1e-2, 1e-1)
)

cat("Selected lambda:", fit_custom$lambda, "\n")

## ----auto-lambda-vector-------------------------------------------------------
fit_vec <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = c(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
  optimizer = "pcg",
  iters    = 30,
  verbose  = TRUE
)

cat("Selected lambda:", fit_vec$lambda, "\n")

## ----auto-lambda-inducing-----------------------------------------------------
fit_ind_auto <- CreateInducingKRR(
  x            = x_train,
  y            = y_train,
  inducing_pts = 1:m_ind,
  kernel       = "rbf",
  sigma        = 1.0,
  lambda       = NULL,
  optimizer    = "pcg",
  iters        = 30
)

cat("Inducing KRR selected lambda:", fit_ind_auto$lambda, "\n")

## ----online-fit---------------------------------------------------------------
# Fit an initial model on the first 80 points
fit_initial <- CreateInducingKRR(
  x            = x_train[1:80, ],
  y            = y_train[1:80],
  inducing_pts = 1:m_ind,
  kernel       = "rbf",
  sigma        = 1.0,
  lambda       = 0.01,
  optimizer    = "pcg",
  iters        = 30
)

cat("Before streaming: n =", fit_initial$n, "\n")

## ----online-stream------------------------------------------------------------
fit_streamed <- fit_initial
for (i in 81:100) {
  fit_streamed <- UpdateModel(fit_streamed, x_train[i, ], y_train[i],
                               recompute_metrics = FALSE)
}

cat("After streaming:  n =", fit_streamed$n, "\n")
cat("Rank-1 updates:  ", fit_streamed$n_updates, "\n")
cat("Original still:   n =", fit_initial$n, "\n")

## ----online-compare-----------------------------------------------------------
y_pred_before <- predict(fit_initial, x_test)
y_pred_after  <- predict(fit_streamed, x_test)

cat("RMSE before streaming:", round(sqrt(mean((y_pred_before - y_test)^2)), 4), "\n")
cat("RMSE after streaming: ", round(sqrt(mean((y_pred_after  - y_test)^2)), 4), "\n")

