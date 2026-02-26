## ---- Fast Kernel Ridge Regression Demo ----
##
## This demo walks through the main features of the fasterKRR package:
##   1. Basic regression with PCG
##   2. Prediction on new data
##   3. Preconditioned PCG (Newton, Nystrom)
##   4. EigenPro2 stochastic optimizer
##   5. ASkotch block coordinate descent
##   6. Inducing-point KRR with PCG
##   7. Classification
##   8. Convergence comparison of different solver configurations
##   9. Online updates with UpdateModel

library(fasterKRR)

cat("=== fasterKRR Demo ===\n\n")

# --------------------------------------------------------------------------- #
# Shared synthetic data
# --------------------------------------------------------------------------- #
set.seed(123)
n <- 100
d <- 5

x_train <- matrix(rnorm(n * d), n, d)
y_train <- sin(x_train[, 1]) + 0.1 * rnorm(n)

x_test <- matrix(rnorm(30 * d), 30, d)
y_test <- sin(x_test[, 1]) + 0.1 * rnorm(30)

# --------------------------------------------------------------------------- #
# 1. Basic regression with PCG
# --------------------------------------------------------------------------- #
cat("--- 1. Basic Regression with PCG ---\n\n")

fit_pcg <- CreateFullKRR(
  x       = x_train,
  y       = y_train,
  kernel  = "rbf",
  sigma   = 1.0,
  lambda  = 0.01,
  optimizer = "pcg",
  iters   = 30,
  x_test  = x_test,
  y_test  = y_test,
  verbose = TRUE
)

cat("\nRelative residual:", format(fit_pcg$metrics$rel_residual, digits = 4), "\n")
cat("Test RMSE:        ", format(fit_pcg$metrics$test_rmse, digits = 4), "\n")
cat("Test MAE:         ", format(fit_pcg$metrics$test_mae, digits = 4), "\n")

readline("Press [Enter] to continue to prediction...")

# --------------------------------------------------------------------------- #
# 2. Prediction on new data
# --------------------------------------------------------------------------- #
cat("\n--- 2. Prediction on New Data ---\n\n")

# Predict on the test set that was already used during fitting
y_pred <- predict(fit_pcg, x_test)
cat("First 5 predictions:", round(y_pred[1:5], 4), "\n")
cat("First 5 actual:     ", round(y_test[1:5], 4), "\n")
cat("Correlation:        ", round(cor(y_pred, y_test), 4), "\n")

# Also works on completely new, unseen points
x_new <- matrix(rnorm(5 * d), 5, d)
cat("\nPredictions for 5 new points:", round(predict(fit_pcg, x_new), 4), "\n")

readline("Press [Enter] to continue to preconditioned PCG...")

# --------------------------------------------------------------------------- #
# 3. Preconditioned PCG (Newton, Nystrom)
# --------------------------------------------------------------------------- #
cat("\n--- 3. Preconditioned PCG ---\n\n")

# Newton preconditioner -- converges extremely fast on small problems
fit_newton <- CreateFullKRR(
  x       = x_train,
  y       = y_train,
  kernel  = "rbf",
  sigma   = 1.0,
  lambda  = 0.01,
  optimizer = "pcg",
  preconditioner = list(type = "newton", rho = "regularization"),
  iters   = 5,
  x_test  = x_test,
  y_test  = y_test,
  verbose = TRUE
)

cat("\nNewton PCG (5 iters):\n")
cat("  Relative residual:", format(fit_newton$metrics$rel_residual, digits = 4), "\n")
cat("  Test RMSE:        ", format(fit_newton$metrics$test_rmse, digits = 4), "\n")

# Nystrom preconditioner -- good for larger problems where Newton is too costly
fit_nystrom <- CreateFullKRR(
  x       = x_train,
  y       = y_train,
  kernel  = "rbf",
  sigma   = 1.0,
  lambda  = 0.01,
  optimizer = "pcg",
  preconditioner = list(type = "nystrom", r = 20, rho = "regularization"),
  iters   = 10,
  x_test  = x_test,
  y_test  = y_test,
  verbose = TRUE
)

cat("\nNystrom PCG (10 iters):\n")
cat("  Relative residual:", format(fit_nystrom$metrics$rel_residual, digits = 4), "\n")
cat("  Test RMSE:        ", format(fit_nystrom$metrics$test_rmse, digits = 4), "\n")

readline("Press [Enter] to continue to EigenPro2...")

# --------------------------------------------------------------------------- #
# 4. EigenPro2 optimizer
# --------------------------------------------------------------------------- #
cat("\n--- 4. EigenPro2 Optimizer ---\n\n")

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
  x_test   = x_test,
  y_test   = y_test,
  verbose  = TRUE
)

cat("\nEigenPro2 (200 iters):\n")
cat("  Relative residual:", format(fit_ep2$metrics$rel_residual, digits = 4), "\n")
cat("  Test RMSE:        ", format(fit_ep2$metrics$test_rmse, digits = 4), "\n")

readline("Press [Enter] to continue to ASkotch...")

# --------------------------------------------------------------------------- #
# 5. ASkotch optimizer
# --------------------------------------------------------------------------- #
cat("\n--- 5. ASkotch Block Coordinate Descent ---\n\n")

fit_ask <- CreateFullKRR(
  x        = x_train,
  y        = y_train,
  kernel   = "rbf",
  sigma    = 1.0,
  lambda   = 0.01,
  optimizer = "askotch",
  iters    = 100,
  B        = 5,
  x_test   = x_test,
  y_test   = y_test,
  verbose  = TRUE
)

cat("\nASkotch (100 iters, B = 5):\n")
cat("  Relative residual:", format(fit_ask$metrics$rel_residual, digits = 4), "\n")
cat("  Test RMSE:        ", format(fit_ask$metrics$test_rmse, digits = 4), "\n")

readline("Press [Enter] to continue to InducingKRR...")

# --------------------------------------------------------------------------- #
# 6. Inducing-point KRR with PCG
# --------------------------------------------------------------------------- #
cat("\n--- 6. Inducing-Point KRR ---\n\n")

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
cat("InducingKRR predictions (first 5):", round(y_pred_ind[1:5], 4), "\n")
cat("Actual values (first 5):          ", round(y_test[1:5], 4), "\n")
cat("Correlation with truth:           ", round(cor(y_pred_ind, y_test), 4), "\n")

readline("Press [Enter] to continue to classification...")

# --------------------------------------------------------------------------- #
# 7. Classification
# --------------------------------------------------------------------------- #
cat("\n--- 7. Classification ---\n\n")

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
  x_test   = x_test,
  y_test   = y_cls_test,
  verbose  = TRUE
)

cat("\nClassification accuracy:", fit_cls$metrics$test_acc, "\n")

readline("Press [Enter] to continue to convergence comparison...")

# --------------------------------------------------------------------------- #
# 8. Convergence comparison
# --------------------------------------------------------------------------- #
cat("\n--- 8. Convergence Comparison ---\n\n")

configs <- list(
  "PCG (no precond)" = list(optimizer = "pcg", iters = 20),
  "Newton PCG"       = list(optimizer = "pcg", iters = 20,
                            preconditioner = list(type = "newton",
                                                  rho = "regularization")),
  "Nystrom PCG"      = list(optimizer = "pcg", iters = 20,
                            preconditioner = list(type = "nystrom", r = 20,
                                                  rho = "regularization"))
)

n_iters <- 20
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
      optimizer = cfg$optimizer,
      iters    = it,
      x_test   = x_test,
      y_test   = y_test,
      verbose  = FALSE
    )
    if (!is.null(cfg$preconditioner)) {
      args$preconditioner <- cfg$preconditioner
    }
    fit_tmp <- do.call(CreateFullKRR, args)
    results[it, cfg_name] <- fit_tmp$metrics$rel_residual
  }
}

cat("Relative residual by iteration:\n")
cat(sprintf("%-5s  %-18s %-18s %-18s\n",
            "Iter", names(configs)[1], names(configs)[2], names(configs)[3]))
for (i in c(1, 2, 5, 10, 15, 20)) {
  cat(sprintf("%-5d  %-18.8f %-18.8f %-18.8f\n",
              i, results[i, 1], results[i, 2], results[i, 3]))
}

readline("Press [Enter] to continue to online updates...")

# --------------------------------------------------------------------------- #
# 9. Online updates with UpdateModel
# --------------------------------------------------------------------------- #
cat("\n--- 9. Online Updates with UpdateModel ---\n\n")

# UpdateModel returns a *new* model -- the original is never modified.
# This lets you keep snapshots or branch from any point in the stream.

# Fit an InducingKRR model on 70 points, then stream 30 more
fit_base <- CreateInducingKRR(
  x            = x_train[1:70, ],
  y            = y_train[1:70],
  inducing_pts = 1:m_ind,
  kernel       = "rbf",
  sigma        = 1.0,
  lambda       = 0.01,
  optimizer    = "pcg",
  iters        = 30
)

y_pred_before <- predict(fit_base, x_test)
rmse_before <- sqrt(mean((y_pred_before - y_test)^2))
cat("Before streaming (n = 70):\n")
cat("  Test RMSE:", format(rmse_before, digits = 4), "\n\n")

# Stream 30 new observations -- fit_base stays unchanged
fit_streamed <- fit_base
for (i in 71:100) {
  fit_streamed <- UpdateModel(fit_streamed, x_train[i, ], y_train[i],
                               recompute_metrics = FALSE)
}

y_pred_after <- predict(fit_streamed, x_test)
rmse_after <- sqrt(mean((y_pred_after - y_test)^2))
cat("After streaming 30 points (n = 100):\n")
cat("  Test RMSE:      ", format(rmse_after, digits = 4), "\n")
cat("  Rank-1 updates: ", fit_streamed$n_updates, "\n")
cat("  RMSE improved:  ", rmse_after < rmse_before, "\n")
cat("  Original n:     ", fit_base$n, " (unchanged)\n")

cat("\n=== Demo Complete ===\n")
