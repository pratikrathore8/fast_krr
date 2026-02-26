# =============================================================================
# Tests for UpdateModel (online learning via Sherman-Morrison)
#
# UpdateModel returns a *new* model object; it does not mutate the original.
# This means you can keep old snapshots:
#   model2 <- UpdateModel(model, x_new, y_new)   # model is unchanged
# Or overwrite to advance a stream:
#   model  <- UpdateModel(model, x_new, y_new)
# =============================================================================

# --- Helper: fit a base InducingKRR model -----------------------------------

make_base_model <- function(seed = 42, n = 80, d = 3, m = 10,
                             kernel = "rbf", sigma = 1.0, lambda = 0.01) {
  set.seed(seed)
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)
  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = kernel,
                            sigma = sigma, lambda = lambda,
                            optimizer = "pcg", iters = 30)
  list(fit = fit, d = d)
}

# 1. Basic: returns valid InducingKRR with updated fields --------------------

test_that("UpdateModel returns a new InducingKRR with updated fields", {
  base <- make_base_model()
  fit <- base$fit
  d <- base$d
  n_before <- fit$n

  set.seed(99)
  x_new <- rnorm(d)
  y_new <- sin(x_new[1]) + 0.1 * rnorm(1)

  fit2 <- UpdateModel(fit, x_new, y_new)

  # The returned model has the new data
  expect_s3_class(fit2, "InducingKRR")
  expect_equal(fit2$n, n_before + 1L)
  expect_equal(nrow(fit2$x), n_before + 1L)
  expect_equal(length(fit2$y), n_before + 1L)
  expect_equal(nrow(fit2$K_nm), n_before + 1L)
  expect_equal(length(fit2$w), fit$m)
  expect_true(!is.null(fit2$P_inv))
  expect_equal(fit2$n_updates, 1L)

  # The original model is unchanged
  expect_equal(fit$n, n_before)
  expect_equal(nrow(fit$x), n_before)
  expect_equal(length(fit$y), n_before)
  expect_null(fit$P_inv)
  expect_null(fit$n_updates)
})

# 2. Correctness: updated w matches direct solve on augmented data -----------

test_that("UpdateModel w matches direct solve on augmented data", {
  base <- make_base_model()
  fit <- base$fit
  d <- base$d

  set.seed(100)
  x_new <- rnorm(d)
  y_new <- sin(x_new[1]) + 0.1 * rnorm(1)

  fit2 <- UpdateModel(fit, x_new, y_new)

  # Direct solve on augmented data
  x_aug <- rbind(fit$x, matrix(x_new, nrow = 1))
  y_aug <- c(fit$y, y_new)
  x_ind <- fit$x[fit$inducing_pts, , drop = FALSE]
  K_nm_aug <- fasterKRR:::ComputeKernel(x_aug, x_ind, fit$kernel_params)
  K_mm <- fit$K_mm
  A <- crossprod(K_nm_aug) + fit$lambda * K_mm
  b <- as.vector(crossprod(K_nm_aug, y_aug))
  w_direct <- as.vector(solve(A, b))

  expect_equal(fit2$w, w_direct, tolerance = 1e-8)
})

# 3. Multiple updates: 5 sequential updates match direct solve ---------------

test_that("Multiple sequential updates match direct solve", {
  base <- make_base_model()
  original <- base$fit
  d <- base$d

  set.seed(200)
  n_new <- 5
  x_news <- matrix(rnorm(n_new * d), n_new, d)
  y_news <- sin(x_news[, 1]) + 0.1 * rnorm(n_new)

  # Build up a chain of updated models; original stays untouched
  updated <- original
  for (i in seq_len(n_new)) {
    updated <- UpdateModel(updated, x_news[i, ], y_news[i])
  }

  # Direct solve on full augmented data
  x_aug <- rbind(original$x, x_news)
  y_aug <- c(original$y, y_news)
  x_ind <- original$x[original$inducing_pts, , drop = FALSE]
  K_nm_aug <- fasterKRR:::ComputeKernel(x_aug, x_ind, original$kernel_params)
  K_mm <- original$K_mm
  A <- crossprod(K_nm_aug) + original$lambda * K_mm
  b <- as.vector(crossprod(K_nm_aug, y_aug))
  w_direct <- as.vector(solve(A, b))

  expect_equal(updated$w, w_direct, tolerance = 1e-8)
  expect_equal(updated$n, original$n + n_new)
  expect_equal(updated$n_updates, n_new)

  # Original is still untouched
  expect_equal(original$n, base$fit$n)
  expect_null(original$P_inv)
})

# 4. Lazy P: P_inv is NULL before first call, present after ------------------

test_that("P_inv is NULL before first UpdateModel, present after", {
  base <- make_base_model()
  fit <- base$fit
  d <- base$d

  expect_null(fit$P_inv)

  set.seed(101)
  x_new <- rnorm(d)
  y_new <- 0.5

  fit2 <- UpdateModel(fit, x_new, y_new)

  # New model has P_inv; original still does not
  expect_true(!is.null(fit2$P_inv))
  expect_equal(nrow(fit2$P_inv), fit$m)
  expect_equal(ncol(fit2$P_inv), fit$m)
  expect_null(fit$P_inv)
})

# 5. FullKRR error -----------------------------------------------------------

test_that("UpdateModel.FullKRR gives informative error", {
  set.seed(50)
  n <- 30; d <- 2
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)
  fit_full <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                             optimizer = "pcg", iters = 5)

  expect_error(
    UpdateModel(fit_full, rnorm(d), 0.5),
    "Online updates are not supported for FullKRR"
  )
})

# 6. x_new wrong dimension --------------------------------------------------

test_that("UpdateModel errors on x_new with wrong dimension", {
  base <- make_base_model()
  fit <- base$fit

  expect_error(
    UpdateModel(fit, rnorm(base$d + 1), 0.5),
    "x_new must have length"
  )
  expect_error(
    UpdateModel(fit, rnorm(1), 0.5),
    "x_new must have length"
  )
})

# 7. y_new not scalar --------------------------------------------------------

test_that("UpdateModel errors when y_new is not scalar", {
  base <- make_base_model()
  fit <- base$fit
  d <- base$d

  expect_error(
    UpdateModel(fit, rnorm(d), c(1, 2)),
    "y_new must be a single numeric value"
  )
  expect_error(
    UpdateModel(fit, rnorm(d), "a"),
    "y_new must be a single numeric value"
  )
})

# 8. recompute=TRUE refreshes P_inv and resets counter -----------------------

test_that("recompute=TRUE refreshes P_inv and resets counter", {
  base <- make_base_model()
  original <- base$fit
  d <- base$d

  set.seed(300)
  # Do 3 updates, building up a chain
  updated <- original
  for (i in 1:3) {
    updated <- UpdateModel(updated, rnorm(d), rnorm(1))
  }
  expect_equal(updated$n_updates, 3L)

  # Do one more with recompute
  x_new <- rnorm(d)
  y_new <- rnorm(1)
  refreshed <- UpdateModel(updated, x_new, y_new, recompute = TRUE)

  # Counter should be 1 (reset to 0, then +1 for this update)
  expect_equal(refreshed$n_updates, 1L)
  expect_true(!is.null(refreshed$P_inv))

  # The previous model still has its old counter
  expect_equal(updated$n_updates, 3L)
})

# 9. recompute_metrics=FALSE sets metrics to NULL ----------------------------

test_that("recompute_metrics=FALSE sets metrics to NULL", {
  base <- make_base_model()
  fit <- base$fit
  d <- base$d

  set.seed(400)
  fit2 <- UpdateModel(fit, rnorm(d), rnorm(1), recompute_metrics = FALSE)

  expect_null(fit2$metrics)
})

# 10. predict works correctly after update -----------------------------------

test_that("predict works correctly after UpdateModel", {
  base <- make_base_model()
  fit <- base$fit
  d <- base$d

  set.seed(500)
  x_new <- rnorm(d)
  y_new <- sin(x_new[1]) + 0.1 * rnorm(1)

  fit2 <- UpdateModel(fit, x_new, y_new)

  # Predict on test points
  x_test <- matrix(rnorm(5 * d), 5, d)
  preds <- predict(fit2, x_test)

  expect_equal(length(preds), 5)

  # Manually compute expected prediction
  x_ind <- fit$x[fit$inducing_pts, , drop = FALSE]
  K_test <- fasterKRR:::ComputeKernel(x_test, x_ind, fit$kernel_params)
  preds_manual <- as.vector(K_test %*% fit2$w)

  expect_equal(preds, preds_manual, tolerance = 1e-10)
})

# 11. Works with Matern kernel -----------------------------------------------

test_that("UpdateModel works with Matern kernel", {
  set.seed(600)
  n <- 60; d <- 3; m <- 8
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "matern",
                            sigma = 1.0, nu = 1.5, lambda = 0.01,
                            optimizer = "pcg", iters = 30)

  x_new <- rnorm(d)
  y_new <- sin(x_new[1])

  fit2 <- UpdateModel(fit, x_new, y_new)

  expect_s3_class(fit2, "InducingKRR")
  expect_equal(fit2$n, n + 1L)
  expect_equal(fit2$n_updates, 1L)

  # Verify correctness against direct solve
  x_aug <- rbind(fit$x, matrix(x_new, nrow = 1))
  y_aug <- c(fit$y, y_new)
  x_ind <- fit$x[fit$inducing_pts, , drop = FALSE]
  K_nm_aug <- fasterKRR:::ComputeKernel(x_aug, x_ind, fit$kernel_params)
  A <- crossprod(K_nm_aug) + fit$lambda * fit$K_mm
  b <- as.vector(crossprod(K_nm_aug, y_aug))
  w_direct <- as.vector(solve(A, b))

  expect_equal(fit2$w, w_direct, tolerance = 1e-8)

  # Original unchanged
  expect_equal(fit$n, n)
})

# 12. Stability threshold triggers message -----------------------------------

test_that("Stability threshold triggers message", {
  base <- make_base_model(n = 40, m = 5)
  d <- base$d

  set.seed(700)
  # Do threshold-1 updates silently
  updated <- base$fit
  for (i in 1:2) {
    updated <- UpdateModel(updated, rnorm(d), rnorm(1), stability_threshold = 3L)
  }
  expect_equal(updated$n_updates, 2L)

  # The 3rd update should produce a message
  expect_message(
    UpdateModel(updated, rnorm(d), rnorm(1), stability_threshold = 3L),
    "rank-1 updates accumulated"
  )
})

# 13. Original model is fully preserved (non-mutation guarantee) -------------

test_that("UpdateModel does not mutate the original model", {
  base <- make_base_model()
  original <- base$fit
  d <- base$d

  # Snapshot every field of the original
  orig_n      <- original$n
  orig_w      <- original$w
  orig_x      <- original$x
  orig_y      <- original$y
  orig_K_nm   <- original$K_nm
  orig_K_nmTy <- original$K_nmTy
  orig_y_norm <- original$y_norm
  orig_metrics <- original$metrics

  set.seed(800)
  x_new <- rnorm(d)
  y_new <- sin(x_new[1])

  # Create updated model in a separate variable
  updated <- UpdateModel(original, x_new, y_new)

  # The updated model is different
  expect_equal(updated$n, orig_n + 1L)
  expect_false(identical(updated$w, orig_w))

  # The original is completely unchanged

  expect_equal(original$n, orig_n)
  expect_identical(original$w, orig_w)
  expect_identical(original$x, orig_x)
  expect_identical(original$y, orig_y)
  expect_identical(original$K_nm, orig_K_nm)
  expect_identical(original$K_nmTy, orig_K_nmTy)
  expect_equal(original$y_norm, orig_y_norm)
  expect_null(original$P_inv)
  expect_null(original$n_updates)
})
