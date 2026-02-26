# =============================================================================
# Tests for model constructors, S3 methods, and public API
# =============================================================================

# --- Integration tests via public API ----------------------------------------

test_that("CreateFullKRR returns a fitted FullKRR model with metrics", {
  set.seed(10)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "pcg", iters = 30)
  expect_s3_class(fit, "FullKRR")
  expect_equal(fit$n, n)
  expect_false(fit$inducing)
  expect_equal(length(fit$w), n)
  expect_true(!is.null(fit$metrics))
  expect_true("rel_residual" %in% names(fit$metrics))
  expect_true(fit$metrics$rel_residual < 0.1)
})

test_that("CreateFullKRR uses default w0 of zeros", {
  set.seed(11)
  n <- 20
  d <- 2
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "pcg", iters = 5)
  expect_s3_class(fit, "FullKRR")
  expect_equal(length(fit$w), n)
  expect_true(!is.null(fit$metrics))
})

test_that("CreateFullKRR with custom w0", {
  set.seed(12)
  n <- 20
  d <- 2
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)
  w0 <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "pcg", iters = 5, w0 = w0)
  expect_s3_class(fit, "FullKRR")
  expect_equal(length(fit$w), n)
})

test_that("CreateFullKRR rejects invalid optimizer", {
  set.seed(13)
  n <- 20
  d <- 2
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  expect_error(
    CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                   optimizer = "bogus", iters = 5),
    "optimizer must be one of"
  )
})

test_that("CreateFullKRR verbose mode produces history", {
  set.seed(14)
  n <- 30
  d <- 2
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "pcg", iters = 10, verbose = TRUE)
  expect_true(!is.null(fit$history))
  expect_true(length(fit$history) > 0)
})

test_that("CreateFullKRR classification task works", {
  set.seed(16)
  n <- 60
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sign(x[, 1])

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        task = "classification", optimizer = "pcg",
                        iters = 30)
  expect_s3_class(fit, "FullKRR")
  expect_true("rel_residual" %in% names(fit$metrics))
})

test_that("CreateFullKRR with Matern kernel works", {
  set.seed(17)
  n <- 30
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "matern", sigma = 1.0, nu = 1.5,
                        lambda = 0.1, optimizer = "pcg", iters = 10)
  expect_s3_class(fit, "FullKRR")
  expect_true(is.finite(fit$metrics$rel_residual))
})

test_that("CreateFullKRR with L1 Laplace kernel works", {
  set.seed(18)
  n <- 30
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "l1_laplace", sigma = 1.0,
                        lambda = 0.1, optimizer = "pcg", iters = 10)
  expect_s3_class(fit, "FullKRR")
  expect_true(is.finite(fit$metrics$rel_residual))
})

test_that("CreateInducingKRR returns a fitted InducingKRR model with metrics", {
  set.seed(19)
  n <- 50
  d <- 3
  m <- 10
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01, optimizer = "pcg",
                            iters = 30)
  expect_s3_class(fit, "InducingKRR")
  expect_true(fit$inducing)
  expect_equal(fit$m, m)
  expect_equal(length(fit$w), m)
  expect_true(!is.null(fit$metrics))
  expect_true("rel_residual" %in% names(fit$metrics))
  expect_true(fit$metrics$rel_residual < 0.5)
})

test_that("CreateInducingKRR uses default w0 of zeros", {
  set.seed(20)
  n <- 40
  d <- 3
  m <- 8
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.1, optimizer = "pcg",
                            iters = 5)
  expect_equal(length(fit$w), m)
  expect_true(!is.null(fit$metrics))
})

test_that("CreateInducingKRR rejects invalid optimizer", {
  set.seed(21)
  n <- 40
  d <- 3
  m <- 8
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  expect_error(
    CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                       sigma = 1.0, lambda = 0.1, optimizer = "invalid",
                       iters = 5),
    "optimizer must be one of"
  )
})

# --- predict.FullKRR ---------------------------------------------------------

test_that("predict.FullKRR returns correct predictions", {
  set.seed(30)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 1e-4,
                        optimizer = "pcg", iters = 30)
  x_new <- matrix(rnorm(5 * d), 5, d)
  preds <- predict(fit, x_new)
  expect_equal(length(preds), 5)
  expect_true(all(is.finite(preds)))

  # Verify against manual kernel computation
  kp <- list(type = "rbf", sigma = 1.0)
  K_new <- fasterKRR:::ComputeKernel(x_new, fit$x, kp)
  expected <- as.vector(K_new %*% fit$w)
  expect_equal(preds, expected, tolerance = 1e-12)
})

test_that("predict.FullKRR on training data recovers signal", {
  set.seed(31)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1])

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 1e-3,
                        optimizer = "pcg", iters = 5,
                        preconditioner = list(type = "newton",
                                              rho = "regularization"))
  train_preds <- predict(fit, x)
  expect_equal(train_preds, y, tolerance = 0.1)
})

test_that("predict.FullKRR works with a single new observation", {
  set.seed(32)
  n <- 30
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "pcg", iters = 10)
  x_single <- matrix(rnorm(d), 1, d)
  pred <- predict(fit, x_single)
  expect_equal(length(pred), 1)
  expect_true(is.finite(pred))
})

test_that("predict.FullKRR with Matern kernel matches manual computation", {
  set.seed(33)
  n <- 20
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "matern", sigma = 1.0, nu = 1.5,
                        lambda = 0.1, optimizer = "pcg", iters = 10)
  x_new <- matrix(rnorm(4 * d), 4, d)
  preds <- predict(fit, x_new)

  kp <- list(type = "matern", sigma = 1.0, nu = 1.5)
  K_new <- fasterKRR:::ComputeKernel(x_new, fit$x, kp)
  expected <- as.vector(K_new %*% fit$w)
  expect_equal(preds, expected, tolerance = 1e-10)
})

test_that("predict.FullKRR classification gives usable scores", {
  set.seed(34)
  n <- 60
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sign(x[, 1])

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        task = "classification", optimizer = "pcg",
                        iters = 30)
  x_new <- matrix(c(2, 0, 0, -2, 0, 0), 2, d, byrow = TRUE)
  scores <- predict(fit, x_new)
  # Point with x1=2 should get positive score, x1=-2 negative
  expect_true(scores[1] > 0)
  expect_true(scores[2] < 0)
})

# --- predict.InducingKRR -----------------------------------------------------

test_that("predict.InducingKRR returns correct predictions", {
  set.seed(40)
  n <- 50
  d <- 3
  m <- 10
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01, optimizer = "pcg",
                            iters = 30)
  x_new <- matrix(rnorm(6 * d), 6, d)
  preds <- predict(fit, x_new)
  expect_equal(length(preds), 6)
  expect_true(all(is.finite(preds)))

  # Verify against manual computation
  kp <- list(type = "rbf", sigma = 1.0)
  x_ind <- x[1:m, , drop = FALSE]
  K_new <- fasterKRR:::ComputeKernel(x_new, x_ind, kp)
  expected <- as.vector(K_new %*% fit$w)
  expect_equal(preds, expected, tolerance = 1e-12)
})

test_that("predict.InducingKRR works with a single new observation", {
  set.seed(41)
  n <- 40
  d <- 3
  m <- 8
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.1, optimizer = "pcg",
                            iters = 5)
  x_single <- matrix(rnorm(d), 1, d)
  pred <- predict(fit, x_single)
  expect_equal(length(pred), 1)
  expect_true(is.finite(pred))
})

test_that("predict.InducingKRR on training data uses inducing points", {
  set.seed(42)
  n <- 40
  d <- 3
  m <- 8
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.1, optimizer = "pcg",
                            iters = 5)
  preds <- predict(fit, x)

  kp <- list(type = "rbf", sigma = 1.0)
  x_ind <- x[1:m, , drop = FALSE]
  K_ni <- fasterKRR:::ComputeKernel(x, x_ind, kp)
  expect_equal(preds, as.vector(K_ni %*% fit$w), tolerance = 1e-12)
})

# --- Unit tests for internal builder functions via ::: -----------------------

test_that("build_full_model creates correct structure", {
  set.seed(50)
  n <- 20
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)
  kp <- list(type = "rbf", sigma = 1.0)

  model <- fasterKRR:::build_full_model(x, y, kernel_params = kp,
                                       k_tr_needed = TRUE,
                                       lambda = 0.1, task = "regression",
                                       w0 = rep(0, n))
  expect_s3_class(model, "FullKRR")
  expect_equal(model$n, n)
  expect_false(model$inducing)
  expect_equal(dim(model$K), c(n, n))
  expect_equal(length(model$w), n)
  expect_equal(model$lambda, 0.1)
  expect_equal(model$y, y)
  expect_equal(model$y_norm, sqrt(sum(y^2)))
})

test_that("build_inducing_model creates correct structure", {
  set.seed(52)
  n <- 30
  d <- 3
  m <- 5
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)
  kp <- list(type = "rbf", sigma = 1.0)

  model <- fasterKRR:::build_inducing_model(x, y, kernel_params = kp,
                                           k_nm_needed = TRUE,
                                           inducing_pts = 1:m,
                                           lambda = 0.1, task = "regression",
                                           w0 = rep(0, m))
  expect_s3_class(model, "InducingKRR")
  expect_true(model$inducing)
  expect_equal(model$m, m)
  expect_equal(dim(model$K_mm), c(m, m))
  expect_equal(dim(model$K_nm), c(n, m))
  expect_equal(length(model$w), m)
  expect_equal(model$lambda, 0.1)
  expect_equal(model$y, y)
  expect_true(!is.null(model$K_nmTy))
  expect_true(!is.null(model$K_nmTy_norm))
  expect_true(!is.null(model$ym_scaled))
})

# --- Unit tests for FullKRR LinOp via ::: ------------------------------------

test_that("FullKRR LinOp computes K*v + lambda*v", {
  set.seed(60)
  n <- 10
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)
  kp <- list(type = "rbf", sigma = 1.0)

  model <- fasterKRR:::build_full_model(x, y, kernel_params = kp,
                                       k_tr_needed = TRUE,
                                       lambda = 0.5, task = "regression",
                                       w0 = rep(0, n))
  v <- rnorm(n)
  result <- fasterKRR:::LinOp(model, v)
  expected <- as.vector(model$K %*% v) + 0.5 * v
  expect_equal(result, expected, tolerance = 1e-10)
})

test_that("FullKRR LinOp works with matrix input", {
  set.seed(61)
  n <- 10
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)
  kp <- list(type = "rbf", sigma = 1.0)

  model <- fasterKRR:::build_full_model(x, y, kernel_params = kp,
                                       k_tr_needed = TRUE,
                                       lambda = 0.1, task = "regression",
                                       w0 = rep(0, n))
  V <- matrix(rnorm(n * 2), n, 2)
  result <- fasterKRR:::LinOp(model, V)
  expected <- model$K %*% V + 0.1 * V
  expect_equal(result, expected, tolerance = 1e-10)
})

# --- Unit tests for InducingKRR LinOp via ::: --------------------------------

test_that("InducingKRR LinOp computes K_nm^T K_nm v + lambda K_mm v", {
  set.seed(62)
  n <- 20
  m <- 5
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)
  kp <- list(type = "rbf", sigma = 1.0)

  model <- fasterKRR:::build_inducing_model(x, y, kernel_params = kp,
                                           k_nm_needed = TRUE,
                                           inducing_pts = 1:m,
                                           lambda = 0.1, task = "regression",
                                           w0 = rep(0, m))
  v <- rnorm(m)
  result <- fasterKRR:::LinOp(model, v)
  K_nm_v <- model$K_nm %*% v
  expected <- as.vector(crossprod(model$K_nm, K_nm_v)) +
    0.1 * as.vector(model$K_mm %*% v)
  expect_equal(result, expected, tolerance = 1e-10)
})

# --- Unit tests for ComputeMetrics via ::: -----------------------------------

test_that("FullKRR ComputeMetrics returns train metrics", {
  set.seed(70)
  n <- 20
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)
  kp <- list(type = "rbf", sigma = 1.0)

  model <- fasterKRR:::build_full_model(x, y, kernel_params = kp,
                                       k_tr_needed = TRUE,
                                       lambda = 0.1, task = "regression",
                                       w0 = rep(0, n))
  metrics <- fasterKRR:::ComputeMetrics(model, model$w)
  expect_true("rel_residual" %in% names(metrics))
  expect_true("train_loss" %in% names(metrics))
})

# --- Field name tests --------------------------------------------------------

test_that("Fitted model uses correct field names (y, lambda, etc.)", {
  set.seed(80)
  n <- 30
  d <- 2
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "pcg", iters = 5)
  expect_equal(fit$y, y)
  expect_equal(fit$lambda, 0.01)
  expect_true(!is.null(fit$y_norm))

  # Old names absent (use [[ to avoid $ partial matching)
  expect_null(fit[["b"]])
  expect_null(fit[["lambd"]])
  expect_null(fit[["b_norm"]])
})

test_that("InducingKRR model uses new field names", {
  set.seed(81)
  n <- 40
  d <- 3
  m <- 8
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01, optimizer = "pcg",
                            iters = 5)
  expect_equal(fit$y, y)
  expect_equal(fit$lambda, 0.01)
  expect_true(!is.null(fit$y_norm))
  expect_true(!is.null(fit$ym_scaled))
  expect_true(!is.null(fit$K_nmTy))
  expect_true(!is.null(fit$K_nmTy_norm))

  # Old names absent (use [[ to avoid $ partial matching)
  expect_null(fit[["b"]])
  expect_null(fit[["lambd"]])
  expect_null(fit[["bm_scaled"]])
  expect_null(fit[["K_nmTb"]])
  expect_null(fit[["K_nmTb_norm"]])
})
