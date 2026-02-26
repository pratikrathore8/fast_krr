# =============================================================================
# Tests for automatic lambda selection via CVST cross-validation
# =============================================================================

test_that("CreateFullKRR with lambda = NULL auto-selects a positive lambda", {
  set.seed(100)
  n <- 60
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = NULL,
                        optimizer = "pcg", iters = 30)
  expect_s3_class(fit, "FullKRR")
  expect_true(fit$lambda > 0)
  expect_true(is.finite(fit$lambda))
  expect_true(!is.null(fit$metrics))
  expect_true(is.finite(fit$metrics$rel_residual))

  # Model should still produce valid predictions
  x_new <- matrix(rnorm(5 * d), 5, d)
  preds <- predict(fit, x_new)
  expect_equal(length(preds), 5)
  expect_true(all(is.finite(preds)))
})

test_that("CreateInducingKRR with lambda = NULL auto-selects a positive lambda", {
  set.seed(101)
  n <- 80
  d <- 3
  m <- 15
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = NULL, optimizer = "pcg",
                            iters = 30)
  expect_s3_class(fit, "InducingKRR")
  expect_true(fit$lambda > 0)
  expect_true(is.finite(fit$lambda))
  expect_true(!is.null(fit$metrics))

  x_new <- matrix(rnorm(5 * d), 5, d)
  preds <- predict(fit, x_new)
  expect_equal(length(preds), 5)
  expect_true(all(is.finite(preds)))
})

test_that("fast_cv = FALSE (standard k-fold CV) works for FullKRR", {
  set.seed(102)
  n <- 60
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = NULL,
                        optimizer = "pcg", iters = 20, fast_cv = FALSE,
                        cv_folds = 3L)
  expect_s3_class(fit, "FullKRR")
  expect_true(fit$lambda > 0)
  expect_true(is.finite(fit$lambda))
})

test_that("Custom lambda_grid is respected", {
  set.seed(103)
  n <- 60
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  custom_grid <- c(0.01, 0.1, 1.0)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = NULL,
                        optimizer = "pcg", iters = 20,
                        lambda_grid = custom_grid)
  expect_s3_class(fit, "FullKRR")
  expect_true(fit$lambda %in% custom_grid)
})

test_that("Numeric lambda still works as before (no regression)", {
  set.seed(104)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "pcg", iters = 30)
  expect_s3_class(fit, "FullKRR")
  expect_equal(fit$lambda, 0.01)
  expect_true(fit$metrics$rel_residual < 0.1)
})

test_that("Numeric lambda still works for InducingKRR (no regression)", {
  set.seed(105)
  n <- 50
  d <- 3
  m <- 10
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01, optimizer = "pcg",
                            iters = 30)
  expect_s3_class(fit, "InducingKRR")
  expect_equal(fit$lambda, 0.01)
})

test_that("select_lambda_cv returns a scalar from the grid", {
  set.seed(106)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  kp <- list(type = "rbf", sigma = 1.0)
  grid <- c(1e-4, 1e-3, 1e-2, 1e-1)

  lam <- fasterKRR:::select_lambda_cv(x, y, kp, lambda_grid = grid,
                                     fast_cv = TRUE)
  expect_true(is.numeric(lam))
  expect_equal(length(lam), 1)
  expect_true(lam > 0)
  expect_true(lam %in% grid)
})

test_that("CreateFullKRR with lambda vector does CV over those values", {
  set.seed(108)
  n <- 60
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  candidates <- c(1e-4, 1e-3, 1e-2, 1e-1)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0,
                        lambda = candidates,
                        optimizer = "pcg", iters = 20)
  expect_s3_class(fit, "FullKRR")
  expect_true(fit$lambda %in% candidates)
  expect_true(is.finite(fit$metrics$rel_residual))

  x_new <- matrix(rnorm(5 * d), 5, d)
  preds <- predict(fit, x_new)
  expect_equal(length(preds), 5)
  expect_true(all(is.finite(preds)))
})

test_that("CreateInducingKRR with lambda vector does CV over those values", {
  set.seed(109)
  n <- 80
  d <- 3
  m <- 15
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  candidates <- c(1e-5, 1e-4, 1e-3, 1e-2)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = candidates,
                            optimizer = "pcg", iters = 20)
  expect_s3_class(fit, "InducingKRR")
  expect_true(fit$lambda %in% candidates)
  expect_true(is.finite(fit$metrics$rel_residual))

  x_new <- matrix(rnorm(5 * d), 5, d)
  preds <- predict(fit, x_new)
  expect_equal(length(preds), 5)
  expect_true(all(is.finite(preds)))
})

test_that("lambda vector with verbose reports auto-selected lambda", {
  set.seed(110)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  msgs <- capture.output(
    fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0,
                          lambda = c(0.001, 0.01, 0.1),
                          optimizer = "pcg", iters = 10, verbose = TRUE),
    type = "message"
  )
  expect_true(any(grepl("Auto-selected lambda", msgs)))
  expect_true(fit$lambda %in% c(0.001, 0.01, 0.1))
})

test_that("verbose mode reports auto-selected lambda", {
  set.seed(107)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  msgs <- capture.output(
    fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = NULL,
                          optimizer = "pcg", iters = 10, verbose = TRUE),
    type = "message"
  )
  expect_true(any(grepl("Auto-selected lambda", msgs)))
})
