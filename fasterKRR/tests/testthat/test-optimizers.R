# =============================================================================
# Tests for optimizers via public API integration tests
# =============================================================================

# --- PCG optimizer ------------------------------------------------------------

test_that("PCG converges on small RBF problem via CreateFullKRR", {
  set.seed(100)
  n <- 50
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "pcg", iters = 30)
  expect_true(fit$metrics$rel_residual < 0.01)
})

test_that("PCG with Newton preconditioner converges fast", {
  set.seed(101)
  n <- 30
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "pcg", iters = 5,
                        preconditioner = list(type = "newton",
                                              rho = "regularization"))
  expect_true(fit$metrics$rel_residual < 1e-6)
})

test_that("PCG with Nystrom preconditioner converges", {
  set.seed(102)
  n <- 50
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "pcg", iters = 15,
                        preconditioner = list(type = "nystrom", r = 10,
                                              rho = "regularization"))
  expect_true(fit$metrics$rel_residual < 0.5)
})

test_that("PCG with Partial Cholesky preconditioner converges", {
  set.seed(103)
  n <- 50
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "pcg", iters = 15,
                        preconditioner = list(type = "partial_cholesky",
                                              r = 10,
                                              rho = "regularization",
                                              mode = "greedy"))
  expect_true(fit$metrics$rel_residual < 0.5)
})

test_that("PCG with InducingKRR model converges", {
  set.seed(110)
  n <- 50
  m <- 10
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01, optimizer = "pcg",
                            iters = 30)
  expect_true(fit$metrics$rel_residual < 0.1)
})

# --- EigenPro2 optimizer ------------------------------------------------------

test_that("EigenPro2 runs without error via CreateFullKRR", {
  set.seed(104)
  n <- 60
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        optimizer = "eigenpro2", iters = 50,
                        block_sz = 50, r = 5)
  expect_true(is.finite(fit$metrics$rel_residual))
})

# --- ASkotch optimizer --------------------------------------------------------

test_that("ASkotch accelerated runs without error via CreateFullKRR", {
  set.seed(105)
  n <- 40
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "askotch", iters = 20,
                        B = 4, accelerated = TRUE)
  expect_true(is.finite(fit$metrics$rel_residual))
})

test_that("ASkotch non-accelerated runs without error", {
  set.seed(106)
  n <- 30
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "askotch", iters = 10,
                        B = 3, accelerated = FALSE)
  expect_true(is.finite(fit$metrics$rel_residual))
})

# --- ASkotchV2 optimizer -----------------------------------------------------

test_that("ASkotchV2 runs without error via CreateFullKRR", {
  set.seed(107)
  n <- 40
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "askotch_v2", iters = 10,
                        block_sz = 10, accelerated = TRUE)
  expect_true(is.finite(fit$metrics$rel_residual))
})

# --- EigenPro3 optimizer (InducingKRR) ----------------------------------------

test_that("EigenPro3 runs without error via CreateInducingKRR", {
  set.seed(120)
  n <- 500
  m <- 120
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01,
                            optimizer = "eigenpro3", iters = 10,
                            block_sz = 200, r = 5,
                            proj_inner_iters = 5)
  expect_true(is.finite(fit$metrics$rel_residual))
})

# --- EigenPro4 optimizer (InducingKRR) ----------------------------------------

test_that("EigenPro4 runs without error via CreateInducingKRR", {
  set.seed(121)
  n <- 500
  m <- 120
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01,
                            optimizer = "eigenpro4", iters = 10,
                            block_sz = 200, r = 5, proj_freq = 5,
                            proj_inner_iters = 5)
  expect_true(is.finite(fit$metrics$rel_residual))
})

# --- Mimosa optimizer (InducingKRR) -------------------------------------------

test_that("Mimosa runs without error via CreateInducingKRR", {
  set.seed(122)
  n <- 80
  m <- 15
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sin(x[, 1]) + 0.1 * rnorm(n)

  fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                            sigma = 1.0, lambda = 0.01,
                            optimizer = "mimosa", iters = 20,
                            bg = 20, bH = 20, bH2 = 20)
  expect_true(is.finite(fit$metrics$rel_residual))
})

# --- Invalid optimizer rejection ----------------------------------------------

test_that("CreateFullKRR rejects invalid optimizer", {
  set.seed(130)
  n <- 20
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  expect_error(
    CreateFullKRR(x, y, optimizer = "eigenpro3"),
    "optimizer must be one of"
  )
  expect_error(
    CreateFullKRR(x, y, optimizer = "mimosa"),
    "optimizer must be one of"
  )
  expect_error(
    CreateFullKRR(x, y, optimizer = "nonexistent"),
    "optimizer must be one of"
  )
})

test_that("CreateInducingKRR rejects invalid optimizer", {
  set.seed(131)
  n <- 20
  m <- 5
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  expect_error(
    CreateInducingKRR(x, y, inducing_pts = 1:m, optimizer = "eigenpro2"),
    "optimizer must be one of"
  )
  expect_error(
    CreateInducingKRR(x, y, inducing_pts = 1:m, optimizer = "askotch"),
    "optimizer must be one of"
  )
  expect_error(
    CreateInducingKRR(x, y, inducing_pts = 1:m, optimizer = "nonexistent"),
    "optimizer must be one of"
  )
})

# --- Classification with PCG -------------------------------------------------

test_that("Classification with PCG works via CreateFullKRR", {
  set.seed(108)
  n <- 100
  d <- 3
  x <- matrix(rnorm(n * d), n, d)
  y <- sign(x[, 1])

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
                        task = "classification", optimizer = "pcg",
                        iters = 30)
  expect_s3_class(fit, "FullKRR")
  expect_true(is.finite(fit$metrics$rel_residual))

  # Verify training accuracy is reasonable
  train_scores <- predict(fit, x)
  train_acc <- mean(sign(train_scores) == y)
  expect_true(train_acc > 0.8)
})

# --- Verbose mode checks convergence history ----------------------------------

test_that("Verbose mode records iteration history", {
  set.seed(109)
  n <- 30
  x <- matrix(rnorm(n * 3), n, 3)
  y <- rnorm(n)

  fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.1,
                        optimizer = "pcg", iters = 10, verbose = TRUE)
  expect_true(!is.null(fit$history))
  expect_true(length(fit$history) > 0)
  # Each entry should have an iter field
  expect_true("iter" %in% names(fit$history[[1]]))
})

# --- Minibatch generator internal test ----------------------------------------

test_that("Minibatch generator closure works", {
  gen <- fasterKRR:::CreateMinibatchGenerator(100, 10)
  expect_true(is.function(gen))
  batch <- gen()
  expect_equal(length(batch), 10)
  expect_true(all(batch >= 1 & batch <= 100))

  # Exhaust one epoch
  for (i in 2:10) gen()
  # Should reshuffle
  batch2 <- gen()
  expect_equal(length(batch2), 10)
})
