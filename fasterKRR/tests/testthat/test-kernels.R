# =============================================================================
# Tests for kernel functions (all internal, accessed via fasterKRR:::)
# =============================================================================

test_that("RBF kernel is symmetric PSD with unit diagonal", {
  set.seed(1)
  x <- matrix(rnorm(30), 6, 5)
  kp <- list(type = "rbf", sigma = 1.0)
  K <- fasterKRR:::ComputeKernel(x, x, kp)
  expect_equal(K, t(K))
  expect_equal(diag(K), rep(1, 6))
  evals <- eigen(K, symmetric = TRUE, only.values = TRUE)$values
  expect_true(all(evals >= -1e-10))
})

test_that("Matern kernels are symmetric with unit diagonal", {
  set.seed(2)
  x <- matrix(rnorm(20), 4, 5)
  for (nu in c(0.5, 1.5, 2.5)) {
    kp <- list(type = "matern", sigma = 1.0, nu = nu)
    K <- fasterKRR:::ComputeKernel(x, x, kp)
    expect_equal(K, t(K), tolerance = 1e-12)
    expect_equal(diag(K), rep(1, 4), tolerance = 1e-6)
  }
})

test_that("L1 Laplace kernel is symmetric with unit diagonal", {
  set.seed(3)
  x <- matrix(rnorm(20), 4, 5)
  kp <- list(type = "l1_laplace", sigma = 1.0)
  K <- fasterKRR:::ComputeKernel(x, x, kp)
  expect_equal(K, t(K))
  expect_equal(diag(K), rep(1, 4))
})

test_that("ComputeKernelRow matches ComputeKernel row", {
  set.seed(4)
  x <- matrix(rnorm(30), 6, 5)
  kp <- list(type = "rbf", sigma = 1.0)
  K <- fasterKRR:::ComputeKernel(x, x, kp)
  for (i in 1:3) {
    row_i <- fasterKRR:::ComputeKernelRow(x[i, , drop = FALSE], x, kp)
    expect_equal(row_i, K[i, ], tolerance = 1e-12)
  }
})

test_that("ComputeKernelDiag returns ones", {
  kp <- list(type = "rbf", sigma = 1.0)
  expect_equal(fasterKRR:::ComputeKernelDiag(5, kp), rep(1, 5))
})

test_that("ComputeKernelTrace returns n", {
  kp <- list(type = "rbf", sigma = 1.0)
  expect_equal(fasterKRR:::ComputeKernelTrace(10, kp), 10)
})

test_that("InitKernels computes training and test kernels", {
  set.seed(5)
  x <- matrix(rnorm(20), 4, 5)
  x_tst <- matrix(rnorm(10), 2, 5)
  kp <- list(type = "rbf", sigma = 1.0)
  result <- fasterKRR:::InitKernels(x, x_tst, kp)
  expect_equal(dim(result$K), c(4, 4))
  expect_equal(dim(result$K_tst), c(2, 4))
})

test_that("InitKernels skips training kernel when not needed", {
  set.seed(6)
  x <- matrix(rnorm(20), 4, 5)
  kp <- list(type = "rbf", sigma = 1.0)
  result <- fasterKRR:::InitKernels(x, NULL, kp, k_tr_needed = FALSE)
  expect_null(result$K)
  expect_null(result$K_tst)
})

test_that("Matern with invalid nu stops", {
  x <- matrix(rnorm(10), 2, 5)
  kp <- list(type = "matern", sigma = 1.0, nu = 3.5)
  expect_error(fasterKRR:::ComputeKernel(x, x, kp), "nu must be")
})

test_that("RBF with missing sigma stops", {
  x <- matrix(rnorm(10), 2, 5)
  kp <- list(type = "rbf")
  expect_error(fasterKRR:::ComputeKernel(x, x, kp), "Missing sigma")
})

test_that("Rectangular kernel matrix has correct dimensions", {
  set.seed(7)
  x1 <- matrix(rnorm(15), 3, 5)
  x2 <- matrix(rnorm(35), 7, 5)
  kp <- list(type = "rbf", sigma = 2.0)
  K <- fasterKRR:::ComputeKernel(x1, x2, kp)
  expect_equal(dim(K), c(3, 7))
})
