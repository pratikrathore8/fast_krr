# =============================================================================
# Tests for preconditioners (all internal, accessed via fasterKRR:::)
# =============================================================================

test_that("NewtonPrecond inverts correctly", {
  set.seed(20)
  n <- 10
  A <- crossprod(matrix(rnorm(n * n), n, n)) + 0.5 * diag(n)
  K_lin_op <- function(v) A %*% v
  precond <- fasterKRR:::CreateNewtonPrecond(rho = 0.5, lambd = 0.5,
                                            K_lin_op = K_lin_op, n = n)
  expect_s3_class(precond, "NewtonPrecond")

  v <- rnorm(n)
  result <- fasterKRR:::InvLinOp(precond, v)
  expected <- solve(A + 0.5 * diag(n), v)
  expect_equal(result, as.vector(expected), tolerance = 1e-8)
})

test_that("NystromPrecond creates and applies", {
  set.seed(21)
  n <- 20
  x <- matrix(rnorm(n * 3), n, 3)
  kp <- list(type = "rbf", sigma = 1.0)
  K <- fasterKRR:::ComputeKernel(x, x, kp)
  K_lin_op <- function(v) K %*% v

  precond <- fasterKRR:::CreateNystromPrecond(r = 5, rho = "regularization",
                                             lambd = 0.1,
                                             K_lin_op = K_lin_op,
                                             K_trace = n, n = n)
  expect_s3_class(precond, "NystromPrecond")

  v <- rnorm(n)
  result <- fasterKRR:::InvLinOp(precond, v)
  expect_equal(length(result), n)
  expect_true(all(is.finite(result)))
})

test_that("FalkonPrecond creates and applies", {
  set.seed(22)
  m <- 8
  x_ind <- matrix(rnorm(m * 3), m, 3)
  kp <- list(type = "rbf", sigma = 1.0)
  K_mm <- fasterKRR:::ComputeKernel(x_ind, x_ind, kp)
  K_mm_lin_op <- function(v) K_mm %*% v

  precond <- fasterKRR:::CreateFalkonPrecond(lambd = 0.1,
                                            K_mm_lin_op = K_mm_lin_op,
                                            K_mm_trace = m, n = 50, m = m)
  expect_s3_class(precond, "FalkonPrecond")

  v <- rnorm(m)
  result <- fasterKRR:::InvLinOp(precond, v)
  expect_equal(length(result), m)
  expect_true(all(is.finite(result)))
})

test_that("PartialCholeskyPrecond greedy mode works", {
  set.seed(23)
  n <- 20
  x <- matrix(rnorm(n * 3), n, 3)
  kp <- list(type = "rbf", sigma = 1.0)
  K_fn <- function(x_i, x_j, get_row = FALSE) {
    if (get_row) {
      fasterKRR:::ComputeKernelRow(x_i, x_j, kp)
    } else {
      fasterKRR:::ComputeKernel(x_i, x_j, kp)
    }
  }
  K_diag <- rep(1, n)

  precond <- fasterKRR:::CreatePartialCholeskyPrecond(
    r = 5, rho = "regularization", lambd = 0.1,
    K_fn = K_fn, K_diag = K_diag, x = x, mode = "greedy"
  )
  expect_s3_class(precond, "PartialCholeskyPrecond")

  v <- rnorm(n)
  result <- fasterKRR:::InvLinOp(precond, v)
  expect_equal(length(result), n)
  expect_true(all(is.finite(result)))
})

test_that("PartialCholeskyPrecond rpc mode works", {
  set.seed(24)
  n <- 30
  x <- matrix(rnorm(n * 3), n, 3)
  kp <- list(type = "rbf", sigma = 1.0)
  K_fn <- function(x_i, x_j, get_row = FALSE) {
    if (get_row) {
      fasterKRR:::ComputeKernelRow(x_i, x_j, kp)
    } else {
      fasterKRR:::ComputeKernel(x_i, x_j, kp)
    }
  }
  K_diag <- rep(1, n)

  precond <- fasterKRR:::CreatePartialCholeskyPrecond(
    r = 5, rho = 0.1, lambd = 0.1,
    K_fn = K_fn, K_diag = K_diag, x = x, mode = "rpc"
  )
  expect_s3_class(precond, "PartialCholeskyPrecond")

  v <- rnorm(n)
  result <- fasterKRR:::InvLinOp(precond, v)
  expect_equal(length(result), n)
})

test_that("InitPrecond returns NULL for NULL params", {
  result <- fasterKRR:::InitPrecond(NULL, NULL, 0.1)
  expect_null(result)
})

test_that("GetDamping resolves regularization", {
  expect_equal(fasterKRR:::GetDamping(0.5, 0.1), 0.5)
  expect_equal(fasterKRR:::GetDamping("regularization", 0.1), 0.1)
  expect_error(fasterKRR:::GetDamping("unknown", 0.1))
})
