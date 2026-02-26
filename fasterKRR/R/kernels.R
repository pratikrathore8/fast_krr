# Kernel functions for fast kernel ridge regression.
#
# Supports RBF, Matern (nu = 0.5, 1.5, 2.5), and L1 Laplace kernels.
# All kernel functions operate on matrices where rows are observations.

# --- Distance helpers --------------------------------------------------------

ComputeSqDistMatrix <- function(x1, x2) {
  sq1 <- rowSums(x1^2)
  sq2 <- rowSums(x2^2)
  D <- outer(sq1, sq2, "+") - 2 * tcrossprod(x1, x2)
  D[D < 0] <- 0
  return(D)
}

ComputeL1DistMatrix <- function(x1, x2) {
  d <- ncol(x1)
  D <- matrix(0, nrow(x1), nrow(x2))
  for (k in seq_len(d)) {
    D <- D + abs(outer(x1[, k], x2[, k], "-"))
  }
  return(D)
}

# --- Individual kernel computations -----------------------------------------

CheckRbfParams <- function(kernel_params) {
  if (is.null(kernel_params$sigma)) {
    stop("Missing sigma for RBF kernel")
  }
}

ComputeRbfKernel <- function(x1, x2, kernel_params) {
  CheckRbfParams(kernel_params)
  D <- ComputeSqDistMatrix(x1, x2)
  return(exp(-D / (2 * kernel_params$sigma^2)))
}

ComputeRbfRow <- function(x_i, x, kernel_params) {
  D <- colSums((as.vector(x_i) - t(x))^2)
  return(exp(-D / (2 * kernel_params$sigma^2)))
}

CheckMaternParams <- function(kernel_params) {
  if (is.null(kernel_params$sigma)) {
    stop("Missing sigma for Matern kernel")
  }
  if (is.null(kernel_params$nu)) {
    stop("Missing nu for Matern kernel")
  }
  if (!(kernel_params$nu %in% c(0.5, 1.5, 2.5))) {
    stop("nu must be 0.5, 1.5, or 2.5")
  }
}

ComputeMaternKernel <- function(x1, x2, kernel_params) {
  CheckMaternParams(kernel_params)
  nu <- kernel_params$nu
  sigma <- kernel_params$sigma
  D <- sqrt(ComputeSqDistMatrix(x1, x2))

  if (nu == 0.5) {
    K <- exp(-D / sigma)
  } else if (nu == 1.5) {
    D_adj <- sqrt(3) * D / sigma
    K <- (1 + D_adj) * exp(-D_adj)
  } else {
    D_adj <- sqrt(5) * D / sigma
    K <- (1 + D_adj + 5 * D^2 / (3 * sigma^2)) * exp(-D_adj)
  }
  return(K)
}

ComputeMaternRow <- function(x_i, x, kernel_params) {
  nu <- kernel_params$nu
  sigma <- kernel_params$sigma
  D <- sqrt(colSums((as.vector(x_i) - t(x))^2))

  if (nu == 0.5) {
    k_i <- exp(-D / sigma)
  } else if (nu == 1.5) {
    D_adj <- sqrt(3) * D / sigma
    k_i <- (1 + D_adj) * exp(-D_adj)
  } else {
    D_adj <- sqrt(5) * D / sigma
    k_i <- (1 + D_adj + 5 * D^2 / (3 * sigma^2)) * exp(-D_adj)
  }
  return(k_i)
}

CheckL1LaplaceParams <- function(kernel_params) {
  if (is.null(kernel_params$sigma)) {
    stop("Missing sigma for L1 Laplace kernel")
  }
}

ComputeL1LaplaceKernel <- function(x1, x2, kernel_params) {
  CheckL1LaplaceParams(kernel_params)
  D <- ComputeL1DistMatrix(x1, x2)
  return(exp(-D / kernel_params$sigma))
}

ComputeL1LaplaceRow <- function(x_i, x, kernel_params) {
  D <- colSums(abs(as.vector(x_i) - t(x)))
  return(exp(-D / kernel_params$sigma))
}

# --- Kernel dispatch ---------------------------------------------------------

KERNEL_CLASSES <- list(
  rbf = list(
    kernel = ComputeRbfKernel,
    row = ComputeRbfRow
  ),
  matern = list(
    kernel = ComputeMaternKernel,
    row = ComputeMaternRow
  ),
  l1_laplace = list(
    kernel = ComputeL1LaplaceKernel,
    row = ComputeL1LaplaceRow
  )
)

#' Compute a kernel matrix.
#'
#' @param x1 n1 x d matrix.
#' @param x2 n2 x d matrix.
#' @param kernel_params List with "type" (one of "rbf", "matern", "l1_laplace")
#'   and kernel-specific parameters (e.g., "sigma", "nu").
#' @return n1 x n2 kernel matrix.
ComputeKernel <- function(x1, x2, kernel_params) {
  ktype <- kernel_params$type
  params <- kernel_params[names(kernel_params) != "type"]
  return(KERNEL_CLASSES[[ktype]]$kernel(x1, x2, params))
}

#' Compute a single row of the kernel matrix.
#'
#' @param x_i 1 x d matrix or length-d vector (single data point).
#' @param x n x d matrix.
#' @param kernel_params Kernel parameter list.
#' @return Length-n vector.
ComputeKernelRow <- function(x_i, x, kernel_params) {
  ktype <- kernel_params$type
  return(KERNEL_CLASSES[[ktype]]$row(x_i, x, kernel_params))
}

#' Compute the diagonal of the kernel matrix.
#'
#' For all supported kernels, the diagonal is all ones.
#'
#' @param n Number of data points.
#' @param kernel_params Kernel parameter list (used for dispatch only).
#' @return Length-n vector of ones.
ComputeKernelDiag <- function(n, kernel_params) {
  return(rep(1, n))
}

#' Compute the trace of the kernel matrix.
#'
#' @param n Number of data points.
#' @param kernel_params Kernel parameter list.
#' @return Scalar trace value (equals n for all supported kernels).
ComputeKernelTrace <- function(n, kernel_params) {
  return(n)
}

#' Set up kernel matrices for training and testing.
#'
#' @param x n x d training matrix.
#' @param x_tst n_tst x d test matrix (or NULL).
#' @param kernel_params Kernel parameter list.
#' @param k_tr_needed Logical; compute training kernel matrix?
#' @return List with K (training kernel or NULL) and K_tst (test kernel or NULL).
InitKernels <- function(x, x_tst, kernel_params, k_tr_needed = TRUE) {
  K <- NULL
  if (k_tr_needed) {
    K <- ComputeKernel(x, x, kernel_params)
  }
  K_tst <- NULL
  if (!is.null(x_tst)) {
    K_tst <- ComputeKernel(x_tst, x, kernel_params)
  }
  return(list(K = K, K_tst = K_tst))
}
