# Cross-validation for automatic lambda selection via the CVST package.
#
# Internal functions — not exported. Called by CreateFullKRR() and
# CreateInducingKRR() when lambda = NULL.

# --- Main entry point --------------------------------------------------------

#' Select lambda by cross-validation using CVST
#'
#' @param x Training features matrix (n x d).
#' @param y Training target vector (length n).
#' @param kernel_params Named list with kernel type and parameters.
#' @param lambda_grid Numeric vector of candidate lambda values.
#' @param fast_cv Logical; if TRUE use CVST::fastCV, else CVST::CV.
#' @param cv_folds Integer number of folds for standard CV.
#' @param inducing Logical; if TRUE use the inducing-point learner.
#' @param m Integer number of inducing points (used when inducing = TRUE).
#' @return A single positive numeric lambda value.
#' @keywords internal
select_lambda_cv <- function(x, y, kernel_params, lambda_grid = NULL,
                             fast_cv = TRUE, cv_folds = 5L,
                             inducing = FALSE, m = NULL) {
  if (is.null(lambda_grid)) {
    lambda_grid <- seq(1e-10, 1e-3, length.out = 100)
  }

  # Build CVST data object
  cvst_data <- CVST::constructData(x = x, y = y)

  # Build CVST.params — only lambda varies; kernel_params and m are
  # captured by the learner closures.
  params <- lapply(lambda_grid, function(lam) list(lambda = lam))
  names(params) <- paste0("lambda=", lambda_grid)
  class(params) <- "CVST.params"

  # Build CVST learner (closures capture kernel_params and m)
  learner <- if (inducing) {
    make_inducing_learner(kernel_params, m)
  } else {
    make_full_learner(kernel_params)
  }

  # Run CV
  if (fast_cv) {
    setup <- CVST::constructCVSTModel()
    result <- CVST::fastCV(cvst_data, learner, params, setup, verbose = FALSE)
  } else {
    result <- CVST::CV(cvst_data, learner, params, fold = cv_folds,
                       verbose = FALSE)
  }

  best_lambda <- result[[1]]$lambda
  best_lambda
}

# --- CVST learner constructors -----------------------------------------------

make_full_learner <- function(kernel_params) {
  learn_fn <- function(data, param) {
    x <- data$x
    y <- data$y
    n <- nrow(x)
    lam <- param$lambda

    K <- ComputeKernel(x, x, kernel_params)
    coefficients <- solve(K + lam * diag(n), y)

    list(x_train = x, coefficients = coefficients,
         kernel_params = kernel_params)
  }

  predict_fn <- function(model, newdata) {
    K_new <- ComputeKernel(newdata$x, model$x_train, model$kernel_params)
    as.vector(K_new %*% model$coefficients)
  }

  CVST::constructLearner(learn_fn, predict_fn)
}

make_inducing_learner <- function(kernel_params, m) {
  learn_fn <- function(data, param) {
    x <- data$x
    y <- data$y
    n <- nrow(x)
    lam <- param$lambda
    m_fold <- min(m, n)

    # Select m random inducing points from the fold's data
    ind <- sample.int(n, m_fold)
    x_ind <- x[ind, , drop = FALSE]

    K_nm <- ComputeKernel(x, x_ind, kernel_params)
    K_mm <- ComputeKernel(x_ind, x_ind, kernel_params)

    # Solve the inducing-point system:
    #   (K_nm^T K_nm + lambda * K_mm) alpha = K_nm^T y
    A <- crossprod(K_nm) + lam * K_mm
    b <- as.vector(crossprod(K_nm, y))
    coefficients <- solve(A, b)

    list(x_ind = x_ind, coefficients = coefficients,
         kernel_params = kernel_params)
  }

  predict_fn <- function(model, newdata) {
    K_new <- ComputeKernel(newdata$x, model$x_ind, model$kernel_params)
    as.vector(K_new %*% model$coefficients)
  }

  CVST::constructLearner(learn_fn, predict_fn)
}
