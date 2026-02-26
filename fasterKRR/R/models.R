# Model constructors, fitting, and S3 methods for kernel ridge regression.
#
# Public API: CreateFullKRR() and CreateInducingKRR() handle the full
# workflow (model setup + optimization + return fitted object).
# predict.FullKRR() and predict.InducingKRR() are the S3 predict methods.
# All other functions are internal.

# --- Helpers ----------------------------------------------------------------

Norm2 <- function(v) {
  return(sqrt(sum(v^2)))
}

build_kernel_params <- function(kernel, sigma, nu = NULL) {
  kp <- list(type = kernel, sigma = sigma)
  if (!is.null(nu)) kp$nu <- nu
  kp
}

# --- S3 Generics -----------------------------------------------------------

#' @keywords internal
LinOp <- function(model, v) {
  UseMethod("LinOp")
}

#' @keywords internal
ComputeMetrics <- function(model, v) {
  UseMethod("ComputeMetrics")
}

#' Update a KRR Model with a New Observation (Online Learning)
#'
#' Returns a \emph{new} model with the observation incorporated.
#' The original model is never modified.
#'
#' @param model A fitted KRR model.
#' @param x_new Numeric vector of length d (or 1 x d matrix) for the new point.
#' @param y_new Single numeric value for the new observation.
#' @param ... Additional arguments passed to methods.
#' @return A new model object with the observation incorporated.
#' @export
UpdateModel <- function(model, x_new, y_new, ...) {
  UseMethod("UpdateModel")
}

# --- Internal model builders ------------------------------------------------

build_full_model <- function(x, y, kernel_params, k_tr_needed = TRUE,
                              lambda, task, w0) {
  x <- as.matrix(x)
  y <- as.vector(y)
  w <- as.vector(w0)
  n <- nrow(x)

  K <- NULL
  if (k_tr_needed) {
    K <- ComputeKernel(x, x, kernel_params)
  }

  model <- list(
    x = x,
    y = y,
    kernel_params = kernel_params,
    lambda = lambda,
    task = task,
    w = w,
    n = n,
    y_norm = Norm2(y),
    K = K,
    inducing = FALSE
  )
  class(model) <- "FullKRR"
  return(model)
}

build_inducing_model <- function(x, y, kernel_params, k_nm_needed,
                                  inducing_pts, lambda, task, w0) {
  x <- as.matrix(x)
  y <- as.vector(y)
  w <- as.vector(w0)
  n <- nrow(x)
  m <- length(inducing_pts)
  ym_scaled <- (lambda / 2) * y[inducing_pts]

  x_ind <- x[inducing_pts, , drop = FALSE]
  K_mm <- ComputeKernel(x_ind, x_ind, kernel_params)

  K_nm <- NULL
  K_nmTy <- NULL
  K_nmTy_norm <- NULL
  if (k_nm_needed) {
    K_nm <- ComputeKernel(x, x_ind, kernel_params)
    K_nmTy <- as.vector(crossprod(K_nm, y))
    K_nmTy_norm <- Norm2(K_nmTy)
  }

  model <- list(
    x = x,
    y = y,
    kernel_params = kernel_params,
    lambda = lambda,
    task = task,
    w = w,
    n = n,
    y_norm = Norm2(y),
    inducing_pts = inducing_pts,
    m = m,
    ym_scaled = ym_scaled,
    inducing = TRUE,
    K_mm = K_mm,
    K_nm = K_nm,
    K_nmTy = K_nmTy,
    K_nmTy_norm = K_nmTy_norm
  )
  class(model) <- "InducingKRR"
  return(model)
}

# --- Optimizer runner -------------------------------------------------------

run_optimizer <- function(model, optimizer, preconditioner, iters, verbose,
                           dots) {
  opt_state <- switch(optimizer,
    "pcg" = CreatePCG(model, preconditioner),
    "eigenpro2" = CreateEigenPro2(
      model,
      block_sz = dots$block_sz %||% min(model$n, 12000L),
      r = dots$r %||% 10L,
      bg = dots$bg,
      gamma = dots$gamma %||% 0.95
    ),
    "askotch" = CreateASkotch(
      model,
      B = dots$B %||% 5L,
      no_store_precond = dots$no_store_precond %||% TRUE,
      precond_params = preconditioner,
      beta = dots$beta %||% 0,
      accelerated = dots$accelerated %||% TRUE
    ),
    "askotch_v2" = CreateASkotchV2(
      model,
      block_sz = dots$block_sz %||% max(1L, model$n %/% 5L),
      sampling_method = dots$sampling_method %||% "uniform",
      precond_params = preconditioner,
      mu = dots$mu,
      nu = dots$nu_param,
      accelerated = dots$accelerated %||% TRUE
    ),
    "eigenpro3" = CreateEigenPro3(
      model,
      block_sz = dots$block_sz %||% min(model$n, 12000L),
      r = dots$r %||% 10L,
      bg = dots$bg,
      proj_inner_iters = dots$proj_inner_iters %||% 10L
    ),
    "eigenpro4" = CreateEigenPro4(
      model,
      block_sz = dots$block_sz %||% min(model$n, 12000L),
      r = dots$r %||% 10L,
      proj_freq = dots$proj_freq %||% 10L,
      bg = dots$bg,
      proj_inner_iters = dots$proj_inner_iters %||% 10L
    ),
    "mimosa" = CreateMimosa(
      model,
      bg = dots$bg %||% max(1L, model$n %/% 10L),
      bH = dots$bH %||% max(1L, model$n %/% 10L),
      bH2 = dots$bH2 %||% max(1L, model$n %/% 10L),
      precond_params = preconditioner
    ),
    stop("Unknown optimizer: ", optimizer)
  )

  log_interval <- max(1L, iters %/% 10L)
  history <- if (verbose) list() else NULL

  for (i in seq_len(iters)) {
    opt_state <- Step(opt_state)
    if (verbose && (i %% log_interval == 0 || i == iters)) {
      metrics_i <- ComputeMetrics(opt_state$model, opt_state$model$w)
      rr <- if (!is.null(metrics_i$rel_residual)) {
        sprintf("rel_residual=%.6f", metrics_i$rel_residual)
      } else {
        ""
      }
      message(sprintf("[fasterKRR] Iter %*d/%d  %s", nchar(iters), i, iters, rr))
      history[[length(history) + 1]] <- c(list(iter = i), metrics_i)
    }
  }

  final_metrics <- ComputeMetrics(opt_state$model, opt_state$model$w)
  opt_state$model$metrics <- final_metrics
  if (verbose) opt_state$model$history <- history

  return(opt_state$model)
}

# --- Public API: CreateFullKRR ----------------------------------------------

#' Fit a Full Kernel Ridge Regression Model
#'
#' Creates and fits a full KRR model in one call. The model setup, optimization,
#' and convergence happen internally. Use \code{predict()} on the returned
#' object to make predictions on new data.
#'
#' @param x Training features matrix (n x d).
#' @param y Training target vector (length n).
#' @param kernel Kernel type: \code{"rbf"} (default), \code{"l1_laplace"}, or
#'   \code{"matern"}.
#' @param sigma Kernel bandwidth parameter.
#' @param nu Matern smoothness parameter (0.5, 1.5, or 2.5). Only used when
#'   \code{kernel = "matern"}.
#' @param lambda Regularization parameter. A single positive numeric value
#'   uses that value directly (default \code{1e-3}). If \code{NULL}, the best
#'   lambda is selected automatically via cross-validation using the
#'   \pkg{CVST} package over a default grid. If a numeric vector of length
#'   greater than one, cross-validation is performed over those candidate
#'   values.
#' @param task \code{"regression"} or \code{"classification"}.
#' @param optimizer Optimizer to use: \code{"pcg"} (default),
#'   \code{"eigenpro2"}, \code{"askotch"}, or \code{"askotch_v2"}. Note that
#'   \code{"eigenpro3"}, \code{"eigenpro4"}, and \code{"mimosa"} are only
#'   available for \code{\link{CreateInducingKRR}}.
#' @param preconditioner Either \code{NULL} (no preconditioner) or a named list
#'   specifying the preconditioner, e.g.
#'   \code{list(type = "newton", rho = "regularization")},
#'   \code{list(type = "nystrom", r = 20, rho = "regularization")},
#'   \code{list(type = "partial_cholesky", r = 20, rho = "regularization",
#'   mode = "greedy")}.
#' @param iters Number of optimization iterations.
#' @param verbose If \code{TRUE}, print progress every ~10\% of iterations.
#' @param w0 Initial weight vector (default: zeros).
#' @param fast_cv Logical; if \code{TRUE} (default), use
#'   \code{CVST::fastCV} (sequential testing) for lambda selection; if
#'   \code{FALSE}, use standard k-fold CV via \code{CVST::CV}. Only
#'   used when \code{lambda = NULL}.
#' @param cv_folds Integer number of folds for standard CV (default
#'   \code{5L}). Ignored when \code{fast_cv = TRUE}. Only used when
#'   \code{lambda = NULL}.
#' @param lambda_grid Numeric vector of candidate lambda values for
#'   cross-validation. If \code{NULL} (default), uses
#'   \code{seq(1e-10, 1e-3, length.out = 100)}. Only used when
#'   \code{lambda = NULL}.
#' @param ... Additional optimizer-specific parameters. For \code{"eigenpro2"}:
#'   \code{block_sz}, \code{r}, \code{bg}, \code{gamma}. For
#'   \code{"askotch"}: \code{B}, \code{no_store_precond}, \code{beta},
#'   \code{accelerated}. For \code{"askotch_v2"}: \code{block_sz},
#'   \code{sampling_method}, \code{mu}, \code{nu_param}, \code{accelerated}.
#' @return A fitted object of class \code{"FullKRR"} with a \code{$metrics}
#'   element containing final training metrics, and a \code{$history}
#'   element if \code{verbose = TRUE}. Use \code{predict()} for new
#'   predictions.
#' @examples
#' set.seed(42)
#' n <- 100; d <- 5
#' x <- matrix(rnorm(n * d), n, d)
#' y <- sin(x[, 1]) + 0.1 * rnorm(n)
#'
#' fit <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 0.01,
#'                       optimizer = "pcg", iters = 30)
#' predict(fit, matrix(rnorm(5 * d), 5, d))
#' @export
CreateFullKRR <- function(x, y, kernel = "rbf", sigma = 1.0, nu = NULL,
                           lambda = 1e-3, task = "regression",
                           optimizer = "pcg", preconditioner = NULL,
                           iters = 30L, verbose = FALSE, w0 = NULL,
                           fast_cv = TRUE, cv_folds = 5L,
                           lambda_grid = NULL, ...) {
  x <- as.matrix(x)
  y <- as.vector(y)
  n <- nrow(x)
  if (is.null(w0)) w0 <- rep(0, n)
  kernel_params <- build_kernel_params(kernel, sigma, nu)

  if (is.null(lambda) || (is.numeric(lambda) && length(lambda) > 1)) {
    lambda <- select_lambda_cv(
      x, y, kernel_params,
      lambda_grid = if (is.numeric(lambda) && length(lambda) > 1) lambda else lambda_grid,
      fast_cv = fast_cv,
      cv_folds = cv_folds,
      inducing = FALSE
    )
    if (verbose) message("[fasterKRR] Auto-selected lambda = ", signif(lambda, 4))
  }

  valid_opts <- c("pcg", "eigenpro2", "askotch", "askotch_v2")
  if (!(optimizer %in% valid_opts)) {
    stop("optimizer must be one of: ", paste(valid_opts, collapse = ", "))
  }

  k_tr_needed <- TRUE
  model <- build_full_model(x, y, kernel_params, k_tr_needed, lambda,
                             task, w0)

  dots <- list(...)
  fitted_model <- run_optimizer(model, optimizer, preconditioner, iters,
                                 verbose, dots)
  return(fitted_model)
}

# --- Public API: CreateInducingKRR ------------------------------------------

#' Fit an Inducing Point Kernel Ridge Regression Model
#'
#' Creates and fits an inducing-point KRR model in one call. This uses a
#' subset of training points (inducing points) to approximate the full kernel,
#' enabling scalability to larger datasets.
#'
#' @param x Training features matrix (n x d).
#' @param y Training target vector (length n).
#' @param inducing_pts Integer vector of inducing point indices into \code{x}.
#' @param kernel Kernel type: \code{"rbf"} (default), \code{"l1_laplace"}, or
#'   \code{"matern"}.
#' @param sigma Kernel bandwidth parameter.
#' @param nu Matern smoothness parameter (0.5, 1.5, or 2.5). Only used when
#'   \code{kernel = "matern"}.
#' @param lambda Regularization parameter. A single positive numeric value
#'   uses that value directly (default \code{1e-3}). If \code{NULL}, the best
#'   lambda is selected automatically via cross-validation using the
#'   \pkg{CVST} package over a default grid. If a numeric vector of length
#'   greater than one, cross-validation is performed over those candidate
#'   values.
#' @param task \code{"regression"} or \code{"classification"}.
#' @param optimizer Optimizer to use: \code{"pcg"}, \code{"eigenpro3"},
#'   \code{"eigenpro4"}, or \code{"mimosa"}.
#' @param preconditioner Either \code{NULL} or a named list specifying the
#'   preconditioner. See \code{\link{CreateFullKRR}} for format. Additionally
#'   supports \code{list(type = "falkon")} for PCG.
#' @param iters Number of optimization iterations.
#' @param verbose If \code{TRUE}, print progress every ~10\% of iterations.
#' @param w0 Initial weight vector (default: zeros of length
#'   \code{length(inducing_pts)}).
#' @param fast_cv Logical; if \code{TRUE} (default), use
#'   \code{CVST::fastCV} (sequential testing) for lambda selection; if
#'   \code{FALSE}, use standard k-fold CV via \code{CVST::CV}. Only
#'   used when \code{lambda = NULL}.
#' @param cv_folds Integer number of folds for standard CV (default
#'   \code{5L}). Ignored when \code{fast_cv = TRUE}. Only used when
#'   \code{lambda = NULL}.
#' @param lambda_grid Numeric vector of candidate lambda values for
#'   cross-validation. If \code{NULL} (default), uses
#'   \code{seq(1e-10, 1e-3, length.out = 100)}. Only used when
#'   \code{lambda = NULL}.
#' @param ... Additional optimizer-specific parameters. For
#'   \code{"eigenpro3"}: \code{block_sz}, \code{r}, \code{bg},
#'   \code{proj_inner_iters}. For \code{"eigenpro4"}: \code{block_sz},
#'   \code{r}, \code{proj_freq}, \code{bg}, \code{proj_inner_iters}. For
#'   \code{"mimosa"}: \code{bg}, \code{bH}, \code{bH2}.
#' @return A fitted object of class \code{"InducingKRR"} with a
#'   \code{$metrics} element and optionally \code{$history}. Use
#'   \code{predict()} for new predictions.
#' @examples
#' set.seed(42)
#' n <- 100; d <- 5; m <- 15
#' x <- matrix(rnorm(n * d), n, d)
#' y <- sin(x[, 1]) + 0.1 * rnorm(n)
#'
#' fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
#'                           sigma = 1.0, lambda = 0.01, optimizer = "pcg",
#'                           iters = 30)
#' predict(fit, matrix(rnorm(5 * d), 5, d))
#' @export
CreateInducingKRR <- function(x, y, inducing_pts, kernel = "rbf",
                               sigma = 1.0, nu = NULL, lambda = 1e-3,
                               task = "regression", optimizer = "pcg",
                               preconditioner = NULL, iters = 30L,
                               verbose = FALSE, w0 = NULL,
                               fast_cv = TRUE, cv_folds = 5L,
                               lambda_grid = NULL, ...) {
  x <- as.matrix(x)
  y <- as.vector(y)
  m <- length(inducing_pts)
  if (is.null(w0)) w0 <- rep(0, m)
  kernel_params <- build_kernel_params(kernel, sigma, nu)

  if (is.null(lambda) || (is.numeric(lambda) && length(lambda) > 1)) {
    lambda <- select_lambda_cv(
      x, y, kernel_params,
      lambda_grid = if (is.numeric(lambda) && length(lambda) > 1) lambda else lambda_grid,
      fast_cv = fast_cv,
      cv_folds = cv_folds,
      inducing = TRUE,
      m = m
    )
    if (verbose) message("[fasterKRR] Auto-selected lambda = ", signif(lambda, 4))
  }

  valid_opts <- c("pcg", "eigenpro3", "eigenpro4", "mimosa")
  if (!(optimizer %in% valid_opts)) {
    stop("optimizer must be one of: ", paste(valid_opts, collapse = ", "))
  }

  k_nm_needed <- TRUE
  model <- build_inducing_model(x, y, kernel_params, k_nm_needed,
                                  inducing_pts, lambda, task, w0)

  dots <- list(...)
  fitted_model <- run_optimizer(model, optimizer, preconditioner, iters,
                                 verbose, dots)
  return(fitted_model)
}

# --- FullKRR S3 methods -----------------------------------------------------

#' @export
LinOp.FullKRR <- function(model, v) {
  if (is.matrix(v)) return(model$K %*% v + model$lambda * v)
  return(as.vector(model$K %*% v) + model$lambda * v)
}

#' @export
ComputeMetrics.FullKRR <- function(model, v) {
  v_lin_op <- LinOp(model, v)
  residual <- v_lin_op - model$y
  metrics <- list(
    rel_residual = Norm2(residual) / model$y_norm,
    train_loss = 0.5 * sum(v * v_lin_op) - sum(model$y * v)
  )
  return(metrics)
}

#' Predict from a fitted FullKRR model
#'
#' @param object A fitted \code{FullKRR} object returned by
#'   \code{\link{CreateFullKRR}}.
#' @param newdata Matrix of new data points (n_new x d).
#' @param ... Ignored.
#' @return Numeric vector of predictions.
#' @export
predict.FullKRR <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  K_new <- ComputeKernel(newdata, object$x, object$kernel_params)
  return(as.vector(K_new %*% object$w))
}

#' @export
UpdateModel.FullKRR <- function(model, x_new, y_new, ...) {
  stop("Online updates are not supported for FullKRR models because the ",
       "weight vector grows with n, requiring O(n^2) per update. ",
       "Use CreateInducingKRR() instead, which supports efficient O(m^2) ",
       "rank-1 updates via UpdateModel().")
}

# --- InducingKRR S3 methods -------------------------------------------------

#' @export
LinOp.InducingKRR <- function(model, v) {
  K_nmv <- as.vector(model$K_nm %*% v)
  return(as.vector(crossprod(model$K_nm, K_nmv)) +
           model$lambda * as.vector(model$K_mm %*% v))
}

#' @export
ComputeMetrics.InducingKRR <- function(model, v) {
  K_nmv <- as.vector(model$K_nm %*% v)
  K_mmv <- as.vector(model$K_mm %*% v)
  residual <- as.vector(crossprod(model$K_nm, K_nmv)) +
    model$lambda * K_mmv - model$K_nmTy
  metrics <- list(
    rel_residual = Norm2(residual) / model$K_nmTy_norm,
    train_loss = 0.5 * Norm2(K_nmv - model$y)^2 +
      (model$lambda / 2) * sum(v * K_mmv)
  )
  return(metrics)
}

#' Predict from a fitted InducingKRR model
#'
#' @param object A fitted \code{InducingKRR} object returned by
#'   \code{\link{CreateInducingKRR}}.
#' @param newdata Matrix of new data points (n_new x d).
#' @param ... Ignored.
#' @return Numeric vector of predictions.
#' @export
predict.InducingKRR <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  x_ind <- object$x[object$inducing_pts, , drop = FALSE]
  K_new <- ComputeKernel(newdata, x_ind, object$kernel_params)
  return(as.vector(K_new %*% object$w))
}

#' @export
UpdateModel.InducingKRR <- function(model, x_new, y_new,
                                     recompute_metrics = TRUE,
                                     recompute = FALSE,
                                     stability_threshold = 50L, ...) {
  # --- Input validation ---
  x_new <- as.numeric(x_new)
  d <- ncol(model$x)
  if (length(x_new) != d) {
    stop("x_new must have length ", d, " (the number of features), got ",
         length(x_new))
  }
  if (!is.numeric(y_new) || length(y_new) != 1L) {
    stop("y_new must be a single numeric value")
  }

  # --- Compute kernel vector between new point and inducing points ---
  x_ind <- model$x[model$inducing_pts, , drop = FALSE]
  k_new <- as.vector(ComputeKernel(matrix(x_new, nrow = 1), x_ind,
                                    model$kernel_params))

  # --- Lazy P computation / recompute ---
  if (is.null(model$P_inv) || isTRUE(recompute)) {
    A <- crossprod(model$K_nm) + model$lambda * model$K_mm
    model$P_inv <- solve(A)
    model$n_updates <- 0L
  }

  # --- Sherman-Morrison rank-1 update on P ---
  P <- model$P_inv
  v <- as.vector(P %*% k_new)
  denom <- 1 + sum(k_new * v)

  if (abs(denom) < .Machine$double.eps * 100) {
    warning("Sherman-Morrison denominator near zero (", denom,
            "); update may be numerically unstable. ",
            "Consider recompute = TRUE.")
  }

  P <- P - tcrossprod(v) / denom
  # Symmetrize to prevent drift
  model$P_inv <- (P + t(P)) / 2

  # --- Update RHS ---
  model$K_nmTy <- model$K_nmTy + k_new * y_new

  # --- New weights ---
  model$w <- as.vector(model$P_inv %*% model$K_nmTy)

  # --- Append data ---
  model$x <- rbind(model$x, matrix(x_new, nrow = 1))
  model$y <- c(model$y, y_new)
  model$K_nm <- rbind(model$K_nm, matrix(k_new, nrow = 1))
  model$n <- model$n + 1L

  # --- Update derived quantities ---
  model$y_norm <- Norm2(model$y)
  model$K_nmTy_norm <- Norm2(model$K_nmTy)
  model$ym_scaled <- (model$lambda / 2) * model$y[model$inducing_pts]

  # --- Stability tracking ---
  model$n_updates <- model$n_updates + 1L
  if (model$n_updates >= stability_threshold) {
    message("UpdateModel: ", model$n_updates, " rank-1 updates accumulated. ",
            "Consider calling UpdateModel(..., recompute = TRUE) to refresh ",
            "the inverse for numerical stability.")
  }

  # --- Metrics ---
  if (isTRUE(recompute_metrics)) {
    model$metrics <- ComputeMetrics(model, model$w)
  } else {
    model$metrics <- NULL
  }

  return(model)
}

# --- FullKRR internal functions ---------------------------------------------

GetBlockGrad <- function(model, w, block) {
  Kbn <- ComputeKernel(model$x[block, , drop = FALSE], model$x,
                        model$kernel_params)
  return(as.vector(Kbn %*% w) + model$lambda * model$w[block] -
           model$y[block])
}

GetBlockLinOps <- function(model, block) {
  Kb <- ComputeKernel(
    model$x[block, , drop = FALSE],
    model$x[block, , drop = FALSE],
    model$kernel_params
  )
  lambda <- model$lambda
  kb_lin_op <- function(v) {
    if (is.matrix(v)) return(Kb %*% v)
    return(as.vector(Kb %*% v))
  }
  kb_lin_op_reg <- function(v) {
    if (is.matrix(v)) return(Kb %*% v + lambda * v)
    return(as.vector(Kb %*% v) + lambda * v)
  }
  kb_trace <- ComputeKernelTrace(length(block), model$kernel_params)
  return(list(
    kb_lin_op = kb_lin_op,
    kb_lin_op_reg = kb_lin_op_reg,
    kb_trace = kb_trace
  ))
}

GetKernelFn <- function(model) {
  kp <- model$kernel_params
  K_fn <- function(x_i, x_j, get_row = FALSE) {
    if (get_row) {
      return(ComputeKernelRow(x_i, x_j, kp))
    } else {
      return(ComputeKernel(x_i, x_j, kp))
    }
  }
  return(K_fn)
}

GetFullLinOp <- function(model) {
  if (inherits(model, "InducingKRR")) {
    K_mat <- model$K_mm
    sz <- model$m
  } else {
    K_mat <- model$K
    sz <- model$n
  }
  K_lin_op <- function(v) {
    if (is.matrix(v)) return(K_mat %*% v)
    return(as.vector(K_mat %*% v))
  }
  K_trace <- ComputeKernelTrace(sz, model$kernel_params)
  return(list(K_lin_op = K_lin_op, K_trace = K_trace))
}

GetDiag <- function(model, sz = NULL) {
  if (is.null(sz)) sz <- model$n
  return(ComputeKernelDiag(sz, model$kernel_params))
}

GetDiagFn <- function(model) {
  kp <- model$kernel_params
  return(function(n) ComputeKernelDiag(n, kp))
}

# --- InducingKRR internal functions -----------------------------------------

KnmLinOp <- function(model, v) {
  return(as.vector(model$K_nm %*% v))
}

KmmLinOp <- function(model, v) {
  if (is.matrix(v)) return(model$K_mm %*% v)
  return(as.vector(model$K_mm %*% v))
}

GetGradRegularizer <- function(model) {
  return(-((model$lambda / 2)^2) * model$w + model$ym_scaled)
}

GetSelectionIdx <- function(v1, v2) {
  common_mask <- v2 %in% v1
  common_in_v2 <- v2[common_mask]
  indices_in_v1 <- which(v1 %in% common_in_v2)
  indices_in_v2 <- vapply(
    v1[indices_in_v1],
    function(elem) which(v2 == elem)[1],
    integer(1)
  )
  return(list(indices_in_v1 = indices_in_v1, indices_in_v2 = indices_in_v2))
}

GetTableAux <- function(model, idx, w, table) {
  sel <- GetSelectionIdx(model$inducing_pts, idx)
  w_selected <- rep(0, length(idx))
  if (length(sel$indices_in_v1) > 0 && length(sel$indices_in_v2) > 0) {
    w_selected[sel$indices_in_v2] <- w[sel$indices_in_v1]
  }
  K_nm_idx <- ComputeKernel(
    model$x[idx, , drop = FALSE],
    model$x[model$inducing_pts, , drop = FALSE],
    model$kernel_params
  )
  new_weights <- model$n * (
    as.vector(K_nm_idx %*% w) +
      (model$lambda / 2) * w_selected - model$y[idx]
  )
  weight_diff <- new_weights - table[idx]
  weight_diff_selected <- rep(0, model$m)
  if (length(sel$indices_in_v1) > 0 && length(sel$indices_in_v2) > 0) {
    weight_diff_selected[sel$indices_in_v1] <- weight_diff[sel$indices_in_v2]
  }
  aux <- as.vector(crossprod(K_nm_idx, weight_diff)) +
    (model$lambda / 2) * weight_diff_selected
  return(list(new_weights = new_weights, aux = aux))
}

GetSubsampledLinOps <- function(model, bH, bH2) {
  x_ind <- model$x[model$inducing_pts, , drop = FALSE]

  hess_pts <- sample.int(model$n, bH)
  K_sm <- ComputeKernel(model$x[hess_pts, , drop = FALSE], x_ind,
                        model$kernel_params)

  hess_pts_lr <- sample.int(model$n, bH2)
  K_sm_lr <- ComputeKernel(model$x[hess_pts_lr, , drop = FALSE], x_ind,
                           model$kernel_params)

  sel <- GetSelectionIdx(model$inducing_pts, hess_pts)
  idx_ind <- sel$indices_in_v1
  idx_hess <- sel$indices_in_v2

  sel_lr <- GetSelectionIdx(model$inducing_pts, hess_pts_lr)
  idx_ind_lr <- sel_lr$indices_in_v1
  idx_hess_lr <- sel_lr$indices_in_v2

  m <- model$m
  n <- model$n
  lambda <- model$lambda

  SubLinOp <- function(v, K_sub, bH_sub, idx_in_ind, idx_in_hess) {
    adj_factor <- n / bH_sub
    K_sub_v <- as.vector(K_sub %*% v)

    if (is.matrix(v)) {
      v_selected <- matrix(0, bH_sub, ncol(v))
      K_sub_v_selected <- matrix(0, m, ncol(v))
    } else {
      v_selected <- rep(0, bH_sub)
      K_sub_v_selected <- rep(0, m)
    }

    if (length(idx_in_ind) > 0 && length(idx_in_hess) > 0) {
      if (is.matrix(v)) {
        K_sub_v_selected[idx_in_ind, ] <- K_sub_v[idx_in_hess, ]
        v_selected[idx_in_hess, ] <- v[idx_in_ind, ]
      } else {
        K_sub_v_selected[idx_in_ind] <- K_sub_v[idx_in_hess]
        v_selected[idx_in_hess] <- v[idx_in_ind]
      }
    }

    result <- adj_factor * (
      as.vector(crossprod(K_sub, K_sub_v)) +
        (lambda / 2) * (K_sub_v_selected +
                         as.vector(crossprod(K_sub, v_selected)))
    )
    return(result)
  }

  precond_lin_op <- function(v) {
    SubLinOp(v, K_sm, bH, idx_ind, idx_hess)
  }
  lr_lin_op <- function(v) {
    SubLinOp(v, K_sm_lr, bH2, idx_ind_lr, idx_hess_lr)
  }

  K_fro_norm2 <- sum(K_sm^2)
  ind_trace <- ComputeKernelTrace(length(idx_ind), model$kernel_params)
  trace_val <- (n / bH) * (K_fro_norm2 + lambda * ind_trace)

  return(list(
    precond_lin_op = precond_lin_op,
    lr_lin_op = lr_lin_op,
    trace = trace_val
  ))
}

GetKmmTrace <- function(model) {
  return(ComputeKernelTrace(model$m, model$kernel_params))
}
