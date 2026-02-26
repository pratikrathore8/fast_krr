# Preconditioner constructors and S3 methods for fast kernel ridge regression.
#
# Provides Newton, Nystrom, Falkon, and Partial Cholesky preconditioners.
# Each constructor creates and updates (factorizes) the preconditioner in one call.
# InvLinOp() is the only S3 generic.

# --- S3 Generic -------------------------------------------------------------

#' Apply inverse preconditioner to a vector.
#'
#' @param precond A preconditioner object.
#' @param v Input vector.
#' @return Preconditioned vector.
InvLinOp <- function(precond, v) {
  UseMethod("InvLinOp")
}

# --- Newton -----------------------------------------------------------------

#' Create a Newton Preconditioner
#'
#' Exact Cholesky-based preconditioner. Constructed and factorized in one call.
#'
#' @param rho Damping: numeric value or "regularization" to use lambd.
#' @param lambd Regularization parameter.
#' @param K_lin_op Function that computes K \%*\% v.
#' @param n Dimension.
#' @return A list of class "NewtonPrecond".
CreateNewtonPrecond <- function(rho, lambd, K_lin_op, n) {
  rho <- GetDamping(rho, lambd)
  K <- K_lin_op(diag(n))
  diag(K) <- diag(K) + rho
  L <- t(chol(K))

  precond <- list(rho = rho, L = L)
  class(precond) <- "NewtonPrecond"
  return(precond)
}

#' @export
InvLinOp.NewtonPrecond <- function(precond, v) {
  y <- forwardsolve(precond$L, v)
  y <- backsolve(t(precond$L), y)
  return(as.vector(y))
}

# --- Nystrom ----------------------------------------------------------------

#' Create a Nystrom Preconditioner
#'
#' Low-rank Nystrom approximation preconditioner.
#'
#' @param r Rank of the approximation.
#' @param rho Damping: numeric, "regularization", or "damped".
#' @param lambd Regularization parameter.
#' @param K_lin_op Function that computes K \%*\% v.
#' @param K_trace Trace of K.
#' @param n Dimension.
#' @return A list of class "NystromPrecond".
CreateNystromPrecond <- function(r, rho, lambd, K_lin_op, K_trace, n) {
  result <- GetDampingNystrom(rho, lambd)
  rho_val <- result$rho
  rho_damped <- result$rho_damped

  Phi <- matrix(stats::rnorm(n * r), n, r) / sqrt(n)
  Phi <- qr.Q(qr(Phi))
  shift <- .Machine$double.eps * K_trace
  Y_shifted <- K_lin_op(Phi) + shift * Phi
  cholesky_target <- crossprod(Phi, Y_shifted)
  C <- t(chol(cholesky_target))
  B <- t(forwardsolve(C, t(Y_shifted)))
  decomp <- svd(B, nu = ncol(B), nv = 0)
  U <- decomp$u
  S <- pmax(decomp$d^2 - shift, 0)
  S[S < .Machine$double.eps] <- 1e-4
  if (rho_damped) {
    rho_val <- lambd + S[length(S)]
  }

  precond <- list(rho = rho_val, U = U, S = S)
  class(precond) <- "NystromPrecond"
  return(precond)
}

#' @export
InvLinOp.NystromPrecond <- function(precond, v) {
  UTv <- crossprod(precond$U, v)
  v_out <- precond$U %*% (UTv / (precond$S + precond$rho)) +
    (1 / precond$rho) * (v - precond$U %*% UTv)
  return(as.vector(v_out))
}

# --- Falkon -----------------------------------------------------------------

#' Create a Falkon Preconditioner
#'
#' Double Cholesky preconditioner for inducing point methods.
#'
#' @param lambd Regularization parameter.
#' @param K_mm_lin_op Function computing K_mm \%*\% v.
#' @param K_mm_trace Trace of K_mm.
#' @param n Number of training points.
#' @param m Number of inducing points.
#' @return A list of class "FalkonPrecond".
CreateFalkonPrecond <- function(lambd, K_mm_lin_op, K_mm_trace, n, m) {
  K_mm <- K_mm_lin_op(diag(m))
  shift <- .Machine$double.eps * K_mm_trace
  TT <- chol(K_mm + shift * diag(m))
  R <- chol((n / m) * crossprod(TT) + lambd * diag(m))

  precond <- list(lambd = lambd, TT = TT, R = R)
  class(precond) <- "FalkonPrecond"
  return(precond)
}

#' @export
InvLinOp.FalkonPrecond <- function(precond, v) {
  v <- forwardsolve(t(precond$TT), v)
  v <- forwardsolve(t(precond$R), v)
  v <- backsolve(precond$R, v)
  v <- backsolve(precond$TT, v)
  return(as.vector(v))
}

# --- Partial Cholesky -------------------------------------------------------

#' Create a Partial Cholesky Preconditioner
#'
#' Low-rank preconditioner via partial pivoted Cholesky factorization.
#'
#' @param r Target rank.
#' @param rho Damping: numeric or "regularization".
#' @param lambd Regularization parameter.
#' @param K_fn Kernel function(x_i, x_j, get_row).
#' @param K_diag Diagonal of the kernel matrix.
#' @param x Data matrix (n x d).
#' @param mode "rpc" (default) or "greedy".
#' @param blk_size Block size for RPC mode (or NULL for auto).
#' @param tol Tolerance for early stopping.
#' @return A list of class "PartialCholeskyPrecond".
CreatePartialCholeskyPrecond <- function(r, rho, lambd, K_fn, K_diag, x,
                                          mode = "rpc", blk_size = NULL,
                                          tol = 1e-6) {
  if (!(mode %in% c("rpc", "greedy"))) {
    stop("PartialCholesky does not support factorization: ", mode)
  }
  rho <- GetDamping(rho, lambd)

  if (mode == "greedy") {
    result <- UpdateGreedy(K_fn, K_diag, x, r, tol)
  } else {
    if (is.null(blk_size)) {
      blk_size <- max(1L, ceiling(r / 10))
    }
    result <- UpdateRpc(K_fn, K_diag, x, r, blk_size, tol)
  }
  L <- result$L
  actual_r <- nrow(L)
  M <- t(chol(tcrossprod(L) + rho * diag(actual_r)))

  precond <- list(rho = rho, L = L, M = M)
  class(precond) <- "PartialCholeskyPrecond"
  return(precond)
}

#' @export
InvLinOp.PartialCholeskyPrecond <- function(precond, v) {
  Lv <- precond$L %*% v
  M_inv_Lv <- forwardsolve(precond$M, Lv)
  MT_inv <- backsolve(t(precond$M), M_inv_Lv)
  v_out <- (v - crossprod(precond$L, as.vector(MT_inv))) / precond$rho
  return(as.vector(v_out))
}

# --- Partial Cholesky internal helpers --------------------------------------

UpdateGreedy <- function(K_fn, K_diag, x, r, tol) {
  n <- nrow(x)
  L <- matrix(0, r, n)
  K_diag <- as.numeric(K_diag)

  for (i in seq_len(r)) {
    idx <- which.max(K_diag)
    k_idx <- K_fn(x[idx, , drop = FALSE], x, get_row = TRUE)
    if (i > 1) {
      L[i, ] <- (k_idx - crossprod(L[1:(i - 1), idx, drop = FALSE],
                                     L[1:(i - 1), , drop = FALSE])) /
        sqrt(K_diag[idx])
    } else {
      L[i, ] <- k_idx / sqrt(K_diag[idx])
    }
    K_diag <- K_diag - L[i, ]^2
    K_diag <- pmax(K_diag, 0)
    if (max(K_diag) <= tol) break
  }
  return(list(L = L))
}

UpdateRpc <- function(K_fn, K_diag, x, r, blk_size, stoptol) {
  K_diag <- as.numeric(K_diag)
  n <- length(K_diag)
  orig_trace <- sum(K_diag)

  L <- matrix(0, r, n)
  rows <- matrix(0, r, n)
  arr_idx <- integer(r)

  counter <- 0L
  while (counter < r) {
    probs <- K_diag / sum(K_diag)
    idx <- sample.int(n, blk_size, replace = TRUE, prob = probs)

    Kbb <- GetBlockKernel(x, idx, K_fn)
    if (counter > 0) {
      H <- Kbb - crossprod(L[1:counter, idx, drop = FALSE])
    } else {
      H <- Kbb
    }
    result <- RejectionCholesky(H)
    C_mat <- result$C
    accepted <- result$accepted
    num_sel <- length(accepted)

    if (num_sel == 0) next

    if (num_sel > r - counter) {
      num_sel <- r - counter
      accepted <- accepted[1:num_sel]
      C_mat <- C_mat[1:num_sel, 1:num_sel, drop = FALSE]
    }

    idx <- idx[accepted]
    sel_range <- (counter + 1):(counter + num_sel)
    arr_idx[sel_range] <- idx

    rows[sel_range, ] <- GetRowKernel(x, idx, K_fn)
    if (counter > 0) {
      L[sel_range, ] <- rows[sel_range, , drop = FALSE] -
        crossprod(
          L[1:counter, idx, drop = FALSE],
          L[1:counter, , drop = FALSE]
        )
    } else {
      L[sel_range, ] <- rows[sel_range, , drop = FALSE]
    }
    L[sel_range, ] <- solve(C_mat, L[sel_range, , drop = FALSE])
    K_diag <- K_diag - colSums(L[sel_range, , drop = FALSE]^2)
    K_diag <- pmax(K_diag, 0)
    counter <- counter + num_sel

    if (stoptol > 0 && sum(K_diag) <= stoptol * orig_trace) {
      L <- L[1:counter, , drop = FALSE]
      rows <- rows[1:counter, , drop = FALSE]
      break
    }
  }
  return(list(L = L))
}

GetBlockKernel <- function(x, block, K_fn) {
  b <- length(block)
  if (b == 1) {
    return(matrix(K_fn(x[block, , drop = FALSE],
                       x[block, , drop = FALSE],
                       get_row = TRUE), 1, 1))
  }
  return(K_fn(x[block, , drop = FALSE],
               x[block, , drop = FALSE],
               get_row = FALSE))
}

GetRowKernel <- function(x, row_indices, K_fn) {
  b <- length(row_indices)
  if (b == 1) {
    return(matrix(K_fn(x[row_indices, , drop = FALSE], x, get_row = TRUE),
                   nrow = 1))
  }
  return(K_fn(x[row_indices, , drop = FALSE], x, get_row = FALSE))
}

RejectionCholesky <- function(H) {
  b <- nrow(H)
  u <- diag(H)
  idx <- integer(0)
  C_mat <- matrix(0, b, b)

  for (j in seq_len(b)) {
    if (stats::runif(1) * u[j] < H[j, j]) {
      idx <- c(idx, j)
      C_mat[j:b, j] <- H[j:b, j] / sqrt(H[j, j])
      if (j < b) {
        for (ii in (j + 1):b) {
          H[ii, (j + 1):b] <- H[ii, (j + 1):b] -
            C_mat[ii, j] * C_mat[(j + 1):b, j]
        }
      }
    }
  }

  if (length(idx) == 0) {
    return(list(C = matrix(0, 0, 0), accepted = integer(0)))
  }
  C_out <- C_mat[idx, idx, drop = FALSE]
  return(list(C = C_out, accepted = idx))
}

# --- Preconditioner initialization -------------------------------------------

GetDamping <- function(rho, lambd) {
  if (is.numeric(rho)) {
    return(rho)
  } else if (identical(rho, "regularization")) {
    return(lambd)
  }
  stop("Unknown rho specification: ", rho)
}

GetDampingNystrom <- function(rho, lambd) {
  if (is.numeric(rho)) {
    return(list(rho = rho, rho_damped = FALSE))
  } else if (identical(rho, "regularization")) {
    return(list(rho = lambd, rho_damped = FALSE))
  } else if (identical(rho, "damped")) {
    return(list(rho = NULL, rho_damped = TRUE))
  }
  stop("Unknown rho specification: ", rho)
}

InitPrecond <- function(precond_params, update_params, lambd) {
  if (is.null(precond_params)) {
    return(NULL)
  }

  ptype <- precond_params$type
  sub_params <- precond_params[!(names(precond_params) %in% c("type", "blk_size"))]

  if (ptype == "newton") {
    precond <- CreateNewtonPrecond(
      rho = sub_params$rho, lambd = lambd,
      K_lin_op = update_params$K_lin_op, n = update_params$n
    )
  } else if (ptype == "nystrom") {
    precond <- CreateNystromPrecond(
      r = sub_params$r, rho = sub_params$rho, lambd = lambd,
      K_lin_op = update_params$K_lin_op,
      K_trace = update_params$K_trace, n = update_params$n
    )
  } else if (ptype == "falkon") {
    precond <- CreateFalkonPrecond(
      lambd = lambd,
      K_mm_lin_op = update_params$K_mm_lin_op,
      K_mm_trace = update_params$K_mm_trace,
      n = update_params$n, m = update_params$m
    )
  } else if (ptype == "partial_cholesky") {
    precond <- CreatePartialCholeskyPrecond(
      r = sub_params$r, rho = sub_params$rho, lambd = lambd,
      K_fn = update_params$K_fn,
      K_diag = update_params$K_diag,
      x = update_params$x,
      mode = if (!is.null(sub_params$mode)) sub_params$mode else "rpc",
      blk_size = precond_params$blk_size
    )
  } else {
    stop("Unknown preconditioner type: ", ptype)
  }

  return(precond)
}
