# BLESS algorithm for approximate ridge leverage score sampling.
#
# Adapted from the Python fast_krr implementation, which was in turn
# adapted from https://github.com/guilgautier/DPPy/blob/master/dppy/bless.py.

StableFilter <- function(eigenvec, eigenval) {
  # Filter eigendecomposition to retain only numerically stable eigenvalues.
  #
  # Args:
  #   eigenvec: n x n eigenvector matrix.
  #   eigenval: Length-n eigenvalue vector.
  #
  # Returns:
  #   List with eigenvec_thin and eigenval_thin.
  n <- nrow(eigenvec)
  thresh <- max(abs(eigenval)) * max(length(eigenval)) * .Machine$double.eps
  stable_eig <- !sapply(eigenval, function(e) {
    abs(e) < thresh
  })

  if (any(eigenval <= -thresh)) {
    stop("Some eigenvalues of a PSD matrix are negative. Min eig: ",
         min(eigenval))
  }
  return(list(
    eigenvec_thin = eigenvec[, stable_eig, drop = FALSE],
    eigenval_thin = eigenval[stable_eig]
  ))
}

StableInvertRoot <- function(eigenvec, eigenval) {
  # Compute the pseudo-inverse square root using stable eigenvalues.
  #
  # Args:
  #   eigenvec: n x n eigenvector matrix.
  #   eigenval: Length-n eigenvalue vector.
  #
  # Returns:
  #   List with eigenvec_thin and eigenval_thin_inv_root.
  filtered <- StableFilter(eigenvec, eigenval)
  eigenval_thin_inv_root <- 1 / sqrt(filtered$eigenval_thin)
  return(list(
    eigenvec_thin = filtered$eigenvec_thin,
    eigenval_thin_inv_root = eigenval_thin_inv_root
  ))
}

EstimateRlsBless <- function(D, X, K_fn, K_diag_fn, lam_new) {
  # Estimate ridge leverage scores given a BLESS dictionary.
  #
  # Args:
  #   D: Named list (CentersDictionary) with X, probs, lam, rls_oversample.
  #   X: Full data matrix (n x d).
  #   K_fn: Kernel function.
  #   K_diag_fn: Kernel diagonal function.
  #   lam_new: Regularization parameter.
  #
  # Returns:
  #   Length-n vector of RLS estimates.
  n <- nrow(X)
  diag_norm <- K_diag_fn(n)

  get_row <- (nrow(D$X) == 1)
  K_DU <- K_fn(D$X, X, get_row = get_row)
  if (is.vector(K_DU)) K_DU <- matrix(K_DU, nrow = 1)
  K_DD <- K_fn(D$X, D$X, get_row = get_row)
  if (is.vector(K_DD)) K_DD <- matrix(K_DD, nrow = 1, ncol = 1)

  decomp <- svd(K_DD + lam_new * diag(D$probs, nrow = length(D$probs)))
  inv_root <- StableInvertRoot(decomp$u, decomp$d)
  E <- inv_root$eigenval_thin_inv_root * t(inv_root$eigenvec_thin)

  X_precond <- E %*% K_DU
  rls_estimate <- (diag_norm - colSums(X_precond^2)) / lam_new

  if (any(rls_estimate < 0)) {
    stop("Some estimated RLS is negative. Min: ", min(rls_estimate))
  }
  return(rls_estimate)
}

ReduceLambda <- function(X_data, K_fn, K_diag_fn, dict, lam_new,
                         rls_oversample_parameter = NULL) {
  # Reduce lambda in the BLESS algorithm by one step.
  #
  # Args:
  #   X_data: Full data matrix.
  #   K_fn: Kernel function.
  #   K_diag_fn: Kernel diagonal function.
  #   dict: Current CentersDictionary (named list).
  #   lam_new: New (smaller) lambda value.
  #   rls_oversample_parameter: Oversampling factor.
  #
  # Returns:
  #   Updated CentersDictionary.
  n <- nrow(X_data)
  if (is.null(rls_oversample_parameter)) {
    rls_oversample_parameter <- dict$rls_oversample
  }

  red_ratio <- dict$lam / lam_new
  if (red_ratio < 1.0) {
    stop("red_ratio = ", red_ratio, " is less than 1.0")
  }

  diag_val <- K_diag_fn(n)
  ucb <- pmin(rls_oversample_parameter * diag_val / (diag_val + lam_new), 1.0)
  U <- stats::runif(n) <= ucb
  u <- sum(U)

  if (u <= 0) {
    stop("No point selected during uniform sampling step. ",
         "Expected: ", n * mean(ucb))
  }

  X_U <- X_data[U, , drop = FALSE]
  rls_estimate <- EstimateRlsBless(dict, X_U, K_fn, K_diag_fn, lam_new)

  probs <- pmin(rls_oversample_parameter * rls_estimate, ucb[U])
  probs_reject <- probs / ucb[U]

  if (any(probs < 0)) {
    stop("Some estimated probability is negative. Min: ", min(probs))
  }

  deff_estimate <- sum(probs_reject) / rls_oversample_parameter
  if (rls_oversample_parameter * deff_estimate < 1.0) {
    stop("Estimated deff is smaller than 1. ",
         "deff_estimate: ", rls_oversample_parameter * deff_estimate)
  }

  selected <- stats::runif(u) <= probs_reject
  s <- sum(selected)
  if (s <= 0) {
    stop("No point selected during RLS sampling step. ",
         "Expected: ", sum(probs_reject))
  }

  U_idx <- which(U)
  new_dict <- list(
    idx = U_idx[selected],
    X = X_U[selected, , drop = FALSE],
    probs = probs[selected],
    lam = lam_new,
    rls_oversample = rls_oversample_parameter
  )
  return(new_dict)
}

BlessSize <- function(X_data, K_fn, K_diag_fn, size_final,
                      rls_oversample_param, nb_iter_bless = NULL) {
  # Run the BLESS algorithm to build a dictionary for RLS estimation.
  #
  # Args:
  #   X_data: Full data matrix (n x d).
  #   K_fn: Kernel function.
  #   K_diag_fn: Kernel diagonal function.
  #   size_final: Target dictionary size.
  #   rls_oversample_param: Oversampling parameter.
  #   nb_iter_bless: Number of lambda-reduction iterations (or NULL for auto).
  #
  # Returns:
  #   List with dict, lam_max, lam_min.
  n <- nrow(X_data)
  diag_norm <- K_diag_fn(n)

  lam_init <- sum(diag_norm) / (size_final - 1)
  lam_max <- lam_init
  lam_final <- 1
  lam_min <- lam_final

  if (is.null(nb_iter_bless)) {
    nb_iter_bless <- ceiling(log2(lam_init))
  }

  lam_sequence <- exp(seq(log(lam_final), log(lam_init),
                           length.out = nb_iter_bless))

  ucb_init <- rls_oversample_param * diag_norm / (diag_norm + lam_init)
  selected_init <- stats::runif(n) <= ucb_init
  selected_init[1] <- TRUE
  ucb_init[1] <- rls_oversample_param

  dict <- list(
    idx = which(selected_init),
    X = X_data[selected_init, , drop = FALSE],
    probs = rep(1, sum(selected_init)) * ucb_init[selected_init],
    lam = lam_init,
    rls_oversample = rls_oversample_param
  )

  # Discard lam_init from the sequence (pop from end)
  lam_new <- lam_sequence[length(lam_sequence)]
  lam_sequence <- lam_sequence[-length(lam_sequence)]
  deff_hat_new <- size_final - 1

  while (length(lam_sequence) > 0) {
    lam_old <- lam_new
    deff_hat_old <- deff_hat_new

    lam_new <- lam_sequence[length(lam_sequence)]
    lam_sequence <- lam_sequence[-length(lam_sequence)]

    dict <- ReduceLambda(X_data, K_fn, K_diag_fn, dict, lam_new)
    deff_hat_new <- length(dict$idx) / dict$rls_oversample

    if (deff_hat_old <= (size_final - 1) / 2 &&
        (size_final - 1) / 2 <= deff_hat_new) {
      lam_max <- lam_old
    }

    if (deff_hat_new >= 2 * (size_final + 1)) {
      lam_min <- lam_new
      break
    }
  }
  return(list(dict = dict, lam_max = lam_max, lam_min = lam_min))
}
