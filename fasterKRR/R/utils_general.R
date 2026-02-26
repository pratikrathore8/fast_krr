# General optimizer utilities.
#
# Provides preconditioner application, largest eigenvalue estimation
# via power iteration, and leverage score estimation via BLESS.

ApplyPrecond <- function(v, precond) {
  if (!is.null(precond)) {
    return(InvLinOp(precond, v))
  }
  return(v)
}

EstimateLargestEigenvalue <- function(mat_lin_op, precond, n) {
  v <- stats::rnorm(n)
  v <- v / Norm2(v)
  max_eig <- NULL

  for (i in seq_len(10)) {
    v_old <- v
    v <- ApplyPrecond(v, precond)
    v <- mat_lin_op(v)
    max_eig <- sum(v_old * v)
    v <- v / Norm2(v)
  }
  return(max_eig)
}

GetLeverageScores <- function(model, size_final, lam_final, rls_oversample_param) {
  K_fn <- GetKernelFn(model)
  K_diag_fn <- GetDiagFn(model)
  bless_result <- BlessSize(model$x, K_fn, K_diag_fn, size_final,
                            rls_oversample_param)
  rls_approx <- EstimateRlsBless(bless_result$dict, model$x, K_fn,
                                  K_diag_fn, lam_final)
  return(rls_approx)
}
