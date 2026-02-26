# PCG preconditioner setup utilities.
#
# Handles preconditioner construction for the PCG optimizer,
# supporting full KRR and inducing KRR models.

GetPcgPrecond <- function(model, precond_params) {
  update_params <- NULL

  if (!is.null(precond_params)) {
    ptype <- precond_params$type

    if (ptype == "falkon") {
      K_mm_lin_op <- function(v) KmmLinOp(model, v)
      update_params <- list(
        K_mm_lin_op = K_mm_lin_op,
        K_mm_trace = GetKmmTrace(model),
        n = model$n,
        m = model$m
      )
    } else if (ptype == "newton") {
      full <- GetFullLinOp(model)
      update_params <- list(K_lin_op = full$K_lin_op, n = model$n)
    } else if (ptype == "nystrom") {
      full <- GetFullLinOp(model)
      update_params <- list(
        K_lin_op = full$K_lin_op,
        K_trace = full$K_trace,
        n = model$n
      )
    } else if (ptype == "partial_cholesky") {
      K_fn <- GetKernelFn(model)
      K_diag <- GetDiag(model)
      blk_size <- precond_params$blk_size
      update_params <- list(
        K_fn = K_fn,
        K_diag = K_diag,
        x = model$x,
        blk_size = blk_size
      )
    }
  }
  return(InitPrecond(precond_params, update_params, model$lambda))
}
