# SGD-related utilities.
#
# Provides preconditioner + learning rate estimation for stochastic methods.

GetPrecondL <- function(model, precond_params, bH, bH2) {
  sub <- GetSubsampledLinOps(model, bH, bH2)

  update_params <- NULL
  if (!is.null(precond_params)) {
    ptype <- precond_params$type
    if (ptype == "newton") {
      update_params <- list(K_lin_op = sub$lr_lin_op, n = model$m)
    } else if (ptype == "nystrom") {
      update_params <- list(
        K_lin_op = sub$precond_lin_op,
        K_trace = sub$trace,
        n = model$m
      )
    }
  }
  precond <- InitPrecond(precond_params, update_params, model$lambda)
  L <- EstimateLargestEigenvalue(sub$lr_lin_op, precond, model$m)
  return(list(precond = precond, L = L))
}
