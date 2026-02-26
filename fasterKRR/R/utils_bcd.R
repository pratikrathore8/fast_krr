# Block coordinate descent utilities.
#
# Provides block partitioning, block preconditioner construction,
# block property estimation, and block update computation.

GetBlocks <- function(n, B) {
  idx <- sample.int(n)
  block_size <- n %/% B
  remainder <- n %% B
  sizes <- rep(block_size, B)
  if (remainder > 0) {
    sizes[1:remainder] <- sizes[1:remainder] + 1L
  }

  blocks <- vector("list", B)
  pos <- 1L
  for (i in seq_len(B)) {
    blocks[[i]] <- idx[pos:(pos + sizes[i] - 1L)]
    pos <- pos + sizes[i]
  }
  return(blocks)
}

GetBlockPrecond <- function(model, precond_params, block) {
  ops <- GetBlockLinOps(model, block)

  update_params <- NULL
  if (!is.null(precond_params)) {
    ptype <- precond_params$type
    if (ptype == "newton") {
      update_params <- list(K_lin_op = ops$kb_lin_op, n = length(block))
    } else if (ptype == "nystrom") {
      update_params <- list(
        K_lin_op = ops$kb_lin_op,
        K_trace = ops$kb_trace,
        n = length(block)
      )
    } else if (ptype == "partial_cholesky") {
      K_fn <- GetKernelFn(model)
      K_diag <- GetDiag(model, sz = length(block))
      blk_size <- precond_params$blk_size
      update_params <- list(
        K_fn = K_fn,
        K_diag = K_diag,
        x = model$x[block, , drop = FALSE],
        blk_size = blk_size
      )
    }
  }
  precond <- InitPrecond(precond_params, update_params, model$lambda)
  return(list(precond = precond, block_lin_op_reg = ops$kb_lin_op_reg))
}

GetBlockPrecondL <- function(model, precond_params, block) {
  result <- GetBlockPrecond(model, precond_params, block)
  L <- EstimateLargestEigenvalue(
    result$block_lin_op_reg, result$precond, length(block)
  )
  return(list(precond = result$precond, L = L))
}

GetBlockProperties <- function(model, precond_params, blocks, no_store_precond) {
  block_preconds <- list()
  block_etas <- numeric(length(blocks))
  block_Ls <- numeric(length(blocks))

  for (i in seq_along(blocks)) {
    result <- GetBlockPrecondL(model, precond_params, blocks[[i]])
    if (!no_store_precond) {
      block_preconds[i] <- list(result$precond)
    }
    block_Ls[i] <- result$L
    block_etas[i] <- 1 / result$L
  }
  return(list(
    block_preconds = block_preconds,
    block_etas = block_etas,
    block_Ls = block_Ls
  ))
}

GetBlockUpdate <- function(model, w, block, precond) {
  gb <- GetBlockGrad(model, w, block)
  return(ApplyPrecond(gb, precond))
}

GetBlock <- function(probs, block_sz) {
  return(sample.int(length(probs), block_sz, replace = FALSE, prob = probs))
}
