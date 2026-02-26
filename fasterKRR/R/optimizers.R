# Optimizer constructors and S3 methods for fast kernel ridge regression.
#
# Provides EigenPro2, EigenPro3, EigenPro4, ASkotch, ASkotchV2, PCG,
# and Mimosa optimizers. Each returns an S3 object; Step() is the
# S3 generic that returns updated state. All functions are internal.

# --- S3 Generic -------------------------------------------------------------

#' @keywords internal
Step <- function(state) {
  UseMethod("Step")
}

# --- EigenPro shared helpers ------------------------------------------------

GetTopEigensys <- function(model, block_sz, r, probs) {
  block <- GetBlock(probs, block_sz)
  ops <- GetBlockLinOps(model, block)
  n_block <- length(block)
  Ks <- ops$kb_lin_op(diag(n_block))

  if (requireNamespace("RSpectra", quietly = TRUE)) {
    eig <- RSpectra::eigs_sym(Ks / n_block, k = r + 1, which = "LM")
    eigvals <- eig$values
    eigvecs <- eig$vectors / sqrt(n_block)
  } else {
    eig <- eigen(Ks / n_block, symmetric = TRUE)
    eigvals <- eig$values[1:(r + 1)]
    eigvecs <- eig$vectors[, 1:(r + 1)] / sqrt(n_block)
  }

  beta <- max(diag(Ks))
  tail_eigval <- eigvals[r + 1]
  eigvals <- eigvals[1:r]
  eigvecs <- eigvecs[, 1:r, drop = FALSE]

  return(list(
    eigvals = eigvals,
    eigvecs = eigvecs,
    beta = beta,
    tail_eigval = tail_eigval,
    block = block
  ))
}

ComputeBgEta <- function(n, bg, eigval, beta) {
  if (is.null(bg)) {
    bg <- min(as.integer(beta / eigval + 1), n %/% 10)
  }
  generator <- CreateMinibatchGenerator(n, bg)

  if (bg < beta / eigval + 1) {
    eta <- bg / beta
  } else {
    eta <- 2 * bg / (beta + (bg - 1) * eigval)
  }
  return(list(eta = eta / bg, bg = bg, generator = generator))
}

# --- EigenPro2 --------------------------------------------------------------

CreateEigenPro2 <- function(model, block_sz, r, bg = NULL, gamma = 0.95) {
  probs <- rep(1 / model$n, model$n)
  K_fn <- GetKernelFn(model)

  eig <- GetTopEigensys(model, block_sz, r, probs)
  scale <- (eig$eigvals[1] / eig$tail_eigval)^gamma
  diag_vals <- (1 - (eig$tail_eigval / eig$eigvals)^gamma) / eig$eigvals
  eigvecs <- eig$eigvecs

  apply_precond <- function(v, kmat) {
    return(eigvecs %*% (diag_vals * crossprod(eigvecs, kmat %*% v)))
  }

  new_top_eigval <- eig$eigvals[1] / scale
  bg_result <- ComputeBgEta(model$n, bg, new_top_eigval, eig$beta)

  state <- list(
    model = model,
    K_fn = K_fn,
    apply_precond = apply_precond,
    eta = bg_result$eta,
    block = eig$block,
    generator = bg_result$generator
  )
  class(state) <- "EigenPro2"
  return(state)
}

#' @export
Step.EigenPro2 <- function(state) {
  idx <- state$generator()
  grad <- GetBlockGrad(state$model, state$model$w, idx)
  state$model$w[idx] <- state$model$w[idx] - state$eta * grad
  Kbm <- state$K_fn(
    state$model$x[state$block, , drop = FALSE],
    state$model$x[idx, , drop = FALSE],
    get_row = FALSE
  )
  d <- state$apply_precond(grad, Kbm)
  state$model$w[state$block] <- state$model$w[state$block] + state$eta * d
  return(state)
}

# --- EigenPro3 --------------------------------------------------------------

CreateEigenPro3 <- function(model, block_sz, r, bg = NULL,
                             proj_inner_iters = 10) {
  probs <- rep(1 / model$n, model$n)
  K_fn <- GetKernelFn(model)

  eig <- GetTopEigensys(model, block_sz, r, probs)
  diag_vals <- (1 - eig$tail_eigval / eig$eigvals) / eig$eigvals
  eigvecs <- eig$eigvecs

  apply_precond <- function(v, kmat) {
    return(eigvecs %*% (diag_vals * crossprod(eigvecs, kmat %*% v)))
  }

  bg_result <- ComputeBgEta(model$n, bg, eig$tail_eigval, eig$beta)

  Kzb <- K_fn(
    model$x[model$inducing_pts, , drop = FALSE],
    model$x[eig$block, , drop = FALSE],
    get_row = FALSE
  )

  state <- list(
    model = model,
    K_fn = K_fn,
    apply_precond = apply_precond,
    eta = bg_result$eta,
    block = eig$block,
    generator = bg_result$generator,
    Kzb = Kzb,
    h = rep(0, length(model$w)),
    proj_inner_iters = proj_inner_iters
  )
  class(state) <- "EigenPro3"
  return(state)
}

#' @export
Step.EigenPro3 <- function(state) {
  idx <- state$generator()
  Kmz <- state$K_fn(
    state$model$x[idx, , drop = FALSE],
    state$model$x[state$model$inducing_pts, , drop = FALSE],
    get_row = FALSE
  )
  grad <- as.vector(Kmz %*% state$model$w) - state$model$y[idx]
  Kbm <- state$K_fn(
    state$model$x[state$block, , drop = FALSE],
    state$model$x[idx, , drop = FALSE],
    get_row = FALSE
  )
  state$h <- as.vector(crossprod(Kmz, grad)) -
    as.vector(state$Kzb %*% state$apply_precond(grad, Kbm))
  state$model$w <- state$model$w - state$eta * ProjectEP(state)
  return(state)
}

ProjectEP <- function(state) {
  proj_model <- build_full_model(
    x = state$model$x[state$model$inducing_pts, , drop = FALSE],
    y = state$h,
    kernel_params = state$model$kernel_params,
    k_tr_needed = TRUE,
    lambda = 0.0,
    task = "regression",
    w0 = rep(0, length(state$model$w))
  )
  ep2 <- CreateEigenPro2(
    proj_model,
    bg = proj_model$n %/% state$proj_inner_iters,
    block_sz = min(proj_model$n, 12000L),
    r = 100L
  )
  for (i in seq_len(state$proj_inner_iters)) {
    ep2 <- Step(ep2)
  }
  return(ep2$model$w)
}

# --- EigenPro4 --------------------------------------------------------------

CreateEigenPro4 <- function(model, block_sz, r, proj_freq,
                             bg = NULL, proj_inner_iters = 10) {
  probs <- rep(1 / model$n, model$n)
  K_fn <- GetKernelFn(model)

  eig <- GetTopEigensys(model, block_sz, r, probs)
  diag_vals <- (1 - eig$tail_eigval / eig$eigvals) / eig$eigvals
  eigvecs <- eig$eigvecs

  apply_precond <- function(v, kmat) {
    return(eigvecs %*% (diag_vals * crossprod(eigvecs, kmat %*% v)))
  }

  bg_result <- ComputeBgEta(model$n, bg, eig$tail_eigval, eig$beta)

  Kzb <- K_fn(
    model$x[model$inducing_pts, , drop = FALSE],
    model$x[eig$block, , drop = FALSE],
    get_row = FALSE
  )

  state <- list(
    model = model,
    K_fn = K_fn,
    apply_precond = apply_precond,
    eta = bg_result$eta,
    block = eig$block,
    block_sz = block_sz,
    generator = bg_result$generator,
    Kzb = Kzb,
    proj_freq = proj_freq,
    proj_inner_iters = proj_inner_iters,
    Z_tmp = integer(0),
    alpha_tmp = numeric(0),
    alpha_b = rep(0, block_sz),
    h = rep(0, length(model$w)),
    n_inner_iters = 0L
  )
  class(state) <- "EigenPro4"
  return(state)
}

#' @export
Step.EigenPro4 <- function(state) {
  idx <- state$generator()
  Kmz <- state$K_fn(
    state$model$x[idx, , drop = FALSE],
    state$model$x[state$model$inducing_pts, , drop = FALSE],
    get_row = FALSE
  )
  grad <- as.vector(Kmz %*% state$model$w) - state$model$y[idx]

  if (length(state$Z_tmp) > 0) {
    Kmz_temp <- state$K_fn(
      state$model$x[idx, , drop = FALSE],
      state$model$x[state$Z_tmp, , drop = FALSE],
      get_row = FALSE
    )
    grad <- grad + as.vector(Kmz_temp %*% state$alpha_tmp)
  }

  Kbm <- state$K_fn(
    state$model$x[state$block, , drop = FALSE],
    state$model$x[idx, , drop = FALSE],
    get_row = FALSE
  )
  grad <- grad + as.vector(crossprod(Kbm, state$alpha_b))

  state$Z_tmp <- c(state$Z_tmp, idx)
  state$alpha_tmp <- c(state$alpha_tmp, -state$eta * grad)

  state$alpha_b <- state$alpha_b +
    state$eta * state$apply_precond(grad, Kbm)
  state$h <- state$h - state$eta * as.vector(crossprod(Kmz, grad))
  state$h <- state$h +
    state$eta * as.vector(state$Kzb %*% state$apply_precond(grad, Kbm))

  state$n_inner_iters <- state$n_inner_iters + 1L
  if (state$n_inner_iters == state$proj_freq) {
    theta <- ProjectEP(state)
    state$model$w <- state$model$w - state$eta * theta
    # Reset
    state$Z_tmp <- integer(0)
    state$alpha_tmp <- numeric(0)
    state$alpha_b <- rep(0, state$block_sz)
    state$h <- rep(0, length(state$model$w))
    state$n_inner_iters <- 0L
  }
  return(state)
}

# --- ASkotch ----------------------------------------------------------------

CreateASkotch <- function(model, B, no_store_precond = TRUE,
                           precond_params = NULL, beta = 0,
                           accelerated = TRUE) {
  alpha <- (1 - beta) / 2

  blocks <- GetBlocks(model$n, B)
  props <- GetBlockProperties(model, precond_params, blocks, no_store_precond)
  S_alpha <- sum(props$block_Ls^alpha)
  block_probs <- props$block_Ls^alpha / S_alpha

  state <- list(
    model = model,
    precond_params = precond_params,
    B = B,
    no_store_precond = no_store_precond,
    beta = beta,
    accelerated = accelerated,
    alpha = alpha,
    blocks = blocks,
    block_preconds = props$block_preconds,
    block_etas = props$block_etas,
    block_Ls = props$block_Ls,
    S_alpha = S_alpha,
    block_probs = block_probs
  )

  if (accelerated) {
    state$tau <- 2 / (1 + sqrt(4 * S_alpha^2 / model$lambda + 1))
    state$gamma_param <- 1 / (state$tau * S_alpha^2)
    state$y <- model$w
    state$z <- model$w
  }

  class(state) <- "ASkotch"
  return(state)
}

#' @export
Step.ASkotch <- function(state) {
  block_idx <- sample.int(state$B, 1, prob = state$block_probs)

  if (state$no_store_precond) {
    bp <- GetBlockPrecond(
      state$model, state$precond_params, state$blocks[[block_idx]]
    )
    block_precond <- bp$precond
  } else {
    block_precond <- state$block_preconds[[block_idx]]
  }
  block <- state$blocks[[block_idx]]
  block_eta <- state$block_etas[block_idx]

  dir <- GetBlockUpdate(state$model, state$model$w, block, block_precond)

  if (state$accelerated) {
    state$y <- state$model$w
    state$y[block] <- state$y[block] - block_eta * dir

    state$z <- (1 / (1 + state$gamma_param * state$model$lambda)) *
      (state$z + state$gamma_param * state$model$lambda * state$model$w)
    state$z[block] <- state$z[block] -
      (1 / (1 + state$gamma_param * state$model$lambda)) *
      state$gamma_param /
      (state$block_probs[block_idx] *
         state$block_Ls[block_idx]^state$beta) * dir

    state$model$w <- state$tau * state$z + (1 - state$tau) * state$y
  } else {
    state$model$w[block] <- state$model$w[block] - block_eta * dir
  }
  return(state)
}

# --- ASkotchV2 --------------------------------------------------------------

CreateASkotchV2 <- function(model, block_sz, sampling_method = "uniform",
                             precond_params = NULL, mu = NULL, nu = NULL,
                             accelerated = TRUE) {
  mu <- if (!is.null(mu)) mu else model$lambda
  nu <- if (!is.null(nu)) nu else model$n / block_sz

  if (sampling_method == "rls") {
    leverage_scores <- GetLeverageScores(
      model = model,
      size_final = as.integer(sqrt(model$n)),
      lam_final = model$lambda,
      rls_oversample_param = 5
    )
    probs <- leverage_scores / sum(leverage_scores)
  } else {
    probs <- rep(1 / model$n, model$n)
  }

  state <- list(
    model = model,
    precond_params = precond_params,
    block_sz = block_sz,
    mu = mu,
    nu = nu,
    accelerated = accelerated,
    probs = probs
  )

  if (accelerated) {
    state$beta_param <- 1 - sqrt(mu / nu)
    state$gamma_param <- 1 / sqrt(mu * nu)
    state$alpha_param <- 1 / (1 + state$gamma_param * nu)
    state$v_vec <- model$w
    state$y <- model$w
  }

  class(state) <- "ASkotchV2"
  return(state)
}

#' @export
Step.ASkotchV2 <- function(state) {
  block <- GetBlock(state$probs, state$block_sz)
  props <- GetBlockProperties(
    state$model, state$precond_params, list(block), FALSE
  )
  block_precond <- props$block_preconds[[1]]
  block_eta <- props$block_etas[1]

  eval_loc <- if (state$accelerated) state$y else state$model$w
  dir <- GetBlockUpdate(state$model, eval_loc, block, block_precond)

  if (state$accelerated) {
    state$model$w <- state$y
    state$model$w[block] <- state$model$w[block] - block_eta * dir
    state$v_vec <- state$beta_param * state$v_vec +
      (1 - state$beta_param) * state$y
    state$v_vec[block] <- state$v_vec[block] -
      block_eta * state$gamma_param * dir
    state$y <- state$alpha_param * state$v_vec +
      (1 - state$alpha_param) * state$model$w
  } else {
    state$model$w[block] <- state$model$w[block] - block_eta * dir
  }
  return(state)
}

# --- PCG --------------------------------------------------------------------

CreatePCG <- function(model, precond_params = NULL) {
  rhs <- if (model$inducing) model$K_nmTy else model$y
  precond <- GetPcgPrecond(model, precond_params)

  r_vec <- rhs - LinOp(model, model$w)
  z_vec <- ApplyPrecond(r_vec, precond)
  p_vec <- z_vec

  state <- list(
    model = model,
    precond = precond,
    rhs = rhs,
    r_vec = r_vec,
    z_vec = z_vec,
    p_vec = p_vec
  )
  class(state) <- "PCG"
  return(state)
}

#' @export
Step.PCG <- function(state) {
  Kp <- LinOp(state$model, state$p_vec)
  r0_dot_z0 <- sum(state$r_vec * state$z_vec)
  alpha <- r0_dot_z0 / sum(state$p_vec * Kp)
  state$model$w <- state$model$w + alpha * state$p_vec
  state$r_vec <- state$r_vec - alpha * Kp
  state$z_vec <- ApplyPrecond(state$r_vec, state$precond)
  beta <- sum(state$r_vec * state$z_vec) / r0_dot_z0
  state$p_vec <- state$z_vec + beta * state$p_vec
  return(state)
}

# --- Mimosa -----------------------------------------------------------------

CreateMimosa <- function(model, bg, bH, bH2, precond_params = NULL) {
  result <- GetPrecondL(model, precond_params, bH, bH2)

  state <- list(
    model = model,
    bg = bg,
    bH = bH,
    bH2 = bH2,
    precond = result$precond,
    eta = 0.5 / result$L,
    generator = CreateMinibatchGenerator(model$n, bg),
    table = rep(0, model$n),
    u = rep(0, model$m)
  )
  class(state) <- "Mimosa"
  return(state)
}

#' @export
Step.Mimosa <- function(state) {
  idx <- state$generator()
  result <- GetTableAux(state$model, idx, state$model$w, state$table)
  new_weights <- result$new_weights
  aux <- result$aux

  g <- state$u + (1 / length(idx)) * aux
  state$u <- state$u + (1 / state$model$n) * aux
  state$table[idx] <- new_weights

  dir <- ApplyPrecond(
    g + GetGradRegularizer(state$model), state$precond
  )
  state$model$w <- state$model$w - state$eta * dir
  return(state)
}
