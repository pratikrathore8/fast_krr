\name{CreateFullKRR}
\alias{CreateFullKRR}
\title{Fit a Full Kernel Ridge Regression Model}
\description{
Fit a kernel ridge regression model using the full \code{n x n} kernel matrix.
This is the primary entry point for exact KRR. The function constructs the
kernel matrix, selects the specified optimizer and optional preconditioner,
runs the iterative solver, and returns the fitted model.
}
\usage{
CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, nu = NULL, lambda = 1e-3,
              task = "regression", optimizer = "pcg", preconditioner = NULL,
              iters = 30L, verbose = FALSE, w0 = NULL,
              fast_cv = TRUE, cv_folds = 5L,
              lambda_grid = NULL, \dots)
}
\arguments{
\item{x}{Training feature matrix (\code{n x d}).}

\item{y}{Training target vector of length \code{n}. For regression this is a
  numeric vector; for classification it should contain class labels.}

\item{kernel}{Character string specifying the kernel type. One of
  \code{"rbf"} (default), \code{"l1_laplace"}, or \code{"matern"}.}

\item{sigma}{Positive numeric kernel bandwidth parameter (default \code{1.0}).
  Used by all kernel types.}

\item{nu}{Smoothness parameter for the Matern kernel. Required when
  \code{kernel = "matern"}; ignored otherwise. Common choices are
  \code{0.5}, \code{1.5}, and \code{2.5}.}

\item{lambda}{Regularization parameter. A single positive numeric value uses
  that value directly (default \code{1e-3}). Larger values increase
  regularization. If \code{NULL}, the best lambda is selected automatically
  via cross-validation using the \pkg{CVST} package over a default grid. If
  a numeric vector of length greater than one (e.g.
  \code{c(1e-4, 1e-3, 1e-2)}), cross-validation is performed over those
  candidate values and the best one is selected.}

\item{task}{Character string: \code{"regression"} (default) or
  \code{"classification"}.}

\item{optimizer}{Character string specifying the iterative solver. One of
  \code{"pcg"} (preconditioned conjugate gradient, default),
  \code{"eigenpro2"}, \code{"askotch"}, or \code{"askotch_v2"}. Note that
  \code{"eigenpro3"}, \code{"eigenpro4"}, and \code{"mimosa"} are only
  available for \code{\link{CreateInducingKRR}}.}

\item{preconditioner}{A named list specifying the preconditioner, or
  \code{NULL} (default) for no preconditioning. Examples:
  \code{list(type = "newton", rho = "regularization")},
  \code{list(type = "nystrom", r = 20, rho = "regularization")},
  \code{list(type = "partial_cholesky", r = 20, rho = "regularization",
  mode = "greedy")}.}

\item{iters}{Positive integer giving the number of solver iterations
  (default \code{30L}).}

\item{verbose}{Logical; if \code{TRUE}, print progress information during
  fitting (default \code{FALSE}).}

\item{w0}{Optional initial weight vector of length \code{n}. If \code{NULL}
  (default), a zero vector is used.}

\item{fast_cv}{Logical; if \code{TRUE} (default), use \code{CVST::fastCV}
  (sequential testing) for lambda selection; if \code{FALSE}, use standard
  k-fold CV via \code{CVST::CV}. Only used when \code{lambda = NULL}.}

\item{cv_folds}{Integer number of folds for standard CV (default \code{5L}).
  Ignored when \code{fast_cv = TRUE}. Only used when \code{lambda = NULL}.}

\item{lambda_grid}{Numeric vector of candidate lambda values for
  cross-validation. If \code{NULL} (default), uses
  \code{seq(1e-10, 1e-3, length.out = 100)}. Only used when
  \code{lambda = NULL}.}

\item{\dots}{Additional arguments passed to the optimizer or preconditioner.}
}
\value{
An object of class \code{"FullKRR"} (a list) containing the fitted model,
including the solution weight vector \code{w} and all information needed for
prediction via \code{\link{predict.FullKRR}}.
}
\examples{
## Basic regression example
set.seed(42)
n <- 100; d <- 3
x <- matrix(rnorm(n * d), n, d)
y <- sin(x[, 1]) + 0.1 * rnorm(n)

model <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 1e-3)

## Predict on new data
x_new <- matrix(rnorm(20 * d), 20, d)
preds <- predict(model, x_new)

## With a Nystrom preconditioner
model2 <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 1e-3,
                         optimizer = "pcg",
                         preconditioner = list(type = "nystrom", r = 20,
                                               rho = "regularization"),
                         iters = 50L, verbose = TRUE)
}
