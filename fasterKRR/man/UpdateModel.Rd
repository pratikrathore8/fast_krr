\name{UpdateModel}
\alias{UpdateModel}
\alias{UpdateModel.InducingKRR}
\alias{UpdateModel.FullKRR}
\title{Update a KRR Model with a New Observation (Online Learning)}
\description{
Return a \emph{new} \code{InducingKRR} model that incorporates a single
new observation via a rank-1 Sherman-Morrison update.  The original model
is never modified, so you can keep old snapshots:

\preformatted{
  model2 <- UpdateModel(model, x_new, y_new)   # model is unchanged
  model  <- UpdateModel(model, x_new, y_new)   # overwrite to advance
}

The system matrix inverse \eqn{P = (K_{nm}^T K_{nm} + \lambda K_{mm})^{-1}}
is stored on the returned model object.
When a new point \eqn{(x_{new}, y_{new})} arrives, the update is:
\enumerate{
  \item Compute \eqn{k_{new} = K(x_{new}, x_{inducing})} (m-vector).
  \item Sherman-Morrison: \eqn{P \leftarrow P - \frac{Pk \, k^T P}{1 + k^T P k}}.
  \item Update RHS: \eqn{b \leftarrow b + k_{new} \cdot y_{new}}.
  \item New weights: \eqn{w = P b}.
}

The first call computes \eqn{P} via \code{solve()} at one-time \eqn{O(m^3)}
cost.
Each subsequent update is \eqn{O(m^2)}.

\code{UpdateModel.FullKRR} raises an error because full KRR models do not
support online updates (the weight vector grows with n).
}
\usage{
UpdateModel(model, x_new, y_new, \dots)

\method{UpdateModel}{InducingKRR}(model, x_new, y_new,
            recompute_metrics = TRUE,
            recompute = FALSE,
            stability_threshold = 50L, \dots)

\method{UpdateModel}{FullKRR}(model, x_new, y_new, \dots)
}
\arguments{
\item{model}{A fitted model object of class \code{"InducingKRR"} (returned
  by \code{\link{CreateInducingKRR}}) or \code{"FullKRR"} (which will
  raise an error).}

\item{x_new}{Numeric vector of length \code{d} (or a 1 x d matrix)
  representing the new observation's features.}

\item{y_new}{Single numeric value for the new observation's target.}

\item{recompute_metrics}{Logical (default \code{TRUE}). If \code{FALSE},
  skip recomputation of training metrics after the update. This saves
  \eqn{O(nm)} time and is recommended for fast streaming workloads.}

\item{recompute}{Logical (default \code{FALSE}). If \code{TRUE}, recompute
  \eqn{P} from scratch using \code{solve()} and reset the update counter.
  Use this periodically to maintain numerical stability after many
  rank-1 updates.}

\item{stability_threshold}{Integer (default \code{50L}). A message is
  printed when the number of accumulated rank-1 updates reaches this
  value, suggesting a full recompute for numerical stability.}

\item{\dots}{Additional arguments (currently unused).}
}
\value{
A \emph{new} \code{InducingKRR} model object (the input is not modified) with:
\itemize{
  \item Updated \code{w}, \code{x}, \code{y}, \code{K_nm}, \code{n},
        \code{y_norm}, \code{K_nmTy}, \code{K_nmTy_norm}, \code{ym_scaled}.
  \item \code{P_inv}: the maintained m x m inverse system matrix.
  \item \code{n_updates}: integer count of accumulated rank-1 updates.
  \item \code{metrics}: updated training metrics (or \code{NULL} if
        \code{recompute_metrics = FALSE}).
}
}
\note{
After many rank-1 updates, floating-point error can accumulate in
\code{P_inv}. Use \code{recompute = TRUE} periodically (e.g., every
50--100 updates) to refresh the inverse from scratch.

\code{UpdateModel.FullKRR} always raises an error. Use
\code{\link{CreateInducingKRR}} for models that support online updates.
}
\examples{
set.seed(42)
n <- 80; d <- 3; m <- 10
x <- matrix(rnorm(n * d), n, d)
y <- sin(x[, 1]) + 0.1 * rnorm(n)

fit <- CreateInducingKRR(x, y, inducing_pts = 1:m, kernel = "rbf",
                          sigma = 1.0, lambda = 0.01, optimizer = "pcg",
                          iters = 30)

# --- Keep the original, get a new model ---
x_new <- rnorm(d)
y_new <- sin(x_new[1]) + 0.1 * rnorm(1)
fit2 <- UpdateModel(fit, x_new, y_new)

cat("Original n:", fit$n, " Updated n:", fit2$n, "\n")
# fit is unchanged; fit2 has the new observation

# --- Overwrite to advance a stream ---
fit_stream <- fit
for (i in 1:5) {
  x_new <- rnorm(d)
  y_new <- sin(x_new[1]) + 0.1 * rnorm(1)
  fit_stream <- UpdateModel(fit_stream, x_new, y_new)
}

cat("After 5 updates: n =", fit_stream$n,
    ", n_updates =", fit_stream$n_updates, "\n")
cat("Original still has n =", fit$n, "\n")

# Predict on new data
x_test <- matrix(rnorm(10 * d), 10, d)
preds <- predict(fit_stream, x_test)
}
