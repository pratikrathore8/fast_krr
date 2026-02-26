\name{predict.FullKRR}
\alias{predict.FullKRR}
\alias{predict.InducingKRR}
\title{Predict from a Kernel Ridge Regression Model}
\description{
Compute predictions for new data from a fitted KRR model.

For \code{FullKRR} objects the prediction is
\code{K(newdata, x_train) \%*\% w}, where \code{x_train} is the original
training data and \code{w} is the fitted weight vector.

For \code{InducingKRR} objects the prediction is
\code{K(newdata, x_inducing) \%*\% w}, where \code{x_inducing} are the
inducing points used during fitting.
}
\usage{
\method{predict}{FullKRR}(object, newdata, \dots)

\method{predict}{InducingKRR}(object, newdata, \dots)
}
\arguments{
\item{object}{A fitted model object of class \code{"FullKRR"} (returned by
  \code{\link{CreateFullKRR}}) or \code{"InducingKRR"} (returned by
  \code{\link{CreateInducingKRR}}).}

\item{newdata}{A numeric matrix of new observations (\code{n_new x d}) with
  the same number of columns as the original training data.}

\item{\dots}{Additional arguments (currently unused; included for
  compatibility with the generic \code{\link[stats]{predict}} method).}
}
\value{
A numeric vector of length \code{n_new} containing predictions. For
regression tasks these are continuous predicted values; for classification
tasks they are predicted class scores.
}
\examples{
## Full KRR prediction
set.seed(42)
n <- 100; d <- 3
x <- matrix(rnorm(n * d), n, d)
y <- sin(x[, 1]) + 0.1 * rnorm(n)

model <- CreateFullKRR(x, y, kernel = "rbf", sigma = 1.0, lambda = 1e-3)

x_new <- matrix(rnorm(10 * d), 10, d)
preds <- predict(model, x_new)

## Inducing KRR prediction
inducing <- sample(n, 20)
model_ind <- CreateInducingKRR(x, y, inducing_pts = inducing,
                                kernel = "rbf", sigma = 1.0, lambda = 1e-3)
preds_ind <- predict(model_ind, x_new)
}
