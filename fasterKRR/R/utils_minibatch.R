# Minibatch generator for stochastic optimizers.
#
# A closure that cycles through random permutations of indices,
# yielding batches of a specified size. Re-shuffles when consumed.

#' Create a minibatch generator closure.
#'
#' @param n Total number of data points.
#' @param bg Batch (minibatch) size.
#' @return A function that, when called, returns the next batch of indices.
CreateMinibatchGenerator <- function(n, bg) {
  idx <- sample.int(n)
  n_batches <- ceiling(n / bg)
  current_batch <- 0L

  function() {
    if (current_batch >= n_batches) {
      idx <<- sample.int(n)
      current_batch <<- 0L
    }
    start <- current_batch * bg + 1L
    end <- min((current_batch + 1L) * bg, n)
    current_batch <<- current_batch + 1L
    return(idx[start:end])
  }
}
