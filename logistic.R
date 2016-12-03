### optimize the model described in https://docs.google.com/document/d/16ztnwqLWQdLlDIxuWuIm_TQq1tvqgQE0vgVHpZ0PHuM/edit#heading=h.nqr54zpu5qnc
### each user and product is represented by a latent vector and the objective function is to minimize the sum of vector norm subject to a bunch of constraints from classical economic theory
# formulate the problem as logistic regression

library(nloptr)
library(Matrix)

# regularization term
lr.lambda <- 1e5

#' compute marginal consumer surplus  according to KPR utility model
#'
#' marginal consumer surplus is defined as substracting price from marginal utility at given quantity
#' KPR utility model: %U(q) = a\log(1+q)
#' where a is the unknown parameter and q is quantity. The marginal consumer surplus is given by,
#' %\Delta CS_{i,j}(q,p) = a_{i,j}\left(\log(q+1) - \log(q)\right) - p
#' 
# 
# @param  q   quantity
# @param  p   price
# @return resulted marginal consumer surplus
kpr.delta.cs <- function(a, q, p) {
  return (a * (log(q+1) - log(q)) - p)
}

#' map marginal consumer surplus to probability through sigmoid function
#' 
#' @param   delta.col   marginal consumer surplus by such as kpr.delta.param
#' @cs   minus   whether subtracting the resulted probability from 1
#' @return  probability
lr.prob <- function(delta.cs) {
  prob <- 1 / (1 + exp(-delta.cs))
  return (prob)
}

log.lr.prob <- function(x) {
  res <- x
  small.idx <- x < -10
  res[small.idx] <- x[small.idx]
  res[x >= -10] <- log(1/(1+exp(-x[x >= -10])))
  return (res)
}


#' compute the objective function value and gradient
#'
#' the objective function vlaue include the likelihood and regularization term
#' @param   env environemnt object
#' @param   M   number of users
#' @param   N   number of items
#' @param   D   latent vector dimension
#' @param   uid   user ids
#' @param   pid   item ids
#' @param   q   purchase quantity
#' @param   p   purchase price
#' @return  list('constraints' = , 'gradient' = )
lr.eval_f <- function (x, M, N, D, uid, pid, q, p) {
  x.mat <- matrix(x[1 : (M * D)],nrow = D, ncol = M)
  y.mat <- matrix(x[(M * D + 1) : ((M + N) *D)], nrow = D, ncol = N)
  u.bias <- matrix(x[((M + N) * D + 1) : ((M + N) * D + M)], nrow = M, ncol=1)
  i.bias <- matrix(x[((M + N) * D + M + 1) : ((M + N) * D + M + N)], nrow = N, ncol = 1)
  n.vars <- length(x)
  global.bias <- x[n.vars]

  # compute the marginal consumer surplus at q and q+1
  a.vec <- colSums(x.mat[,uid] * y.mat[,pid]) + u.bias[uid,1] + i.bias[pid,1] + global.bias
  delta.cs.q <- kpr.delta.cs(a.vec, q, p )
  # quantity: q+1
  delta.cs.qg1 <- kpr.delta.cs(a.vec, q + 1, p)
  prob.q <- lr.prob(delta.cs.q)
  prob.qg1 <- lr.prob(delta.cs.qg1)

  # sum of log likelihood
  ll.sum <- sum(log.lr.prob(c(delta.cs.q,-delta.cs.qg1)))

  # add regularization
  f.val <- ll.sum - 0.5 * lr.lambda * sum(x^2)

  # calculate gradient regarding x
  # log(1/(1+e^(-x)))' = 1/(1+e^x)
  # initialize gradient vector  as zero vector
  f.grad <- rep(0,length(x))
  n.rows <- length(uid)
  # iterate through user-item purchase tuples  
  for (i in 1:n.rows) {
    tmp.uid <- uid[i]
    tmp.pid <- pid[i]
    tmp.p <- p[i]
    tmp.q <- q[i]
    tmp.a <- a.vec[i]
    # index of x_i in x
    xi <- x.mat[,tmp.uid]
    yj <- y.mat[,tmp.pid]
    xi.idx <- ((tmp.uid - 1) * D + 1) : (tmp.uid * D)
    # index of y_j in x
    yj.idx <- ((tmp.pid + M - 1) * D + 1) : ((tmp.pid + M) * D)

    # gradient of xi
    comm.term <- (1 - prob.q[i]) * (log(tmp.q+1) - log(tmp.q)) + prob.qg1[i] * (log(tmp.q + 1) - log(tmp.q + 2))
    f.grad[xi.idx] <- f.grad[xi.idx] + comm.term * yj - lr.lambda * xi

    # gradient of yj
    f.grad[yj.idx] <- f.grad[yj.idx] + comm.term * xi - lr.lambda * yj

    u.bias.idx <- (M + N) * D + tmp.uid
    i.bias.idx <- (M + N) * D + M + tmp.pid

    # gradient are 1 for the bias terms
    # fixed a bug
    f.grad[c(u.bias.idx,i.bias.idx,n.vars)] <- f.grad[c(u.bias.idx,i.bias.idx,n.vars)] + comm.term - lr.lambda * c(x[u.bias.idx],x[i.bias.idx],x[n.vars])
  }
  return (list("objective" = -f.val, "gradient" = -f.grad))

}


lr.train <- function(env) {
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- env$config$l.dim
  train.data <- env$fil.data[env$train.samples,]

  env$lr$options <- list("algorithm" = "NLOPT_LD_MMA","xtol_rel" = 1.0e-8,  "print_level" = 1
    # ,"local_opts" = list("algorithm" = "NLOPT_LD_MMA","xtol_rel" = 1e-8)
    )

  # model training on the training dataset split
  # a_ij = x_i^Ty_j + b_u + b_i + b0
  n.vars <- (M + N) * D + M + N + 1
  x0 <- runif(n.vars, 0,1)
  lb <- rep(-Inf,n.vars)
  ub <- rep(Inf,n.vars)
  env$lr$model <- nloptr(x0=x0, eval_f = lr.eval_f, opts = env$lr$options,lb = lb,ub = ub, M = M, N = N, D = D,uid = train.data$uid, pid = train.data$pid, p = train.data$price, q = train.data$quantity)
  # extracting parameters
  env$kpr$model$params <- lr.extract.param(env)
}


# predict quantity given aij
kpr.pred.q <- function(a,price) {
  tmp <- exp(price/a) - 1
  return (as.integer(1/tmp))
}

# extract model parameters
lr.extract.param <- function(env){
  x <- env$lr$model$solution
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- env$config$l.dim
  x.mat <- matrix(x[1 : (M * D)],nrow = D, ncol = M)
  y.mat <- matrix(x[(M * D + 1) : ((M + N) *D)], nrow = D, ncol = N)
  u.bias <- matrix(x[((M + N) * D + 1) : ((M + N) * D + M)], nrow = M, ncol=1)
  i.bias <- matrix(x[((M + N) * D + M + 1) : ((M + N) * D + M + N)], nrow = N, ncol = 1)
  n.vars <- length(x)
  global.bias <- x[n.vars]
  return (list(user.mat = x.mat, item.mat = y.mat, user.bias = u.bias, item.bias = i.bias, global.bias = global.bias))
}


# model evaluation on the testing split
lr.eval <- function(env, test.set.idx) {
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- env$config$l.dim
  test.data <- env$fil.data[test.set.idx,]
  train.data <- env$fil.data[env$train.samples,]
  train.items <- unique(train.data$pid)
  # exclude items which are not in the training set
  test.data <- test.data[test.data$pid %in% train.items, ]

  uids <- test.data$uid
  pids <- test.data$pid

  q <- test.data$quantity
  p <- test.data$price

  n.rows <- dim(test.data)[1]
  err <- 0
  model.params <- lr.extract.param(env)

  # evaluate the utility of the first unit
  a.vec <- colSums(model.params$user.mat[,uids] * model.params$item.mat[,pids]) + model.params$user.bias[uids] + model.params$item.bias[pids] + model.params$global.bias
  pred.q <- kpr.pred.q(a.vec, p)
  pred.err <- (pred.q - q)^2
  rmse <- sqrt(mean(pred.err))
  result <- list("true.q" = q, "pred.q" = pred.q, "pred.err" = pred.err, "rmse" = rmse, "test.data" = test.data)
  return (result)
}

# baseline algorithm
mean.eval <- function(env, test.set.idx) {
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- env$config$l.dim
  test.data <- env$fil.data[test.set.idx,]
  train.data <- env$fil.data[env$train.samples,]
  train.items <- unique(train.data$pid)
  # exclude items which are not in the training set
  test.data <- test.data[test.data$pid %in% train.items, ]
  q <- test.data$quantity
  pred.q <- mean(train.data$quantity)
  pred.err <- (pred.q - q)^2
  rmse <- sqrt(mean(pred.err))
  result <- list("true.q" = q, "pred.q" = pred.q, "pred.err" = pred.err, "rmse" = rmse)
  return (result)
}

# lr.train(shopcom.env)
# train.eval.result <- lr.eval(shopcom.env,shopcom.env$train.samples)
# test.eval.result <- lr.eval(shopcom.env,shopcom.env$test.samples)

# train.eval.result$rmse
# test.eval.result$rmse