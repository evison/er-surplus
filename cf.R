# collaborative filtering on quantity
source('logistic.R')

cf.lambda <- 0.05
cf.dim <- 10

cf.eval_f <- function(x, M, N, D, uid, pid, q, p) {
  x.mat <- matrix(x[1 : (M * D)],nrow = D, ncol = M)
  y.mat <- matrix(x[(M * D + 1) : ((M + N) *D)], nrow = D, ncol = N)
  u.bias <- matrix(x[((M + N) * D + 1) : ((M + N) * D + M)], nrow = M, ncol=1)
  i.bias <- matrix(x[((M + N) * D + M + 1) : ((M + N) * D + M + N)], nrow = N, ncol = 1)
  n.vars <- length(x)
  global.bias <- x[n.vars]

  # compute the marginal consumer surplus at q and q+1
  pred.q <- colSums(x.mat[,uid] * y.mat[,pid]) + u.bias[uid,1] + i.bias[pid,1] + global.bias
  pred.err <- pred.q - q
  f.val <- 0.5 * sum(pred.err^2) + 0.5 * cf.lambda * sum(x^2)
  f.grad <- rep(0,n.vars)
  n.rows <- length(uid)

  for(i in 1:n.rows) {
  	tmp.err <- pred.err[i]
    tmp.uid <- uid[i]
    tmp.pid <- pid[i]
    # index of x_i in x
    xi <- x.mat[,tmp.uid]
    yj <- y.mat[,tmp.pid]
    xi.idx <- ((tmp.uid - 1) * D + 1) : (tmp.uid * D)
    yj.idx <- ((tmp.pid + M - 1) * D + 1) : ((tmp.pid + M) * D)
    f.grad[xi.idx] <- f.grad[xi.idx] + tmp.err * yj + cf.lambda * xi
    f.grad[yj.idx] <- f.grad[yj.idx] + tmp.err * xi + cf.lambda * yj

    u.bias.idx <- (M + N) * D + tmp.uid
    i.bias.idx <- (M + N) * D + M + tmp.pid
    f.grad[c(u.bias.idx,i.bias.idx,n.vars)] <- f.grad[c(u.bias.idx,i.bias.idx,n.vars)] + tmp.err + cf.lambda * x[c(u.bias.idx,i.bias.idx,n.vars)]
  }

  return (list("objective" = f.val, "gradient" = f.grad))
}



# cf.lambda <- 0.01

# cf.eval_f <- function(x, M, N, D, uid, pid, q, p) {
#   x.mat <- matrix(x[1 : (M * D)],nrow = D, ncol = M)
#   y.mat <- matrix(x[(M * D + 1) : ((M + N) *D)], nrow = D, ncol = N)
#   u.bias <- matrix(x[((M + N) * D + 1) : ((M + N) * D + M)], nrow = M, ncol=1)
#   i.bias <- matrix(x[((M + N) * D + M + 1) : ((M + N) * D + M + N)], nrow = N, ncol = 1)
#   n.vars <- length(x)
#   global.bias <- x[n.vars]

#   # compute the marginal consumer surplus at q and q+1
#   pred.q <- global.bias
#   pred.err <- pred.q - q
#   f.val <- 0.5 * sum(pred.err^2) + 0.5 * cf.lambda * sum(x^2)
#   f.grad <- rep(0,n.vars)
#   f.grad[n.vars] <- sum(pred.err)
#   return (list("objective" = f.val, "gradient" = f.grad))
# }

# extract model parameters
cf.extract.param <- function(env){
  x <- env$cf$model$solution
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- cf.dim
  x.mat <- matrix(x[1 : (M * D)],nrow = D, ncol = M)
  y.mat <- matrix(x[(M * D + 1) : ((M + N) *D)], nrow = D, ncol = N)
  u.bias <- matrix(x[((M + N) * D + 1) : ((M + N) * D + M)], nrow = M, ncol=1)
  i.bias <- matrix(x[((M + N) * D + M + 1) : ((M + N) * D + M + N)], nrow = N, ncol = 1)
  n.vars <- length(x)
  global.bias <- x[n.vars]
  return (list(user.mat = x.mat, item.mat = y.mat, user.bias = u.bias, item.bias = i.bias, global.bias = global.bias))
}


cf.train <- function(env) {
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- cf.dim
  train.data <- env$fil.data[env$train.samples,]

  env$lr$options <- list("algorithm" = "NLOPT_LD_MMA","xtol_rel" = 1.0e-4,  "print_level" = 1
    # ,"local_opts" = list("algorithm" = "NLOPT_LD_MMA","xtol_rel" = 1e-8)
    )

  # model training on the training dataset split
  # a_ij = x_i^Ty_j + b_u + b_i + b0
  n.vars <- (M + N) * D + M + N + 1
  x0 <- runif(n.vars, 0,1)
  lb <- rep(-Inf,n.vars)
  ub <- rep(Inf,n.vars)
  env$cf$model <- nloptr(x0=x0, eval_f = cf.eval_f, opts = env$lr$options,lb = lb,ub = ub, M = M, N = N, D = D,uid = train.data$uid, pid = train.data$pid, p = train.data$price, q = train.data$quantity)
  # extracting parameters
  env$cf$model$params <- cf.extract.param(env)
}


cf.eval <- function(env, test.set.idx){
 model.params <- env$cf$model$params
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- cf.dim
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

  # evaluate the utility of the first unit
  pred.q <- colSums(model.params$user.mat[,uids] * model.params$item.mat[,pids]) + model.params$user.bias[uids] + model.params$item.bias[pids] + model.params$global.bias
  pred.err <- (pred.q - q)^2
  rmse <- sqrt(mean(pred.err))
  result <- list("true.q" = q, "pred.q" = pred.q, "pred.err" = pred.err, "rmse" = rmse, "test.data" = test.data)
  return (result)
}
