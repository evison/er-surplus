# sgd trainer
source("logistic.R")

lr.sgd.train <- function(env) {
  M <- env$data.summary$n.users
  N <- env$data.summary$n.items
  D <- env$config$l.dim
  train.data <- env$fil.data[env$train.samples,]
  n.samples <- dim(train.data)[1]

  # model parameters
  user.mat <- matrix(data=runif(M * D), nrow=D, ncol = M)
  item.mat <- matrix(data=runif(N * D), nrow = D, ncol = N)
  user.bias <- matrix(data = 0, nrow = M, ncol = 1)
  item.bias <- matrix(data = 0, nrow = N, ncol = 1)
  global.bias <- 0 

  max.epochs <- 50

  uids <- train.data$uid
  pids <- train.data$pid
  prices <- train.data$price
  quants <- train.data$quantity

  t <- 0
  reg.coef <- lr.lambda

  min.batch <- 10
  u.vec.grad <- matrix(0,nrow=D,ncol=M)
  i.vec.grad <- matrix(0,nrow=D,ncol=N)
  u.bias.grad <- matrix(0,nrow=M,ncol=1)
  i.bias.grad <- matrix(0,nrow=N,ncol=1)
  global.bias.grad <- 0

  for(epoch in 1:max.epochs) {
    # shuffle the samples
    # function value
    f.val <- 0
    for(i in sample(1:n.samples,size=n.samples)) {
      t <- t + 1
      uid <- uids[i]
      pid <- pids[i]
      p <- prices[i]
      q <- quants[i]

      u.vec <- user.mat[,uid]
      i.vec <- item.mat[,pid]
      u.bias <- user.bias[uid,1]
      i.bias <- item.bias[pid,1]
      aij <- sum(u.vec * i.vec) + u.bias + i.bias + global.bias

      # quantity: q+1
      delta.cs.q <- kpr.delta.cs(aij, q, p )
      delta.cs.qg1 <- kpr.delta.cs(aij, q + 1, p)

      prob.q <- lr.prob(delta.cs.q)
      prob.qg1 <- lr.prob(delta.cs.qg1)

      comm.term <- (1 - prob.q) * (log(q+1) - log(q)) - prob.qg1 * (log(q + 2) - log(q + 1))

      u.vec.grad[,uid] <- u.vec.grad[,uid] - (comm.term * i.vec - reg.coef * u.vec)
      i.vec.grad[,pid] <- i.vec.grad[,pid] - (comm.term * u.vec - reg.coef * i.vec)
      u.bias.grad[uid,1] <- u.bias.grad[uid,] - (comm.term - reg.coef * u.bias)
      i.bias.grad[pid,1] <- i.bias.grad[pid,] - (comm.term - reg.coef * i.bias)
      global.bias.grad <- global.bias.grad - (comm.term - reg.coef * global.bias)

      # update learning rate
      lrate <- 1/(1 + 1e-5 * t)
      # lrate <- 0.5/epoch
      # batch update
      if (t %% min.batch == 0) {
        # perform gradient decent
        user.mat <- user.mat - lrate * (u.vec.grad)
        item.mat <- item.mat - lrate * (i.vec.grad)
        user.bias <- user.bias - lrate * (u.bias.grad)
        item.bias <- item.bias - lrate * (i.bias.grad)
        global.bias <- global.bias - lrate * (global.bias.grad)
        # reset mini-batch gradient
        u.vec.grad[,] <- 0
        i.vec.grad[,] <- 0
        u.bias.grad[,] <- 0
        i.bias.grad[,] <- 0
        global.bias.grad <- 0
      }
    }

    a.vec <- colSums(user.mat[,uids] * item.mat[,pids]) + user.bias[uids,] + item.bias[pids,] + global.bias
    delta.cs.q <- kpr.delta.cs(a.vec,quants,prices)
    delta.cs.qg1 <- kpr.delta.cs(aij, quants + 1, p)

    f.val <- - sum(log.lr.prob(delta.cs.q) + log.lr.prob(-delta.cs.qg1)) + 0.5 * reg.coef * (sum(user.mat^2) + sum(item.mat^2) + sum(user.bias^2) + sum(item.bias^2) + global.bias^2)

    # report RMSE on quantity
    param <- list(user.mat = user.mat, item.mat = item.mat, user.bias = user.bias, item.bias = item.bias, global.bias = global.bias)
    eval.train.res <- lr.sgd.eval(env,env$train.samples, param)
    eval.test.res <- lr.sgd.eval(env,env$test.samples, param)

    print(sprintf("sgd - epoch: %d, function value: %.4f, learning rate: %.4f, train RMSE: %.4f, test RMSE: %.4f", epoch, f.val,lrate, eval.train.res$rmse,eval.test.res$rmse))
    env$lr$sgd.model <- param
  }
}


# model evaluation on the testing split
lr.sgd.eval <- function(env, test.set.idx, model.params) {
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

  # evaluate the utility of the first unit
  a.vec <- colSums(model.params$user.mat[,uids] * model.params$item.mat[,pids]) + model.params$user.bias[uids] + model.params$item.bias[pids] + model.params$global.bias
  pred.q <- kpr.pred.q(a.vec, p)
  pred.err <- (pred.q - q)^2
  rmse <- sqrt(mean(pred.err))
  result <- list("true.q" = q, "pred.q" = pred.q, "pred.err" = pred.err, "rmse" = rmse, "test.data" = test.data)
  return (result)
}



# lr.sgd.train(shopcom.env)