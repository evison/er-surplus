### optimize the model described in https://docs.google.com/document/d/16ztnwqLWQdLlDIxuWuIm_TQq1tvqgQE0vgVHpZ0PHuM/edit#heading=h.nqr54zpu5qnc
### each user and product is represented by a latent vector and the objective function is to minimize the sum of vector norm subject to a bunch of constraints from classical economic theory

library(nloptr)
library(Matrix)

# create environment
if (!exists('my.env')) {
  my.env <- new.env('shop.com')
  my.env$data.path <- "/run/media/qi/7dfdd270-c104-4863-aca2-6f189fdebb7c/shopcom/structureddata/product_data/orderdata.brief.txt"
  my.env$data.loaded = F
}

filter.data <- function(my.env) {
    print("filter out long tail users and items and large quantity purchase")
    uid <- as.character(my.env$data$uid)
    pid <- as.character(my.env$data$pid)
    u.hist <- as.data.frame(table(uid))
    i.hist <- as.data.frame(table(pid))
    names(u.hist) <- c("id","Freq")
    names(i.hist) <- c("id","Freq")

    u.hist$id <- as.character(u.hist$id)
    i.hist$id <- as.character(i.hist$id)

    freq.items <- i.hist$id[i.hist$Freq >= 5]
    freq.users <- i.hist$id[u.hist$Freq >= 5]
    # filter out non-frequenty users and items and large quantity
    my.env$fil.data <- my.env$data[pid %in% freq.items & uid %in% freq.users & my.env$data$quantity <= 20,]
}

load.data <- function(my.env) {
  if (!my.env$data.loaded) {
    print("load shopcom data")
    my.env$data <- read.csv(file=my.env$data.path,header=T,sep='\t')
    filter.data(my.env)
    my.env$data.loaded = T
  }
}


eval_f <- function (x) {
  f.val <- 0.5 * sum(x^2)
  f.grad <- x
  # print(sprintf("function value:%.4f",f.val))
  return (list('objective' = f.val,'gradient' = f.grad))
}

# inequality constraints
eval_g_ineq <- function(x) {
  result <- list ()
  # reshape x 
  M <- my.env$optim$n.u
  N <- my.env$optim$n.i
  D <- my.env$optim$dim
  x.mat <- matrix(x[1 : (M * D)],nrow = D, ncol = M)
  y.mat <- matrix(x[(M * D + 1) : ((M + N) *D)], nrow = D, ncol = N)
  
  # extract row information
  uid <- my.env$fil.data[["uid"]]
  pid <- my.env$fil.data[["pid"]]
  n.rows <- dim(my.env$fil.data)[1]
  q <- my.env$fil.data[["quantity"]]
  p <- my.env$fil.data[["price"]]
  a.vec <- colSums(x.mat[,uid] * y.mat[,pid])
  
  exp.aq <- exp(a.vec * q)
  exp.aql1 <- exp(a.vec * (q - 1))
  exp.aqg1 <- exp(a.vec * (q + 1))
  
  # two constraints from the google doc
  result$constraints = c(
    a.vec * p + exp.aq * (1 - exp(a.vec)),
    exp.aq * (1 - exp(-a.vec)) - a.vec * p
  )
  
  for (i in 1:n.rows) {
    tmp.uid <- uid[i]
    tmp.pid <- pid[i]
    tmp.x <- x.mat[,tmp.uid]
    tmp.y <- y.mat[,tmp.pid]
    x.idx <- ((tmp.uid - 1) * D + 1) : (tmp.uid * D)
    y.idx <- ((M + tmp.pid - 1) * D + 1) : ((tmp.pid + M) * D )
    
    # for the first constraint
    tmp.range <- ((i-1) * 2 * D + 1) : (2 * i *  D)
    # first component of i-th constraint
    my.env$ijv[tmp.range,1] <- i
    my.env$ijv[tmp.range,2] <- as.integer(c(x.idx,y.idx))
    comm.term <- (p[i] + q[i]*exp.aq[i] - (q[i]+1)*exp.aqg1[i])
    tmp.vals <- c(tmp.y,tmp.x ) * comm.term

    my.env$ijv[tmp.range,3] <- tmp.vals
    if(any(is.na(tmp.vals))) {
      print("na values 1")
      browser()
    }
    # second component of i-th constraint, the offset is: 2 * D * n.rows
    tmp.range <- ((i-1) * 2 * D + 1 + 2 * D * n.rows) : (2 * i *  D + 2 * D * n.rows)
    my.env$ijv[tmp.range,1] <- i + n.rows
    my.env$ijv[tmp.range,2] <- as.integer(c(x.idx,y.idx))
    comm.term <- q[i] * exp.aq[i] - (q[i] - 1) * exp.aql1[i] - p[i]
    tmp.vals <- c(tmp.y,tmp.x) * comm.term
    my.env$ijv[tmp.range,3] <- tmp.vals

    if(any(is.na(tmp.vals))) {
      print("na values 2")
      browser()
    }

  }
  result$jacobian <- as.matrix(sparseMatrix(i=my.env$ijv[,1],j=my.env$ijv[,2],x=my.env$ijv[,3]))
  
  return (result)
}

gd.solver <- function(my.env) {
  # map uid and pid to consecutive numbers starting from 1
  user.map <- data.frame(name = unique(my.env$fil.data[["uid"]]))
  user.map$id <- seq(1,length(user.map$name))
  item.map <- data.frame(name = unique(my.env$fil.data[["pid"]]))
  item.map$id <- seq(1,length(item.map$name))
  my.env$fil.data$uid <- user.map$id[match(my.env$fil.data$uid,user.map$name)]
  my.env$fil.data$pid <- item.map$id[match(my.env$fil.data$pid,item.map$name)]
  M <- max(my.env$fil.data[["uid"]])
  N <- max(my.env$fil.data[["pid"]])
  D <- 5
  my.env$optim$dim <- D
  my.env$optim <- list("dim" = D, "n.u" = M, "n.i" = N,"n.rows" = dim(my.env$fil.data)[1])
  # triples for sparse matrix specification
  my.env$ijv <- as.data.frame(matrix(0,nrow = 4 * my.env$optim$n.rows * D, ncol = 3))

  my.env$optim$options <- list("algorithm" = "NLOPT_LD_AUGLAG","xtol_rel" = 1.0e-4, "print_level" = 1,"local_opts" = list("algorithm" = "NLOPT_LD_LBFGS","xtol_rel" = 1e-4))
  my.env$optim$result <- nloptr(x0=runif((M+N)*D,0.001,1), eval_f = eval_f, eval_g_ineq = eval_g_ineq, opts = my.env$optim$options,lb = rep(0,(M+N)*D),ub = rep(Inf,(M+N)*D))
}


# predict quantity given aij
pred.q <- function(a,price) {
  prev.util <- 0
  for (q in 1:20) {
    cur.util <- (1 - exp(-a * q)) / a
    if (cur.util - prev.util < price) {
      return (q - 1)
    }
    prev.util <- cur.util
  }
}

# solution evaluation
sol.eval <- function(env) {
  M <- env$optim$n.u
  N <- env$optim$n.i
  D <- env$optim$dim
  solution <- env$optim$result$solution
  uids <- env$fil.data$uid
  pids <- env$fil.data$pid
  prices <- env$fil.data$price
  q <- env$fil.data$quantity
  err <- 0
  for (k in 1:env$optim$n.rows){
    i <- uids[k]
    j <- pids[k]
    xi <- solution[((i-1) * D + 1) : (i * D)]
    yj <- solution[((j-1) * D + 1 + M *D): (j * D + M * D)]
    a <- sum(xi*yj)
    q1 <- pred.q(a, prices[k])
    print(sprintf("aij:%.4f, predicted q: %d",a, q1))
    err <- err + (q1 - q[k])^2
  }
  err <- sqrt(err/env$optim$n.rows)
  print(sprintf("fitting error:%.4f",err))
}


#load.data(my.env)
#gd.solver(my.env)
