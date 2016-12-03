### optimize the model described in https://docs.google.com/document/d/16ztnwqLWQdLlDIxuWuIm_TQq1tvqgQE0vgVHpZ0PHuM/edit#heading=h.nqr54zpu5qnc
### each user and product is represented by a latent vector and the objective function is to minimize the sum of vector norm subject to a bunch of constraints from classical economic theory

library(nloptr)
library(Matrix)

# create environment
if (!exists('shopcom.env')) {
  shopcom.env <- new.env()
  shopcom.env$data.path <- "/run/media/qi/7dfdd270-c104-4863-aca2-6f189fdebb7c/shopcom/structureddata/product_data/orderdata.compact.merge.txt"
  shopcom.env$data.loaded = F
  # configuration
  shopcom.env$config = list("l.dim" = 5, "train.percent" = 0.75, "lr.lambda" = -0.1)
}


filter.data <- function(env) {
    min.u.freq <- 10
    min.i.freq <- 10
    print("filter out long tailed users and products")
    uid <- as.character(env$data$uid)
    pid <- as.character(env$data$pid)
    u.hist <- as.data.frame(table(uid))
    i.hist <- as.data.frame(table(pid))
    names(u.hist) <- c("id","Freq")
    names(i.hist) <- c("id","Freq")

    u.hist$id <- as.character(u.hist$id)
    i.hist$id <- as.character(i.hist$id)

    # adjust the items
    freq.users <- u.hist$id[u.hist$Freq >= min.u.freq]
    freq.items <- i.hist$id[i.hist$Freq >= min.i.freq]
    # randomly sample 20% of the quanlified usres
    set.seed(123456)
    # freq.users.sample <- sample(freq.users, as.integer(0.02 * length(freq.users)))
    freq.users.sample <- freq.users
 
    # filter out non-frequenty users and items and large quantity
    # env$fil.data.all <-  env$data[uid %in% freq.users & env$data$quantity > 0 & env$data$quantity <= 20 & env$data$price > 0 & !is.na(env$data$quantity) & !is.na(env$data$price), ]
    # env$fil.data <- env$data[uid %in% freq.users.sample & env$data$quantity > 0 & env$data$quantity <= 20 & env$data$price > 0 & !is.na(env$data$quantity) & !is.na(env$data$price), ]

    # both user and item must have at least 5 purchase records

    env$fil.data <- env$data[uid %in% freq.users & pid %in% freq.items & env$data$quantity > 0 & env$data$quantity <= 20 & env$data$price > 0 & !is.na(env$data$quantity) & !is.na(env$data$price),]

    # while(T) {
    #   # filter by user
    #   env$fil.data <- env$fil.data[uid %in% freq.users, ]

    #   env$fil.data <- env$fil.data[pid %in% freq.items, ]

    #   u.hist <- as.data.frame(table(env$fil.data$uid))
    #   i.hist <- as.data.frame(table(env$fil.data$pid))
    #   names(u.hist) <- c("id","Freq")
    #   names(i.hist) <- c("id","Freq")

    #   u.hist$id <- as.character(u.hist$id)
    #   i.hist$id <- as.character(i.hist$id)

    #   # adjust the items
    #   freq.users <- u.hist$id[u.hist$Freq >= min.u.freq]
    #   freq.items <- i.hist$id[i.hist$Freq >= min.i.freq]
    #   if (length(freq.users) == length(u.hist$id) & length(freq.items) == length(i.hist$id)) {
    #     break;
    #   }
    # }

    # map uid and pid to consecutive numbers starting from 1
    user.map <- data.frame(name = unique(env$fil.data[["uid"]]))
    user.map$id <- seq(1,length(user.map$name))
    item.map <- data.frame(name = unique(env$fil.data[["pid"]]))
    item.map$id <- seq(1,length(item.map$name))
    env$user.map <- user.map
    env$item.map <- item.map
    env$fil.data$uid <- user.map$id[match(env$fil.data$uid,user.map$name)]
    env$fil.data$pid <- item.map$id[match(env$fil.data$pid,item.map$name)]

    # summary
    env$data.summary <- list("n.users" = max(env$fil.data$uid), "n.items" = max(env$fil.data$pid), "n.rows" = dim(env$fil.data)[1])    
}


# randomly split data into two portions: one for training and one for testing
rand.split <- function(env) {
  # split by user
  # fix the seed so the experiment can be reproduced
  # first order by user id
  env$fil.data <- env$fil.data[order(env$fil.data$uid),]
  set.seed(123456)
  num.rows <- dim(env$fil.data)[1]
  train.flag <- rep(0,num.rows)
  cnt <- 0

  last.user <- env$fil.data$uid[1]
  last.idx <- 1
  for (i in 2:num.rows) {
    cur.user <- env$fil.data$uid[i]
    if (cur.user != last.user) {
      # a new user starts, process previous user
      u.rows <- last.idx : (i - 1)
      smp.cnt <- as.integer(length(u.rows) * env$config$train.percent)
      train.rows <- sample(u.rows, smp.cnt)
      train.flag[train.rows] <- 1
      last.user <- cur.user
      last.idx <- i
    }
    if (i == num.rows) {
      u.rows <- last.idx : i
      smp.cnt <- as.integer(length(u.rows) * env$config$train.percent)
      train.rows <- sample(u.rows, smp.cnt)
      train.flag[train.rows] <- 1      
    }
  }
  # training and test sample indexes
  env$train.samples <- which(train.flag > 0)
  env$test.samples <- which(train.flag == 0)
  # only keep items and users that appeared in training dataset
  train.data <- env$fil.data[env$train.samples,]
  train.uids <- unique(train.data$uid)
  train.pids <- unique(train.data$pid)
  test.data <- env$fil.data[env$test.samples,]
  env$test.samples <- env$test.samples[(test.data$uid %in% train.uids) & (test.data$pid %in% train.pids)]
  # subsample the users
  test.data <- env$fil.data[env$test.samples,]
  rec.users <- unique(test.data$uid)
  set.seed(123456)
  # randomly select 1000 users for testing
  rec.users <- sample(rec.users, size = 1000)
  env$test.samples1000 <- env$test.samples[test.data$uid %in% rec.users]
}


load.data <- function(env) {
  if (!env$data.loaded) {
    print("load shopcom data")
    env$data <- read.csv(file=env$data.path,header=T,sep='\t')
    filter.data(env)
    rand.split(env)
    env$data.loaded = T
    print("save the processed result")
    save.data()
  }
}

save.data <- function(env) {
  save(env,file = "data/shopcom.env.RData")
}


# sgd model by cpp
load.sgd.model <- function(env) {
  if(!exists('kpr.sgd.params',envir=env)){
    user.mat <- read.csv(file="data/rcpp_umat.csv",sep=",",header=F)
    item.mat <- read.csv(file="data/rcpp_imat.csv",sep=",",header=F)
    user.bias <- read.csv(file="data/rcpp_ubias.csv",sep=",",header=F)
    item.bias <- read.csv(file="data/rcpp_ibias.csv",sep=",",header=F)
    global.bias <- read.csv(file="data/rcpp_globalbias.csv",sep=",",header=F)
    global.bias <- global.bias[1,1]

    env$kpr.sgd.params = list(
      user.mat = t(user.mat),
      item.mat = t(item.mat),
      user.bias = user.bias,
      item.bias = item.bias,
      global.bias = global.bias
      )    
  }
}

# export data for cf
export.cf <- function (env) {
  write.table(env$fil.data[env$train.samples,c(1,3,4,5)],file="data/1015/train.csv",quote=F,row.names=F, col.names = F, sep='\t')
  write.table(env$fil.data[env$test.samples,c(1,3,4,5)],file="data/1015/test.csv",quote=F,row.names = F, col.names = F, sep='\t')
  # also generate recommendation file
  # also generate prediction list for 1000 randomly selected users
  test.data <- env$fil.data[env$test.samples,]
  rec.items <- unique(test.data$pid)
  rec.users <- unique(env$fil.data$uid[env$test.samples1000])

  # recommendation testing subjects
  write.table(matrix(rec.users,ncol=1), file="data/1015/rec_users.txt",quote=F,row.names = F, col.names = F, sep='\t')
  # candidate items
  write.table(matrix(rec.items,ncol=1), file="data/1015/rec_items.txt",quote=F,row.names = F, col.names = F, sep='\t')

  num.users<- length(rec.users)
  num.items <- length(rec.items)
  rec.mat <- matrix(0,nrow = num.users * num.items, ncol = 2)

  for (i in 1:num.users) {
    rec.mat[((i-1) * num.items + 1) : (i * num.items),1]  = rec.users[i]
    rec.mat[((i-1) * num.items + 1) : (i * num.items),2]  = rec.items
  }

  write.table(rec.mat,file="data/1015/rec_useritem.csv",quote=F,row.names = F, col.names = F, sep='\t')

}

# export for mymedialite packge
export.mml <- function(env) {
  # export dataset as csv
  # only export user,item,quantity
  # training dataset
  write.table(env$fil.data[env$train.samples,c(1,3,5)],file="data/mml/train.csv",quote=F,row.names=F, col.names = F, sep='\t')
  write.table(env$fil.data[env$test.samples,c(1,3,5)],file="data/mml/test.csv",quote=F,row.names = F, col.names = F, sep='\t')
  # also generate prediction list
  test.data <- env$fil.data[env$test.samples,]
  rec.items <- unique(test.data$pid)
  rec.users <- unique(test.data$uid)
  num.users<- length(rec.users)
  num.items <- length(rec.items)
  rec.mat <- matrix(0,nrow = num.users * num.items, ncol = 2)

  for (i in 1:num.users) {
    rec.mat[((i-1) * num.items + 1) : (i * num.items),1]  = rec.users[i]
    rec.mat[((i-1) * num.items + 1) : (i * num.items),2]  = rec.items
  }

  write.table(rec.mat,file="data/mml/test_no_rating.csv",quote=F,row.names = F, col.names = F, sep='\t')
}