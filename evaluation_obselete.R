


# return a ranked list given user id(s)
# cf.rec.list <- function(env, user) {
# 	M <- env$data.summary$n.users
# 	N <- env$data.summary$n.items
# 	D <- env$config$l.dim
# 	model.params <- env$cf$model$params
# 	test.data <- env$fil.data[env$test.samples,]
# 	train.data <- env$fil.data[env$train.samples,]
# 	train.items <- unique(train.data$pid)
# 	# exclude items which are not in the training set
# 	test.data <- test.data[test.data$pid %in% train.items, ]
# 	test.pids <- unique(test.data$pid)
# 	test.p.vec <- model.params$item.mat[,test.pids]
# 	test.p.bias <- model.params$item.bias[test.pids,1]
# 	u.vec <- model.params$user.mat[,user]
# 	u.bias <- model.params$user.bias[user,1]

# 	# 1xN row pred
# 	pred.q <- t(u.vec) %*% test.p.vec + u.bias + test.p.bias + model.params$global.bias
# 	# order by pred.q in descending
# 	rank.list <- test.pids[order(pred.q,decreasing=T)]
# 	return (unique(rank.list))
# }

# recommendation by kpr utility
# kpr.rec.list <- function(env, user) {
# 	M <- env$data.summary$n.users
# 	N <- env$data.summary$n.items
# 	D <- env$config$l.dim
# 	model.params <- env$kpr$model$params
# 	test.data <- env$fil.data[env$test.samples,]
# 	train.data <- env$fil.data[env$train.samples,]
# 	train.items <- unique(train.data$pid)
# 	# exclude items which are not in the training set
# 	test.data <- test.data[test.data$pid %in% train.items, ]
# 	test.pids <- test.data$pid
# 	test.p.vec <- model.params$item.mat[,test.pids]
# 	test.p.bias <- model.params$item.bias[test.pids,1]
# 	u.vec <- model.params$user.mat[,user]
# 	u.bias <- model.params$user.bias[user,1]

# 	# 1xN row vector
# 	a.vec <- t(u.vec) %*% test.p.vec + u.bias + test.p.bias + model.params$global.bias
# 	pred.q <- kpr.pred.q(a.vec, test.data$price)
# 	rank.list <- test.pids[order(pred.q,decreasing=T)]
# 	return (unique(rank.list))
# }

# rank products by decreasing social surplus
# ss.rec.list <- function(env, user) {
# 	M <- env$data.summary$n.users
# 	N <- env$data.summary$n.items
# 	D <- env$config$l.dim
# 	model.params <- env$kpr$model$params
# 	test.data <- env$fil.data[env$test.samples,]
# 	train.data <- env$fil.data[env$train.samples,]
# 	train.items <- unique(train.data$pid)
# 	# exclude items which are not in the training set
# 	test.data <- test.data[test.data$pid %in% train.items, ]
# 	# remove duplicated user-item purchase entries
# 	test.pids <- test.data$pid
# 	test.p.vec <- model.params$item.mat[,test.pids]
# 	test.p.bias <- model.params$item.bias[test.pids,1]
# 	u.vec <- model.params$user.mat[,user]
# 	u.bias <- model.params$user.bias[user,1]

# 	# 1xN row vector
# 	a.vec <- t(u.vec) %*% test.p.vec + u.bias + test.p.bias + model.params$global.bias
# 	pred.q <- kpr.pred.q(a.vec, test.data$price)
# 	# evaluate social surplus
# 	# order by pred.q in descending
# 	cost.percent <- env$cost.percent
# 	# social surplus: a_i,j x log(q+1) - q * c * p
# 	item.ss <- a.vec * log(pred.q + 1) - pred.q * cost.percent * test.data$price
# 	# order by pred.q in descending
# 	rank.list <- test.pids[order(item.ss,decreasing=T)]
# 	return (unique(rank.list))
# }
