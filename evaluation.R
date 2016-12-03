# cf recommendation results by mml library
# 1000 testing user and each of them has top 100 recommendations
mml.rec.load <- function (env) {
	if(!exists('mml.top100',envir=env)) {
		rec.mat <- read.csv(file='data/mml/test_nr_prediction_top100.csv',sep="\t",header=F)
		names(rec.mat) <- c("uid","pid","quantity")
		env$mml.top100 <- list()
		env$mml.top100.users <- unique(rec.mat[,1])
		for(i in 1:as.integer(dim(rec.mat)[1]/100)) {
			row.idx <- (i-1) * 100 + 1
			uid <- rec.mat[row.idx,1]
			env$mml.top100[[uid]] <- rec.mat[row.idx : (row.idx + 99),2]
		}
	}	
}

mml.rec.list <- function(env, user) {
	mml.rec.load(env)
	return (env$mml.top100[[user]])
}

# cf on quantity by CFSGD
# 1000 testing user and each of them has top 100 recommendations
quantity.cf.rec.load <- function (env) {
	if(!exists('quantity.cf.top100',envir=env)) {
		rec.mat <- read.csv(file='data/rcpp_test_nr_prediction_top100.csv',sep="\t",header=F)
		names(rec.mat) <- c("uid","pid","quantity")
		env$quantity.cf.top100 <- list()
		env$quantity.cf.top100.users <- unique(rec.mat[,1])
		for(i in 1:as.integer(dim(rec.mat)[1]/100)) {
			row.idx <- (i-1) * 100 + 1
			uid <- rec.mat[row.idx,1]
			env$quantity.cf.top100[[uid]] <- rec.mat[row.idx : (row.idx + 99),2]
		}
	}	
}

quantity.cf.rec.list <- function(env, user) {
	quantity.cf.rec.load(env)
	return (env$quantity.cf.top100[[user]])
}

# cf on quantity by SSMax
# 1000 testing user and each of them has top 100 recommendations
# @param 	env 	shop.com environment
# @param 	model.name 	SSMax model which is used to identify the recommendation result
ssmax.rec.load <- function (env, model.name) {
	reclist.name <- sprintf("%s_top100",model.name)
	rec.user.name <- sprintf("%s_top100.users", model.name)
	file.path <- sprintf("data/1016/%s_rec_useritem_ranked_top100.csv", model.name)

	if(!exists(reclist.name,envir=env)) {
		rec.mat <- read.csv(file = file.path, sep = "\t", header = F)
		names(rec.mat) <- c("uid","pid","quantity")
		env[[reclist.name]] <- list()
		env[[rec.user.name]] <- unique(rec.mat[,1])

		for(i in 1:as.integer(dim(rec.mat)[1]/100)) {
			row.idx <- (i-1) * 100 + 1
			uid <- rec.mat[row.idx,1]
			env[[reclist.name]][[uid]] <- rec.mat[row.idx : (row.idx + 99),2]
		}
	}	
}

ssmax.rec.list <- function(env, user, params) {
	ssmax.rec.load(env, params$model.name)
	reclist.name <- sprintf("%s_top100",params$model.name)
	return (env[[reclist.name]][[user]])
}

# # rank products by decreasing social surplus
# # the model parameters is loaded through load.sgd.model() defined in init.R file
# ss.sgd.rec.list <- function(env, user) {
# 	load.sgd.model(env)
# 	model.params <- env$kpr.sgd.params
# 	test.data <- env$fil.data[env$test.samples,]
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
# 	pred.q[pred.q < 0] <- 0
# 	item.ss <- a.vec * log(pred.q + 1) - pred.q * cost.percent * test.data$price
# 	# order by pred.q in descending
# 	rank.list <- unique(test.pids[order(item.ss,decreasing=T)])

# 	# rank.list <- unique(test.pids[order(a.vec,decreasing=T)])	

# 	return (rank.list[1:100])
# }


## the entry function for all algorithms to evaluate
# @param 	env 	shopcom.env variable
# @param 	REC.FUNC 	product recommendation algorithm, one of ss.sgd.rec.list, mml.rec.list
# @return 	precision-recall curves
topk.eval <- function(env,REC.FUNC, PARAMS = list()) {
	# entire testing dataset
	test.data <- env$fil.data[env$test.samples,]
	# randomly select 1000 for testing
	test.users1000 <- unique(env$fil.data$uid[env$test.samples1000])
	n.test.users <- length(test.users1000)

	# up to 100 recommendations
	n.test.pids <- 100
	user.recall <- matrix(data=0, nrow=n.test.users, ncol=n.test.pids)
	user.precision <- user.recall
	user.conversion.rate <- user.recall
	user.auc <- c(0,n.test.users)
	user.f1 <- user.recall

	for(i in 1:n.test.users) {
		# get user id
		user <- test.users1000[i]
		# get items actually purchased by user
		user.items <- unique(test.data$pid[test.data$uid == user])
		# recommendation list for current user
		u.rec.list <- REC.FUNC(env,user, PARAMS)
		# generate pr curve given purchased items and recommendation list
		res <- pr.curve(ref.set = user.items, rec.list = u.rec.list)
		# keep the pr curves
		user.recall[i,] <- res$recall
		user.precision[i,] <- res$precision
		user.conversion.rate[i,] <- res$conversion.rate
		user.auc[i] <- res$auc
	}

	# now do the average
	avg.recall <- colMeans(user.recall)
	avg.precision <- colMeans(user.precision)
	avg.conversion.rate <- colMeans(user.conversion.rate)
	avg.f1 <- 2 * avg.recall * avg.precision / (avg.recall + avg.precision)
	avg.auc <- mean(user.auc)
	# also calculate AUC

	return (list(recall = avg.recall, precision = avg.precision, f1 = avg.f1, auc = avg.auc, conversion.rate = avg.conversion.rate ))
}



# generate precision-recall curve given reference product set and recommended list
# change precision to conversion rate
pr.curve <- function(ref.set, rec.list) {
	# indicate whether each of the recommendation is a hit
	# a hit means there is a match of the recommendation in ref.set
	hit.res <- rec.list %in% ref.set
	# size of recall
	n.ref <- length(ref.set)
	# pr curve
	recall.curve <- rep(0,length(hit.res))
	precision.curve <- rep(0,length(hit.res))
	# conversion rate defined as the percentage of hit event at k
	conversion.rate <- rep(0,length(hit.res))

	hit.cnt <- 0
	last.recall.idx <- 0
	auc <- 0

	for(i in 1:length(hit.res)) {
		if(hit.res[i]) {
			hit.cnt <- hit.cnt + 1
		}
		# at least one hit needed by k
		if (hit.cnt > 0) {
			conversion.rate[i] <- 1
		}
		recall.curve[i] <- hit.cnt / n.ref
		if (hit.cnt == n.ref) {
			last.recall.idx <- i
		}
		precision.curve[i] <- hit.cnt / i
	}

	# if(recall.curve[length(recall.curve)] != 1.0) {
	# 	browser()
	# }

	auc <- sum(precision.curve[1:last.recall.idx])/last.recall.idx 
	return (list("precision" = precision.curve, "recall" = recall.curve,'auc' = auc, "conversion.rate" = conversion.rate))
}

# output results by pr.curve
# save top-5, top-10, top-15, top-20 and plots

save.pr <- function(pr.result, filename) {
	# extract top-5 top-10 top-15 top-20 
	res.tab <- matrix(nrow = 4, ncol = 3)
	rownames(res.tab) <- c("Precision","Recall", "F1 Score","auc")
	colnames(res.tab) <- c("@1","@5","@10")

	res.tab[1,] <- pr.result$precision[c(1,5,10)]
	res.tab[2,] <- pr.result$recall[c(1,5,10)]
	res.tab[3,] <- pr.result$f1[c(1,5,10)]
	res.tab[4,1] <- pr.result$auc

	# dump into file
	fmt.res <- format(res.tab * 100, digits = 2)
	write.csv(fmt.res, file=sprintf("data/%s_pr.csv",filename), quote=FALSE)
	# make the plot
	n <- length(pr.result$recall)
	pdf(sprintf("data/%s_pr.pdf",filename))
	plot(pr.result$recall * 100, pr.result$precision * 100, type="l",xlab="recall (%)",ylab="precision (%)",main="precision-recall curve for CF method")
	dev.off()
}



# model evaluation on the testing split
sgd.rmse.eval <- function(env, test.set.idx) {
  test.data <- env$fil.data[test.set.idx,]
  uids <- test.data$uid
  pids <- test.data$pid
  q <- test.data$quantity
  p <- test.data$price

  model.params <- env$kpr.sgd.params

  # evaluate the utility
  a.vec <- colSums(model.params$user.mat[,uids] * model.params$item.mat[,pids]) + model.params$user.bias[uids,1] + model.params$item.bias[pids,1] + model.params$global.bias
  
  pred.q <- kpr.pred.q(a.vec, p)
  pred.err <- (pred.q - q)^2
  rmse <- sqrt(mean(pred.err))
  result <- list("true.q" = q, "pred.q" = pred.q, "pred.err" = pred.err, "rmse" = rmse, "test.data" = test.data)
  return (result)
}



##### get pr curves for CF ####

# cf.pr.res <- topk.eval(shopcom.env, REC.FUNC = cf.rec.list)
# save.pr(cf.pr.res, filename="cf")


#### get pr curve for KRR utility model ##############
# kpr.pr.res <- topk.eval(shopcom.env, REC.FUNC = kpr.rec.list)
# save.pr(kpr.pr.res, filename="kpr")

# for(cost.percent in seq(0.1,0.5,by=0.1)) {
# 	shopcom.env$cost.percent <- cost.percent
# 	ss.pr.res <- topk.eval(shopcom.env,REC.FUNC=ss.rec.list)
# 	# save it
# 	save.pr(ss.pr.res,sprintf("ss_%.2f",cost.percent))
# }



