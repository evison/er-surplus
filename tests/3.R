#### test rcpp code

test.rcpp.kprDelta <- function(){
	a <- 1
	q <- 10
	p <- 10
	checkEquals(kpr.delta.cs(a,q,p), kprDeltaCs(a,q,p));
}

test.rcpp.kprDeltaV <- function(){
	a1 <- c(1,2,3)
	q1 <- c(10,20,30)
	p1 <- c(5,6,7)
	checkEquals(kpr.delta.cs(a1,q1,p1), as.vector(kprDeltaCsV(a1,q1,p1)))
}

test.rcpp.lrProb <- function () {
	delta.cs <- 10
	checkEquals(lr.prob(delta.cs),lrProb(delta.cs))
	delta.cs <- -100
	checkEquals(lr.prob(delta.cs),lrProb(delta.cs))
}


test.rcpp.lrProbV <- function () {
	delta.cs <- c(10,20,30)
	checkEquals(lr.prob(delta.cs),as.vector(lrProbV(delta.cs)))
	delta.cs <- c(-20,-50,-200)
	checkEquals(lr.prob(delta.cs),as.vector(lrProbV(delta.cs)))
}



test.rcpp.logLrProb <- function () {
	delta.cs <- 10
	checkEquals(log.lr.prob(delta.cs),logLrProb(delta.cs))
	delta.cs <- -100
	checkEquals(log.lr.prob(delta.cs),logLrProb(delta.cs))
}


test.rcpp.logLrProbV <- function () {
	delta.cs <- c(10,20,30)
	checkEquals(log.lr.prob(delta.cs),as.vector(logLrProbV(delta.cs)))
	delta.cs <- c(-20,-50,-200)
	checkEquals(log.lr.prob(delta.cs),as.vector(logLrProbV(delta.cs)))
}


test.rcpp.kprPredQ <- function () {
	checkEquals(kpr.pred.q(1,2),kprPredQuantity(1,2))
	checkEquals(kpr.pred.q(5,4),kprPredQuantity(5,4))
}