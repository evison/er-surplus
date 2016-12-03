#' test logistic regression

# test kpr consumer surplus model
test.kpr.delta.cs <- function () {
	a <- 10
	q <- 1
	p <- a * (log(q+1) - log(q))
	checkEquals(0, kpr.delta.cs(a,q,p))
}

# test sigmoid funciton
test.lr.prob <- function() {
	checkEquals(0.5, lr.prob(0))
	checkEquals(1, lr.prob(Inf))
	checkEquals(0, lr.prob(-Inf))
}

# test lr.eval_f
test.lr.eval_f <- function () {
	# construct all parameters
	M <- 1
	N <- 1
	D <- 5
	uid <- c(1)
	pid <- c(1)
	q <- c(1)
	p <- c(1)
	x <- rep(0, (M+N) * D)
	
}
