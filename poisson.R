# verify poisson distribution
# evaluate expected social surplus based on small number of q instances


pos.ss <- function(lambda) {
	res <- 0
	x0 <- as.integer(lambda - 5)
	x1 <- as.integer(lambda + 5)
	for(x in as.integer(0,x0): x1 ) {
		prob <- (lambda^x * exp(-lambda)/factorial(x));
		res <- res + prob * x;
	}
	return (res);
}


pos.prob <- function(lambda, q) {
	return ((lambda^q * exp(-lambda)/factorial(q)))
}

optimal.lambda <- function(cost.percent) {
	lambda.vals <- seq(0.1,10,by=0.1)
	cs.vals <- lambda.vals
	i <- 1
	for (lambda in lambda.vals) {
		q.vals <- 0:50
		q.probs <- pos.prob(lambda, q.vals)
		q.cs <- log(q.vals + 1) - cost.percent * q.vals
		mean.cs <- sum(q.probs * q.cs)
		cs.vals[i] <- mean.cs
		i <- i + 1
	}
	max.idx <- which.max(cs.vals)
	print(sprintf("lambda:%.2f, mean.cs: %.4f",lambda.vals[max.idx],cs.vals[max.idx]))
}