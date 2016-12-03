#### generate data report needed by paper

# generate top100 conversion rate curves with varying regularizer
source('evaluation.R')


# conversion rate and social surplus
gen.conversion.rate.report <- function () {

	algorithms <- c("quantity_cf","SSMAX_gamma0.1","SSMAX_gamma1","SSMAX_gamma2","SSMAX_gamma5","SSMAX_gamma10")
	legends <- c("CF", expression(TSM^{0.1}),expression(TSM^{1}),expression(TSM^{2}),expression(TSM^{5}),expression(TSM^{10}))

	k.samples <- c(5,10,15,20)

	# only conversion rate curves
	shopcom.env$eval$crs <- matrix(nrow=100,ncol=length(algorithms))

	shopcom.env$eval$crs.samples <- matrix(nrow=length(algorithms),ncol=length(k.samples))
	rownames(shopcom.env$eval$crs.samples) <- legends
	colnames(shopcom.env$eval$crs.samples) <- sprintf("@%d",k.samples)

	key.names <- c()
	for (i in 1:length(algorithms)) {
		algorithm <- algorithms[i]
		# conversion rate
		shopcom.env$eval$pr[[algorithm]] <- topk.eval(shopcom.env,ssmax.rec.list, list("model.name" = algorithm))
		# also load the social surplus
		shopcom.env$eval$crs[,i] <- shopcom.env$eval$pr[[algorithm]]$conversion.rate
		shopcom.env$eval$crs.samples[i,] <- shopcom.env$eval$crs[k.samples,i]
	}

	colnames(shopcom.env$eval$crs) <- algorithms
	# pretty strange, instead times by 100, times by 10
	shopcom.env$eval$crs <- shopcom.env$eval$crs * 10
	# combine all conversion rate curves
	pdf("data/writeup/conversionrate.pdf")
	matplot(1:100, shopcom.env$eval$crs, type="l", xlab="N", ylab="Conversion rate (%)")
	legend("topleft",legend=legends, lty=1:length(algorithms), col=1:length(algorithms))
	dev.off()

	# write the samples
	write.table(shopcom.env$eval$crs.samples, file="data/writeup/ss_conversion_rate.csv",quote=F)
}


# now do the plot
gen.ss.report <- function () {
	algorithms <- c("quantity_cf","SSMAX_gamma0.1","SSMAX_gamma1","SSMAX_gamma2","SSMAX_gamma5","SSMAX_gamma10")
	legends <- c("CF", expression(TSM^{0.1}),expression(TSM^{1}),expression(TSM^{2}),expression(TSM^{5}),expression(TSM^{10}))
	
	algorithm.ss <- matrix(ncol=length(algorithms), nrow=100)
	colnames(algorithm.ss) <- legends
	rownames(algorithm.ss) <- 1:100

	k.samples <- c(5,10,15,20)
	algorithm.ss.sample <- matrix(nrow=length(algorithms),ncol=length(k.samples))
	rownames(algorithm.ss.sample) <- legends
	colnames(algorithm.ss.sample) <- sprintf("@%d",k.samples)

	for(i in 1:length(algorithms)) {
		algorithm <- algorithms[i]
		ss.file <- sprintf("data/1016/%s_sssum_top100.csv",algorithm)
		user.ss <- read.csv(file=ss.file, header=F)
		user.ss.mean <- colMeans(user.ss)
		algorithm.ss[,i] <- user.ss.mean
		algorithm.ss.sample[i,] <- user.ss.mean[k.samples]
	}

	pdf("data/writeup/topkss.pdf")
	matplot(1:100, algorithm.ss, type="l", xlab="N", ylab="Social surplus per user ($)")
	legend("topleft",legend=legends, lty=1:length(algorithms), col=1:length(algorithms))
	dev.off()

	# write samples
	write.table(algorithm.ss.sample,file="data/writeup/sssum_samples.csv",quote=F)
}