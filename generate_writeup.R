pdf("data/quantity_dist.pdf")
hist(shopcom.env$fil.data$quantity[shopcom.env$train.samples], prob=T,breaks=20, xlab="quantity", ylab="probability density", main="Purchase quantity distribuation")
dev.off()