# output graphs
pdf('data/quantity_hist.pdf')
hist(shopcom.env$fil.data$quantity,freq=F,breaks=20, xlab="quantity", ylab="Density", main="purchase quantity distribution")
dev.off()