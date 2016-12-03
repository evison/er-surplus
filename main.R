# create enviroment for shopcom experiment
source("init.R")
# load logistic regression based algorithm
source("logistic.R")
# load data
load.data(shopcom.env)

# train the model
lr.train(shopcom.env)
lr.test(shopcom.env)

save.data(shopcom.env)