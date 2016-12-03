library(RUnit)
library(Rcpp)

# source all files
source('logistic.R')
source('evaluation.R')
sourceCpp('rcpp/CSMax.cpp')

test.suite <- defineTestSuite("shopcom",
                              dirs = file.path("tests"),
                              testFileRegexp = '^\\d+\\.R')
 
test.result <- runTestSuite(test.suite)