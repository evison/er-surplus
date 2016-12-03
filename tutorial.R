library("nloptr")

eval_f1 <- function(x,a,b) {
	return ( list(
		"objective" = sqrt(x[2]),
		"gradient" = c(0, 0.5/sqrt(x[2]))
		) )
}

eval_g1 <- function(x,a,b) {
	constraints <- (a * x[1] + b)^3 - x[2]
	jacobian <- rbind(
		c(3 * a[1] * (a[1] * x[1] + b[1])^2, -1)
		,c(3 * a[2] * (a[2] * x[1] + b[2])^2, -1)
	)
	return (list("constraints" = constraints, "jacobian" = jacobian))
}

lb <- c(-Inf, 0)
ub <- c(Inf,Inf)

nloptr(x0 = c(-1.1,10), eval_f = eval_f1, eval_g_ineq = eval_g1, lb = lb, ub = ub, opts = list("algorithm" = "NLOPT_LD_AUGLAG","print_level"=1,"xtol_rel" = 1e-7 , "local_opts" = list("algorithm" = "NLOPT_LD_LBFGS","xtol_rel" = 1e-7)),a=c(2,-1),b=c(0,1))

# nloptr(x0 = c(-1.1,10), eval_f = eval_f1, eval_g_ineq = eval_g1, lb = lb, ub = ub, opts = list("algorithm" = "NLOPT_LD_MMA","print_level"=1,"xtol_rel" = 1e-7 ),a=c(2,-1),b=c(0,1))