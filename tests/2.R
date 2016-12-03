test.pr.curve <- function() {
	ref.set <- 1:5
	rec.list <- ref.set

precision <- rep(1,5)
	recall <- ref.set / 5
	f1 <- 2 * precision * recall / (precision + recall)
	checkEquals(list("precision" = precision, "recall" = recall, "f1" = f1), pr.curve(ref.set,rec.list))
}