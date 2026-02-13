library(data.table)
x <- fread(file, sep = "\t", quote = "", strip.white = FALSE, na.strings = NULL)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- x[sample(NROW(x), 100), ]
d <- x[X10 > 3, ]
e <- x[, .(mean(X18)), by = X10]
