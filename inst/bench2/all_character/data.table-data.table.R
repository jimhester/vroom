library(data.table)
x <- fread(file, sep = "\t", quote = "", strip.white = FALSE, na.strings = NULL)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- x[sample(NROW(x), 100), ]
d <- x[X1 == "helpless_sheep", ]
e <- x[, .N, by = X1]
