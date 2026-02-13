({
  .libPaths(c("lib", .libPaths()))
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom:::vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X1 > 3)
e <- group_by(x, as.integer(X2)) %>% summarise(avg_X1 = mean(X1))
