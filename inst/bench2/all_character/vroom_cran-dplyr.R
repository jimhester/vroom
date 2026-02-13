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
d <- filter(x, X1 == "helpless_sheep")
e <- group_by(x, X1) %>% summarise(n = n())
