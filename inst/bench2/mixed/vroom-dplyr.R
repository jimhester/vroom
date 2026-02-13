({
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
d <- filter(x, X10 > 3)
e <- group_by(x, X10) %>% summarise(avg = mean(X18))
