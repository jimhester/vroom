{
  library(vroom)
  data <- vroom(file, col_types = c(pickup_datetime = "c"))
  data[] <- lapply(data, identity)
}

write.table(
  data,
  tempfile(fileext = ".tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
