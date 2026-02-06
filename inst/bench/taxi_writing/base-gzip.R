{
  library(vroom)
  data <- vroom(file, col_types = c(pickup_datetime = "c"))
  data[] <- lapply(data, identity)
}

{
  con <- gzfile(tempfile(fileext = ".gz"), "wb")
  write.table(data, con, sep = "\t", quote = FALSE, row.names = FALSE)
  close(con)
}
