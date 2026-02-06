{
  library(vroom)
  data <- vroom(file, col_types = c(pickup_datetime = "c"))
  data[] <- lapply(data, identity)
}

data.table::fwrite(data, tempfile(fileext = ".gz"), sep = "\t")
