{
  library(vroom)
  data <- vroom(file, col_types = c(pickup_datetime = "c"))
  data[] <- lapply(data, identity)
}

vroom_write(
  data,
  pipe(sprintf("zstd > %s", tempfile(fileext = ".zst"))),
  delim = "\t"
)
