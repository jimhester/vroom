{
  library(vroom)
  data <- vroom(file, col_types = c(pickup_datetime = "c"))
  data[] <- lapply(data, identity)
}

readr::write_tsv(data, pipe(sprintf("zstd > %s", tempfile(fileext = ".zst"))))
