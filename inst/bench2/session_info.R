#!/usr/bin/env Rscript

vroom::vroom_write(
  sessioninfo::package_info(
    c("vroom", "dplyr", "data.table", "arrow", "base"),
    dependencies = FALSE,
    include_base = TRUE
  ),
  here::here("inst", "bench2", "session_info.tsv"),
  delim = "\t"
)
