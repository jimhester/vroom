#!/usr/bin/env Rscript
# Profile: vroom_arrow() vs arrow::read_delim_arrow() read performance
# Run from inst/bench2/ directory with input files already generated.
#
# Usage: Rscript profile_read.R [data_type]
#   data_type: "all_numeric", "all_character", or "mixed" (default: all three)

library(vroom)
library(arrow)
library(bench)

args <- commandArgs(trailingOnly = TRUE)
data_types <- if (length(args) > 0) args else c("all_numeric", "all_character", "mixed")

for (dt in data_types) {
  file <- file.path(dt, "input.tsv")
  if (!file.exists(file)) {
    cat(sprintf("Skipping %s (no input.tsv — run 'make %s/input.tsv' first)\n", dt, dt))
    next
  }

  cat(sprintf("\n=== %s ===\n", dt))
  cat(sprintf("File size: %.1f MB\n", file.size(file) / 1e6))

  # Warm up filesystem cache
  invisible(readBin(file, "raw", file.size(file)))

  results <- bench::mark(
    vroom_arrow = vroom_arrow(file, na = character()),
    arrow_table = read_delim_arrow(file, delim = "\t", as_data_frame = FALSE),
    arrow_df = read_delim_arrow(file, delim = "\t", as_data_frame = TRUE),
    min_iterations = 5,
    check = FALSE
  )

  print(results[, c("expression", "min", "median", "mem_alloc", "n_itr")])

  # Also report chunk counts
  cat("\nvroom_arrow chunk structure:\n")
  tbl <- vroom_arrow(file, na = character())
  cat(sprintf("  Chunks: %d\n", length(tbl$batches())))
  cat(sprintf("  Rows: %s\n", format(tbl$num_rows, big.mark = ",")))
  cat(sprintf("  Cols: %d\n", tbl$num_columns))
}
