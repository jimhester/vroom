#!/usr/bin/env Rscript

# Benchmark: native multi-file path vs per-file libvroom + vec_rbind
# vs legacy (old parser) path
# Usage: Rscript multi_file_benchmark.R

library(bench)
library(vroom)

BENCH_DIR <- file.path(tempdir(), "vroom_multi_file_bench")
dir.create(BENCH_DIR, showWarnings = FALSE, recursive = TRUE)

generate_test_files <- function(n_files, n_rows, type, seed = 42) {
  set.seed(seed)
  filepaths <- character(n_files)

  for (i in seq_len(n_files)) {
    filepath <- file.path(
      BENCH_DIR,
      sprintf("test_%s_%d_%d_file%03d.csv", type, n_files, n_rows, i)
    )

    if (type == "numeric") {
      df <- data.frame(
        id = 1:n_rows,
        num1 = rnorm(n_rows, mean = 100, sd = 50),
        num2 = rnorm(n_rows, mean = 500, sd = 100),
        num3 = runif(n_rows, 0, 1000),
        stringsAsFactors = FALSE
      )
    } else if (type == "character") {
      words <- c(
        "apple", "banana", "cherry", "date", "elderberry",
        "fig", "grape", "honeydew", "kiwi", "lemon"
      )
      df <- data.frame(
        id = 1:n_rows,
        str1 = sample(words, n_rows, replace = TRUE),
        str2 = sample(words, n_rows, replace = TRUE),
        str3 = sample(words, n_rows, replace = TRUE),
        stringsAsFactors = FALSE
      )
    } else {
      words <- c("apple", "banana", "cherry", "date", "elderberry")
      df <- data.frame(
        id = 1:n_rows,
        str1 = sample(words, n_rows, replace = TRUE),
        num1 = rnorm(n_rows, mean = 100, sd = 50),
        lgl1 = sample(c(TRUE, FALSE), n_rows, replace = TRUE),
        stringsAsFactors = FALSE
      )
    }

    vroom::vroom_write(df, filepath, delim = ",")
    filepaths[i] <- filepath
  }

  filepaths
}

# Read + immediate full column access (forces materialization)
read_and_access <- function(filepaths, use_libvroom_val) {
  result <- suppressWarnings(vroom::vroom(
    filepaths,
    delim = ",",
    id = "source",
    show_col_types = FALSE,
    use_libvroom = use_libvroom_val
  ))
  # Force full materialization by accessing all columns
  for (col in result) {
    sum(lengths(col))
  }
  result
}

benchmark_comparison <- function(n_files, n_rows, type, iterations = 5) {
  filepaths <- generate_test_files(n_files, n_rows, type)
  total_size_mb <- sum(file.size(filepaths)) / 1e6

  # Phase 1: Just reading (deferred for native, immediate for legacy)
  read_result <- bench::mark(
    native = vroom::vroom(
      filepaths,
      delim = ",",
      id = "source",
      show_col_types = FALSE,
      use_libvroom = TRUE
    ),
    legacy = suppressWarnings(vroom::vroom(
      filepaths,
      delim = ",",
      id = "source",
      show_col_types = FALSE,
      use_libvroom = FALSE
    )),
    iterations = iterations,
    check = FALSE,
    filter_gc = FALSE
  )

  # Phase 2: Read + access all data (forces materialization)
  access_result <- bench::mark(
    native_access = read_and_access(filepaths, TRUE),
    legacy_access = read_and_access(filepaths, FALSE),
    iterations = iterations,
    check = FALSE,
    filter_gc = FALSE
  )

  unlink(filepaths)

  data.frame(
    n_files = n_files,
    n_rows = n_rows,
    type = type,
    total_rows = n_files * n_rows,
    size_mb = round(total_size_mb, 1),
    native_read_ms = round(as.numeric(read_result$median[1]) * 1000, 1),
    legacy_read_ms = round(as.numeric(read_result$median[2]) * 1000, 1),
    read_speedup = round(
      as.numeric(read_result$median[2]) / as.numeric(read_result$median[1]),
      2
    ),
    native_access_ms = round(as.numeric(access_result$median[1]) * 1000, 1),
    legacy_access_ms = round(as.numeric(access_result$median[2]) * 1000, 1),
    access_speedup = round(
      as.numeric(access_result$median[2]) / as.numeric(access_result$median[1]),
      2
    )
  )
}

check_altrep <- function() {
  cat("\n=== Altrep Preservation Check ===\n")
  filepaths <- generate_test_files(5, 1000, "mixed")

  result <- vroom::vroom(
    filepaths,
    delim = ",",
    id = "source",
    show_col_types = FALSE,
    use_libvroom = TRUE
  )

  cat(sprintf("Result: %d rows x %d cols\n", nrow(result), ncol(result)))

  for (nm in names(result)) {
    inspect_out <- utils::capture.output(.Internal(inspect(result[[nm]])))
    is_altrep <- any(grepl("vroom_arrow_|vroom_rle", inspect_out))
    cat(sprintf(
      "  %-10s (%s): %s\n", nm, typeof(result[[nm]]),
      if (is_altrep) "ALTREP" else "materialized"
    ))
  }

  unlink(filepaths)
}

# Main
if (!interactive()) {
  cat("=== Native vs Legacy Multi-File Benchmark ===\n\n")

  grid <- expand.grid(
    n_files = c(3, 10, 50),
    n_rows = c(1000, 10000),
    type = c("mixed", "numeric", "character"),
    stringsAsFactors = FALSE
  )

  results <- list()
  for (i in seq_len(nrow(grid))) {
    g <- grid[i, ]
    cat(sprintf(
      "  %d files x %d rows (%s)...",
      g$n_files, g$n_rows, g$type
    ))
    r <- tryCatch(
      benchmark_comparison(g$n_files, g$n_rows, g$type),
      error = function(e) {
        cat(sprintf(" ERROR: %s\n", e$message))
        NULL
      }
    )
    if (!is.null(r)) {
      cat(sprintf(
        " read=%.1fx, access=%.1fx\n",
        r$read_speedup, r$access_speedup
      ))
      results[[length(results) + 1]] <- r
    }
  }

  summary_df <- do.call(rbind, results)
  cat("\n=== Results ===\n")
  cat("(speedup > 1.0 = native faster, < 1.0 = legacy faster)\n\n")
  print(tibble::as_tibble(summary_df), n = Inf)

  check_altrep()

  unlink(BENCH_DIR, recursive = TRUE)
  cat("\nDone!\n")
}
