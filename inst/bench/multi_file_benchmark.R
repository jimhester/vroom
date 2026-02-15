#!/usr/bin/env Rscript

# Benchmark: native multi-file path vs per-file libvroom + vec_rbind
# Both paths use the libvroom SIMD parser. The difference is:
# - native: reads all files in one C++ call, returns multi-chunk Altrep
# - per_file: reads each file separately, combines with vctrs::vec_rbind()
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

# Simulate the per-file + vec_rbind path (what vroom does without
# the native multi-file fast path). Each file is read individually
# through libvroom, then combined with vctrs::vec_rbind().
read_per_file_vec_rbind <- function(filepaths, id = "source") {
  results <- lapply(filepaths, function(f) {
    one <- vroom::vroom(
      f,
      delim = ",",
      show_col_types = FALSE
    )
    if (!is.null(id)) {
      one[[id]] <- f
      one <- one[c(id, setdiff(names(one), id))]
    }
    one
  })
  vctrs::vec_rbind(!!!results)
}

benchmark_comparison <- function(n_files, n_rows, type, iterations = 5) {
  filepaths <- generate_test_files(n_files, n_rows, type)
  total_size_mb <- sum(file.size(filepaths)) / 1e6

  # Phase 1: Just reading (deferred for native, eager for per-file+rbind)
  read_result <- bench::mark(
    native = vroom::vroom(
      filepaths,
      delim = ",",
      id = "source",
      show_col_types = FALSE
    ),
    per_file_rbind = read_per_file_vec_rbind(filepaths),
    iterations = iterations,
    check = FALSE,
    filter_gc = FALSE
  )

  # Phase 2: Read + access all data (forces full materialization)
  read_result_access <- bench::mark(
    native_access = {
      res <- vroom::vroom(
        filepaths,
        delim = ",",
        id = "source",
        show_col_types = FALSE
      )
      for (col in res) length(col) # trigger materialization
      res
    },
    per_file_rbind_access = {
      res <- read_per_file_vec_rbind(filepaths)
      for (col in res) length(col)
      res
    },
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
    rbind_read_ms = round(as.numeric(read_result$median[2]) * 1000, 1),
    read_ratio = round(
      as.numeric(read_result$median[1]) / as.numeric(read_result$median[2]),
      2
    ),
    native_access_ms = round(as.numeric(read_result_access$median[1]) * 1000, 1),
    rbind_access_ms = round(as.numeric(read_result_access$median[2]) * 1000, 1),
    access_ratio = round(
      as.numeric(read_result_access$median[1]) / as.numeric(read_result_access$median[2]),
      2
    )
  )
}

check_altrep <- function() {
  cat("\n=== Altrep Preservation Check ===\n")
  filepaths <- generate_test_files(5, 1000, "mixed")

  cat("Native multi-file:\n")
  native <- vroom::vroom(
    filepaths,
    delim = ",",
    id = "source",
    show_col_types = FALSE
  )
  cat(sprintf("  %d rows x %d cols\n", nrow(native), ncol(native)))
  for (nm in names(native)) {
    inspect_out <- utils::capture.output(.Internal(inspect(native[[nm]])))
    is_altrep <- any(grepl("vroom_arrow_|vroom_rle", inspect_out))
    cat(sprintf("  %-10s (%s): %s\n", nm, typeof(native[[nm]]),
      if (is_altrep) "ALTREP" else "materialized"
    ))
  }

  cat("\nPer-file + vec_rbind:\n")
  rbind_result <- read_per_file_vec_rbind(filepaths)
  cat(sprintf("  %d rows x %d cols\n", nrow(rbind_result), ncol(rbind_result)))
  for (nm in names(rbind_result)) {
    inspect_out <- utils::capture.output(.Internal(inspect(rbind_result[[nm]])))
    is_altrep <- any(grepl("vroom_arrow_|vroom_rle|vroom_chr|vroom_", inspect_out))
    cat(sprintf("  %-10s (%s): %s\n", nm, typeof(rbind_result[[nm]]),
      if (is_altrep) "ALTREP" else "materialized"
    ))
  }

  unlink(filepaths)
}

# Main
if (!interactive()) {
  cat("=== Native Multi-File vs Per-File + vec_rbind Benchmark ===\n")
  cat("(Both use libvroom SIMD parser)\n\n")

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
        " read=%.2fx, access=%.2fx\n",
        r$read_ratio, r$access_ratio
      ))
      results[[length(results) + 1]] <- r
    }
  }

  summary_df <- do.call(rbind, results)
  cat("\n=== Results ===\n")
  cat("(ratio = native/rbind; <1 = native faster, >1 = rbind faster)\n\n")
  print(tibble::as_tibble(summary_df), n = Inf)

  check_altrep()

  unlink(BENCH_DIR, recursive = TRUE)
  cat("\nDone!\n")
}
