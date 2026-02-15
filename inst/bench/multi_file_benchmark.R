#!/usr/bin/env Rscript

# Benchmark script for measuring multi-file read performance
# Establishes baseline numbers with current vec_rbind implementation
# Usage: Rscript multi_file_benchmark.R

library(bench)
library(vroom)
library(tibble)

# Configuration
BENCH_DIR <- file.path(tempdir(), "vroom_multi_file_bench")
dir.create(BENCH_DIR, showWarnings = FALSE, recursive = TRUE)

# Helper to generate test CSV files
generate_test_files <- function(n_files, n_rows, type, seed = 42) {
  set.seed(seed)

  cat(sprintf(
    "Generating %d files (%d rows, type=%s)...\n",
    n_files,
    n_rows,
    type
  ))

  filepaths <- character(n_files)

  for (i in seq_len(n_files)) {
    filepath <- file.path(
      BENCH_DIR,
      sprintf("test_%s_%d_%d_file%03d.csv", type, n_files, n_rows, i)
    )

    # Create different column type mixes
    if (type == "numeric") {
      df <- data.frame(
        id = 1:n_rows,
        num1 = rnorm(n_rows, mean = 100, sd = 50),
        num2 = rnorm(n_rows, mean = 500, sd = 100),
        num3 = runif(n_rows, 0, 1000),
        num4 = rnorm(n_rows, mean = 0, sd = 1),
        stringsAsFactors = FALSE
      )
    } else if (type == "character") {
      words <- c(
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon",
        "mango",
        "nectarine",
        "orange",
        "papaya",
        "quince"
      )
      df <- data.frame(
        id = 1:n_rows,
        str1 = sample(words, n_rows, replace = TRUE),
        str2 = sample(words, n_rows, replace = TRUE),
        str3 = sample(words, n_rows, replace = TRUE),
        str4 = sample(words, n_rows, replace = TRUE),
        stringsAsFactors = FALSE
      )
    } else {
      # mixed
      words <- c(
        "apple",
        "banana",
        "cherry",
        "date",
        "elderberry",
        "fig",
        "grape",
        "honeydew",
        "kiwi",
        "lemon"
      )
      df <- data.frame(
        id = 1:n_rows,
        str1 = sample(words, n_rows, replace = TRUE),
        str2 = sample(words, n_rows, replace = TRUE),
        num1 = rnorm(n_rows, mean = 100, sd = 50),
        num2 = rnorm(n_rows, mean = 500, sd = 100),
        int1 = sample.int(10000, n_rows, replace = TRUE),
        stringsAsFactors = FALSE
      )
    }

    vroom::vroom_write(df, filepath, delim = ",")
    filepaths[i] <- filepath
  }

  total_size_mb <- sum(file.size(filepaths)) / 1e6
  cat(sprintf("  Total size: %.1f MB\n", total_size_mb))

  filepaths
}

# Benchmark multi-file reading
benchmark_multi_file <- function(n_files, n_rows, type, iterations = 3) {
  # Generate test files
  filepaths <- generate_test_files(n_files, n_rows, type)

  # Benchmark reading all files at once
  result <- bench::mark(
    multi_file = vroom::vroom(
      filepaths,
      delim = ",",
      show_col_types = FALSE,
      altrep = TRUE
    ),
    iterations = iterations,
    check = FALSE,
    filter_gc = FALSE
  )

  # Clean up
  unlink(filepaths)

  result
}

# Run benchmarks across parameter grid
run_benchmark_grid <- function() {
  cat("=== Multi-File Read Benchmark ===\n")
  cat(sprintf("Benchmark directory: %s\n\n", BENCH_DIR))

  # Parameter grid
  n_files_vec <- c(3, 10, 50, 200)
  n_rows_vec <- c(100, 10000)
  type_vec <- c("mixed", "numeric", "character")

  results <- list()

  for (n_files in n_files_vec) {
    for (n_rows in n_rows_vec) {
      for (type in type_vec) {
        cat(sprintf(
          "\n--- n_files=%d, n_rows=%d, type=%s ---\n",
          n_files,
          n_rows,
          type
        ))

        result <- tryCatch(
          {
            bm <- benchmark_multi_file(n_files, n_rows, type)

            median_time_ms <- as.numeric(bm$median[1]) * 1000
            mem_alloc_mb <- as.numeric(bm$mem_alloc[1]) / 1e6

            cat(sprintf("  Median time: %.1f ms\n", median_time_ms))
            cat(sprintf("  Memory alloc: %.1f MB\n", mem_alloc_mb))

            list(
              n_files = n_files,
              n_rows = n_rows,
              type = type,
              median_ms = median_time_ms,
              mem_alloc_mb = mem_alloc_mb,
              total_rows = n_files * n_rows
            )
          },
          error = function(e) {
            cat(sprintf("  ERROR: %s\n", e$message))
            NULL
          }
        )

        if (!is.null(result)) {
          results[[length(results) + 1]] <- result
        }
      }
    }
  }

  # Summary table
  cat("\n\n=== Results Summary ===\n")
  summary_df <- do.call(rbind, lapply(results, as.data.frame))
  summary_df <- tibble::as_tibble(summary_df)
  print(summary_df)

  summary_df
}

# Check Altrep preservation after multi-file reads
check_altrep_preservation <- function() {
  cat("\n\n=== Altrep Preservation Check ===\n\n")

  # Generate a small set of test files
  filepaths <- generate_test_files(n_files = 5, n_rows = 100, type = "mixed")

  # Read with Altrep enabled
  cat("\nReading 5 files with altrep=TRUE...\n")
  df <- vroom::vroom(
    filepaths,
    delim = ",",
    show_col_types = FALSE,
    altrep = TRUE
  )

  cat(sprintf("Result dimensions: %d rows x %d cols\n", nrow(df), ncol(df)))
  cat("\nColumn types and Altrep status:\n")

  for (col_name in names(df)) {
    col <- df[[col_name]]
    col_class <- class(col)[1]

    # Capture .Internal(inspect()) output to check for "altrep"
    inspect_output <- capture.output(.Internal(inspect(col)))
    is_altrep <- any(grepl("altrep", inspect_output, ignore.case = TRUE))

    cat(sprintf(
      "  %s (%s): %s\n",
      col_name,
      col_class,
      if (is_altrep) "ALTREP" else "materialized"
    ))
  }

  cat("\nSample of first few rows:\n")
  print(head(df, 3))

  # Clean up
  unlink(filepaths)

  invisible(df)
}

# Main execution
if (!interactive()) {
  # Run full benchmark grid
  results <- run_benchmark_grid()

  # Check Altrep preservation
  check_altrep_preservation()

  # Clean up benchmark directory
  cat("\n\nCleaning up...\n")
  unlink(BENCH_DIR, recursive = TRUE)

  cat("\nBenchmark complete!\n")
}
