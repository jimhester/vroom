#!/usr/bin/env Rscript

# Benchmark: multi-file read performance comparison
#
# Compares three approaches:
# 1. native: new multi-file fast path (one C++ call, multi-chunk Altrep)
# 2. per_file_rbind: per-file libvroom + vctrs::vec_rbind() (current dev fallback)
# 3. cran_vroom: CRAN vroom (1.6.5) multi-file read
#
# Usage: Rscript multi_file_benchmark.R [--skip-cran]

library(bench)
library(vroom)

BENCH_DIR <- file.path(tempdir(), "vroom_multi_file_bench")
dir.create(BENCH_DIR, showWarnings = FALSE, recursive = TRUE)

# Install CRAN vroom into a local lib/ directory for comparison
setup_cran_vroom <- function() {
  lib <- file.path(BENCH_DIR, "lib")
  if (!file.exists(file.path(lib, "vroom"))) {
    cat("Installing CRAN vroom to", lib, "...\n")
    dir.create(lib, recursive = TRUE, showWarnings = FALSE)
    install.packages("vroom", lib = lib, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  lib
}

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

# Per-file + vec_rbind path (current dev fallback when native path unavailable)
read_per_file_vec_rbind <- function(filepaths, id = "source") {
  results <- lapply(filepaths, function(f) {
    one <- vroom::vroom(f, delim = ",", show_col_types = FALSE)
    if (!is.null(id)) {
      one[[id]] <- f
      one <- one[c(id, setdiff(names(one), id))]
    }
    one
  })
  vctrs::vec_rbind(!!!results)
}

# CRAN vroom multi-file read (uses separate lib)
read_cran_vroom <- function(filepaths, cran_lib) {
  # Load CRAN vroom in a subprocess to avoid namespace conflicts
  callr::r(
    function(files, lib) {
      .libPaths(c(lib, .libPaths()))
      library(vroom, lib.loc = lib)
      vroom::vroom(files, delim = ",", id = "source", show_col_types = FALSE)
    },
    args = list(files = filepaths, lib = cran_lib),
    show = FALSE
  )
}

benchmark_comparison <- function(
    n_files, n_rows, type, cran_lib = NULL, iterations = 5) {
  filepaths <- generate_test_files(n_files, n_rows, type)
  total_size_mb <- sum(file.size(filepaths)) / 1e6

  exprs <- list(
    native = rlang::expr(vroom::vroom(
      filepaths,
      delim = ",",
      id = "source",
      show_col_types = FALSE
    )),
    per_file_rbind = rlang::expr(read_per_file_vec_rbind(filepaths))
  )

  access_exprs <- list(
    native_access = rlang::expr({
      res <- vroom::vroom(
        filepaths,
        delim = ",",
        id = "source",
        show_col_types = FALSE
      )
      for (col in res) length(col)
      res
    }),
    per_file_rbind_access = rlang::expr({
      res <- read_per_file_vec_rbind(filepaths)
      for (col in res) length(col)
      res
    })
  )

  if (!is.null(cran_lib)) {
    exprs$cran_vroom <- rlang::expr(read_cran_vroom(filepaths, cran_lib))
    access_exprs$cran_vroom_access <- rlang::expr({
      res <- read_cran_vroom(filepaths, cran_lib)
      for (col in res) length(col)
      res
    })
  }

  read_result <- bench::mark(
    !!!exprs,
    iterations = iterations,
    check = FALSE,
    filter_gc = FALSE
  )

  access_result <- bench::mark(
    !!!access_exprs,
    iterations = iterations,
    check = FALSE,
    filter_gc = FALSE
  )

  row <- data.frame(
    n_files = n_files,
    n_rows = n_rows,
    type = type,
    total_rows = n_files * n_rows,
    size_mb = round(total_size_mb, 1),
    native_read_ms = round(as.numeric(read_result$median[1]) * 1000, 1),
    rbind_read_ms = round(as.numeric(read_result$median[2]) * 1000, 1),
    native_access_ms = round(as.numeric(access_result$median[1]) * 1000, 1),
    rbind_access_ms = round(as.numeric(access_result$median[2]) * 1000, 1),
    stringsAsFactors = FALSE
  )

  if (!is.null(cran_lib)) {
    row$cran_read_ms <- round(as.numeric(read_result$median[3]) * 1000, 1)
    row$cran_access_ms <- round(as.numeric(access_result$median[3]) * 1000, 1)
  }

  unlink(filepaths)
  row
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
    cat(sprintf(
      "  %-10s (%s): %s\n", nm, typeof(native[[nm]]),
      if (is_altrep) "ALTREP" else "materialized"
    ))
  }

  cat("\nPer-file + vec_rbind:\n")
  rbind_result <- read_per_file_vec_rbind(filepaths)
  cat(sprintf("  %d rows x %d cols\n", nrow(rbind_result), ncol(rbind_result)))
  for (nm in names(rbind_result)) {
    inspect_out <- utils::capture.output(.Internal(inspect(rbind_result[[nm]])))
    is_altrep <- any(grepl("vroom_arrow_|vroom_rle|vroom_chr|vroom_", inspect_out))
    cat(sprintf(
      "  %-10s (%s): %s\n", nm, typeof(rbind_result[[nm]]),
      if (is_altrep) "ALTREP" else "materialized"
    ))
  }

  unlink(filepaths)
}

# Main
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  skip_cran <- "--skip-cran" %in% args

  cat("=== Multi-File Read Performance Benchmark ===\n\n")

  cran_lib <- NULL
  if (!skip_cran) {
    if (!requireNamespace("callr", quietly = TRUE)) {
      cat("callr not installed, skipping CRAN vroom comparison\n")
      cat("Install with: install.packages('callr')\n\n")
    } else {
      cran_lib <- tryCatch(
        setup_cran_vroom(),
        error = function(e) {
          cat(sprintf("Could not install CRAN vroom: %s\n", e$message))
          cat("Skipping CRAN comparison\n\n")
          NULL
        }
      )
    }
  }

  approaches <- "native (multi-chunk Altrep) vs per-file + vec_rbind"
  if (!is.null(cran_lib)) {
    approaches <- paste0(approaches, " vs CRAN vroom")
  }
  cat("Comparing: ", approaches, "\n\n")

  grid <- expand.grid(
    n_files = c(3, 10, 50),
    n_rows = c(1000, 10000),
    type = c("mixed", "numeric", "character"),
    stringsAsFactors = FALSE
  )

  results <- list()
  for (i in seq_len(nrow(grid))) {
    g <- grid[i, ]
    cat(sprintf("  %d files x %d rows (%s)...", g$n_files, g$n_rows, g$type))
    r <- tryCatch(
      benchmark_comparison(g$n_files, g$n_rows, g$type, cran_lib),
      error = function(e) {
        cat(sprintf(" ERROR: %s\n", e$message))
        NULL
      }
    )
    if (!is.null(r)) {
      ratio <- r$native_access_ms / r$rbind_access_ms
      msg <- sprintf(" native/rbind=%.2fx", ratio)
      if (!is.null(cran_lib) && !is.null(r$cran_access_ms)) {
        cran_ratio <- r$native_access_ms / r$cran_access_ms
        msg <- paste0(msg, sprintf(", native/cran=%.2fx", cran_ratio))
      }
      cat(msg, "\n")
      results[[length(results) + 1]] <- r
    }
  }

  summary_df <- do.call(rbind, results)
  cat("\n=== Results ===\n")
  cat("(times in ms; ratio <1 = native faster)\n\n")
  print(tibble::as_tibble(summary_df), n = Inf, width = Inf)

  check_altrep()

  unlink(BENCH_DIR, recursive = TRUE)
  cat("\nDone!\n")
}
