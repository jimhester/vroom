#!/usr/bin/env Rscript

lib <- file.path(getwd(), "lib")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)

install.packages("vroom", lib = lib, repos = "https://cloud.r-project.org")
cat("CRAN vroom installed to", lib, "\n")
