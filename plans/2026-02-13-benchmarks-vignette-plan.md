# Benchmarks Vignette Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a new benchmarks vignette comparing vroom (libvroom) vs CRAN vroom vs arrow vs data.table, with a Makefile-driven benchmark runner.

**Architecture:** Benchmark scripts in `inst/bench2/` follow the existing `inst/bench/` pattern: per-package .R files run by a shared `run-bench.R` runner, orchestrated by a GNUmakefile. Results aggregate into summary TSV files. The vignette reads TSVs and visualizes with ggplot2.

**Tech Stack:** R, bench, ggplot2, patchwork, dplyr, data.table, arrow, knitr/rmarkdown

---

### Task 1: Create benchmark infrastructure files

**Files:**
- Create: `inst/bench2/run-bench.R`
- Create: `inst/bench2/setup.R`
- Create: `inst/bench2/summarise-benchmarks.R`
- Create: `inst/bench2/session_info.R`
- Create: `inst/bench2/GNUmakefile`

**Step 1: Create `inst/bench2/run-bench.R`**

This is the runner that executes a benchmark .R file and captures timing. Nearly identical to `inst/bench/run-bench.R`.

```r
#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
source_file <- args[[1]]
out_file <- args[[2]]

file <- args[-c(1:2)]

cat(source_file, "\n")
out <- bench::workout_expressions(as.list(parse(
  source_file,
  keep.source = FALSE
)))

x <- vroom::vroom(file, col_types = list())

out$size <- sum(file.size(file))
out$rows <- nrow(x)
out$cols <- ncol(x)
out$process <- as.numeric(out$process)
out$real <- as.numeric(out$real)
out$max_memory <- as.numeric(bench::bench_process_memory()[["max"]])

vroom::vroom_write(out, out_file)
```

Make it executable: `chmod +x inst/bench2/run-bench.R`

**Step 2: Create `inst/bench2/setup.R`**

Installs CRAN vroom into a local `lib/` directory for the `vroom_cran` benchmarks.

```r
#!/usr/bin/env Rscript

lib <- file.path(getwd(), "lib")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)

install.packages("vroom", lib = lib, repos = "https://cloud.r-project.org")
cat("CRAN vroom installed to", lib, "\n")
```

Make it executable: `chmod +x inst/bench2/setup.R`

**Step 3: Create `inst/bench2/summarise-benchmarks.R`**

Aggregates per-package .tsv files in each scenario directory into one summary .tsv per scenario.

```r
library(vroom)
library(dplyr)
library(fs)
library(purrr)
library(tidyr)

summarise_dir <- function(dir, desc) {
  out_file <- path(path_dir(dir), path_ext_set(path_file(dir), "tsv"))
  col_types <- cols(
    exprs = col_character(),
    process = col_character(),
    real = col_character(),
    size = col_double(),
    rows = col_double(),
    cols = col_double()
  )

  dir_ls(dir, glob = "*tsv") %>%
    discard(~ endsWith(.x, "input.tsv")) %>%
    vroom(id = "path", col_types = col_types) %>%
    mutate(path = path_ext_remove(path_file(path))) %>%
    group_by(path) %>%
    mutate(op = desc) %>%
    separate(path, c("reading_package", "manip_package"), "-") %>%
    pivot_longer(
      .,
      cols = c(process, real),
      names_to = "type",
      values_to = "time"
    ) %>%
    select(
      reading_package,
      manip_package,
      op,
      type,
      time,
      size,
      max_memory,
      rows,
      cols
    ) %>%
    vroom_write(out_file, delim = "\t")
}

desc <- c("setup", "read", "materialize", "print", "head", "tail", "sample", "filter", "aggregate")

summarise_dir(
  here::here("inst/bench2/mixed"),
  desc
)
summarise_dir(
  here::here("inst/bench2/all_numeric"),
  desc
)
summarise_dir(
  here::here("inst/bench2/all_character"),
  desc
)
```

**Step 4: Create `inst/bench2/session_info.R`**

```r
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
```

Make it executable: `chmod +x inst/bench2/session_info.R`

**Step 5: Create `inst/bench2/GNUmakefile`**

```makefile
MAKEFLAGS += --no-builtin-rules

BENCH_ROWS := 1000000
BENCH_COLS := 25

BENCH_INPUTS := mixed/input.tsv all_numeric/input.tsv all_character/input.tsv
BENCH_SRC := $(wildcard */*.R)
BENCH_MARKS := $(filter-out $(BENCH_INPUTS:.tsv=.R),$(BENCH_SRC))
BENCH_MARKS_TSV := $(BENCH_MARKS:.R=.tsv)
BENCH_OUT := mixed.tsv all_numeric.tsv all_character.tsv

all: $(BENCH_OUT) session_info.tsv

$(BENCH_OUT): $(BENCH_MARKS_TSV)
	Rscript summarise-benchmarks.R

session_info.tsv: session_info.R
	./$<

lib/vroom: setup.R
	Rscript $<
	touch $@

%/input.tsv: %/input.R
	Rscript $< $(BENCH_ROWS) $(BENCH_COLS) $@

# vroom_cran benchmarks depend on lib/vroom being installed
vroom_cran-%.tsv: vroom_cran-%.R lib/vroom
	./run-bench.R $< $@ $(@D)/input.tsv

%.tsv: %.R
	./run-bench.R $< $@ $(@D)/input.tsv

clean:
	rm -f $(BENCH_INPUTS) $(BENCH_MARKS_TSV) $(BENCH_OUT) session_info.tsv
	rm -rf lib/
```

**Step 6: Commit**

```bash
git add inst/bench2/run-bench.R inst/bench2/setup.R inst/bench2/summarise-benchmarks.R inst/bench2/session_info.R inst/bench2/GNUmakefile
git commit -m "feat: add benchmark infrastructure for libvroom comparison (issue #73)"
```

---

### Task 2: Create mixed-type scenario benchmark scripts

**Files:**
- Create: `inst/bench2/mixed/input.R`
- Create: `inst/bench2/mixed/vroom-dplyr.R`
- Create: `inst/bench2/mixed/vroom_cran-dplyr.R`
- Create: `inst/bench2/mixed/arrow-dplyr.R`
- Create: `inst/bench2/mixed/data.table-data.table.R`

**Step 1: Create `inst/bench2/mixed/input.R`**

Generates synthetic mixed-type data (strings + integers + doubles).

```r
args <- commandArgs(trailingOnly = TRUE)
rows <- as.integer(args[[1]])
cols <- as.integer(args[[2]])
output <- args[[3]]

set.seed(42)
RNGversion("3.5.3")

# Mixed types: ~1/3 character, ~1/3 integer, ~1/3 double
n_chr <- ceiling(cols / 3)
n_int <- ceiling(cols / 3)
n_dbl <- cols - n_chr - n_int

col_types <- stats::setNames(
  c(
    rep(list(vroom::col_character()), n_chr),
    rep(list(vroom::col_integer()), n_int),
    rep(list(vroom::col_double()), n_dbl)
  ),
  make.names(seq_len(cols))
)

data <- vroom::gen_tbl(rows, cols, col_types = col_types)

vroom::vroom_write(data, output, "\t")
```

**Step 2: Create `inst/bench2/mixed/vroom-dplyr.R`**

```r
({
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X1 > 3)
e <- group_by(x, as.integer(X2)) %>% summarise(avg_X1 = mean(X1))
```

Note: The filter/aggregate operations reference column names that depend on the generated data. For mixed data, string columns are named X1..Xn_chr, integer columns Xn_chr+1..., double columns the rest. Since `gen_tbl` uses `make.names(seq_len(cols))`, columns will be `X1`, `X2`, ..., `X25`. The first ~9 columns are character, next ~8 integer, rest double.

We need to adjust filter/aggregate to work across all packages. For mixed data, let's filter on a character column and aggregate on a numeric column. Since the column names are `X1` through `X25`:
- X1..X9 are character
- X10..X17 are integer
- X18..X25 are double

Filter: `X1 == <some_value>` (character column)
Aggregate: mean of X18 (double) grouped by X10 (integer)

Revised `inst/bench2/mixed/vroom-dplyr.R`:

```r
({
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X10 > 3)
e <- group_by(x, X10) %>% summarise(avg = mean(X18))
```

**Step 3: Create `inst/bench2/mixed/vroom_cran-dplyr.R`**

```r
({
  .libPaths(c("lib", .libPaths()))
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X10 > 3)
e <- group_by(x, X10) %>% summarise(avg = mean(X18))
```

**Step 4: Create `inst/bench2/mixed/arrow-dplyr.R`**

```r
({
  library(arrow)
  library(dplyr)
})
x <- read_csv_arrow(file, as_data_frame = TRUE)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- slice_sample(x, n = 100)
d <- filter(x, X10 > 3)
e <- group_by(x, X10) %>% summarise(avg = mean(X18))
```

Note: the `invisible(NULL)` line corresponds to the "materialize" operation slot. Arrow fully materializes on read, so this is a no-op placeholder to keep the operation count consistent.

**Step 5: Create `inst/bench2/mixed/data.table-data.table.R`**

```r
library(data.table)
x <- fread(file, sep = "\t", quote = "", strip.white = FALSE, na.strings = NULL)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- x[sample(NROW(x), 100), ]
d <- x[X10 > 3, ]
e <- x[, .(mean(X18)), by = X10]
```

**Step 6: Commit**

```bash
git add inst/bench2/mixed/
git commit -m "feat: add mixed-type benchmark scenario (issue #73)"
```

---

### Task 3: Create all-numeric scenario benchmark scripts

**Files:**
- Create: `inst/bench2/all_numeric/input.R`
- Create: `inst/bench2/all_numeric/vroom-dplyr.R`
- Create: `inst/bench2/all_numeric/vroom_cran-dplyr.R`
- Create: `inst/bench2/all_numeric/arrow-dplyr.R`
- Create: `inst/bench2/all_numeric/data.table-data.table.R`

**Step 1: Create `inst/bench2/all_numeric/input.R`**

```r
args <- commandArgs(trailingOnly = TRUE)
rows <- as.integer(args[[1]])
cols <- as.integer(args[[2]])
output <- args[[3]]

set.seed(42)
RNGversion("3.5.3")

data <- vroom::gen_tbl(rows, cols, col_types = strrep("d", cols))

vroom::vroom_write(data, output, "\t")
```

**Step 2: Create `inst/bench2/all_numeric/vroom-dplyr.R`**

```r
({
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X1 > 3)
e <- group_by(x, as.integer(X2)) %>% summarise(avg_X1 = mean(X1))
```

**Step 3: Create `inst/bench2/all_numeric/vroom_cran-dplyr.R`**

```r
({
  .libPaths(c("lib", .libPaths()))
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X1 > 3)
e <- group_by(x, as.integer(X2)) %>% summarise(avg_X1 = mean(X1))
```

**Step 4: Create `inst/bench2/all_numeric/arrow-dplyr.R`**

```r
({
  library(arrow)
  library(dplyr)
})
x <- read_csv_arrow(file, as_data_frame = TRUE)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- slice_sample(x, n = 100)
d <- filter(x, X1 > 3)
e <- group_by(x, as.integer(X2)) %>% summarise(avg_X1 = mean(X1))
```

**Step 5: Create `inst/bench2/all_numeric/data.table-data.table.R`**

```r
library(data.table)
x <- fread(file, sep = "\t", quote = "", strip.white = FALSE, na.strings = NULL)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- x[sample(NROW(x), 100), ]
d <- x[X1 > 3, ]
e <- x[, .(mean(X1)), by = as.integer(X2)]
```

**Step 6: Commit**

```bash
git add inst/bench2/all_numeric/
git commit -m "feat: add all-numeric benchmark scenario (issue #73)"
```

---

### Task 4: Create all-character scenario benchmark scripts

**Files:**
- Create: `inst/bench2/all_character/input.R`
- Create: `inst/bench2/all_character/vroom-dplyr.R`
- Create: `inst/bench2/all_character/vroom_cran-dplyr.R`
- Create: `inst/bench2/all_character/arrow-dplyr.R`
- Create: `inst/bench2/all_character/data.table-data.table.R`

**Step 1: Create `inst/bench2/all_character/input.R`**

Uses factor column with known rare level for filtering (same pattern as existing `inst/bench/all_character-long/input.R`).

```r
args <- commandArgs(trailingOnly = TRUE)
rows <- as.integer(args[[1]])
cols <- as.integer(args[[2]])
output <- args[[3]]

set.seed(42)
RNGversion("3.5.3")

library(vroom)

# We want ~ 1000 rows to filter
num_levels <- 5
levels <- c("helpless_sheep", gen_name(num_levels - 1))

filt_p <- 1000 / rows

# The prob for the rest should just be evenly spaced
rest_p <- rep((1 - filt_p) / (num_levels - 1), num_levels - 1)

col_types <- stats::setNames(
  c(
    list(
      col_factor(levels = levels, prob = c(filt_p, rest_p))
    ),
    rep(list(col_character()), cols - 1)
  ),
  make.names(seq_len(cols))
)

data <- gen_tbl(rows, cols, col_types = col_types)

vroom_write(data, output, "\t")
```

**Step 2: Create `inst/bench2/all_character/vroom-dplyr.R`**

```r
({
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X1 == "helpless_sheep")
e <- group_by(x, X1) %>% summarise(n = n())
```

**Step 3: Create `inst/bench2/all_character/vroom_cran-dplyr.R`**

```r
({
  .libPaths(c("lib", .libPaths()))
  library(vroom)
  library(dplyr)
})
x <- vroom(
  file,
  trim_ws = FALSE,
  quote = "",
  escape_double = FALSE,
  na = character()
)
vroom_materialize(x, replace = FALSE)
print(x)
a <- head(x)
b <- tail(x)
c <- sample_n(x, 100)
d <- filter(x, X1 == "helpless_sheep")
e <- group_by(x, X1) %>% summarise(n = n())
```

**Step 4: Create `inst/bench2/all_character/arrow-dplyr.R`**

```r
({
  library(arrow)
  library(dplyr)
})
x <- read_csv_arrow(file, as_data_frame = TRUE)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- slice_sample(x, n = 100)
d <- filter(x, X1 == "helpless_sheep")
e <- group_by(x, X1) %>% summarise(n = n())
```

**Step 5: Create `inst/bench2/all_character/data.table-data.table.R`**

```r
library(data.table)
x <- fread(file, sep = "\t", quote = "", strip.white = FALSE, na.strings = NULL)
invisible(NULL)
print(x)
a <- head(x)
b <- tail(x)
c <- x[sample(NROW(x), 100), ]
d <- x[X1 == "helpless_sheep", ]
e <- x[, .N, by = X1]
```

**Step 6: Commit**

```bash
git add inst/bench2/all_character/
git commit -m "feat: add all-character benchmark scenario (issue #73)"
```

---

### Task 5: Update .Rbuildignore and add DESCRIPTION dependencies

**Files:**
- Modify: `.Rbuildignore`
- Modify: `DESCRIPTION`

**Step 1: Add bench2 patterns to `.Rbuildignore`**

Append these lines:

```
^inst/bench2/.+/.*\.tsv$
^inst/bench2/lib$
```

**Step 2: Add arrow to DESCRIPTION Suggests**

Add `arrow,` to the Suggests list in `DESCRIPTION` (alphabetical order, after `archive`).

**Step 3: Add data.table to DESCRIPTION Suggests**

Add `data.table,` to the Suggests list (alphabetical order, after `curl`).

**Step 4: Add callr to DESCRIPTION Suggests (if not already present)**

Check if `callr` is needed. Actually callr is not needed since we run separate Rscript processes via the Makefile. No callr dependency needed.

**Step 5: Add sessioninfo to DESCRIPTION Suggests**

Add `sessioninfo,` to the Suggests list (alphabetical order, after `scales`). This is needed by `session_info.R`.

**Step 6: Commit**

```bash
git add .Rbuildignore DESCRIPTION
git commit -m "chore: add bench2 to .Rbuildignore and new Suggests deps (issue #73)"
```

---

### Task 6: Create the vignette

**Files:**
- Create: `vignettes/benchmarks-libvroom.Rmd`

**Step 1: Create the vignette file**

```rmd
---
title: "libvroom Benchmarks"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{libvroom Benchmarks}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---
```

The setup chunk (hidden) defines helper functions matching the existing `benchmarks.Rmd` style:

- `pretty_sec()` - format seconds
- `read_benchmark()` - read summary TSV, parse reading_package/manip_package, create labels, factor ops
- `generate_subtitle()` - "rows x cols - size"
- `plot_benchmark()` - horizontal stacked bar chart + memory side panel via patchwork
- `make_table()` - kable HTML table with pretty-printed times

Then three sections (mixed, all_numeric, all_character) each with:
- A `read_benchmark()` call to load the summary TSV
- `plot_benchmark()` call
- `make_table()` call
- Brief prose explanation

And a session info section at the end.

The full vignette content is provided in the implementation (too long to include inline in the plan — the implementer should follow the exact structure from the existing `vignettes/benchmarks.Rmd` but adapted for the new scenarios and packages).

Key differences from existing `benchmarks.Rmd`:
- Package labels: vroom, vroom (CRAN), arrow, data.table (no readr, no read.delim)
- Operations include "materialize" between "read" and "print"
- No altrep column (simplify labels to just "package\nmanip_package")
- TSV paths use `path_package("vroom", "bench2", "<scenario>.tsv")`

**Step 2: Commit**

```bash
git add vignettes/benchmarks-libvroom.Rmd
git commit -m "feat: add libvroom benchmarks vignette (issue #73)"
```

---

### Task 7: Test the benchmark infrastructure locally

**Step 1: Generate test input data (small size)**

```bash
cd inst/bench2
Rscript mixed/input.R 10000 25 mixed/input.tsv
Rscript all_numeric/input.R 10000 25 all_numeric/input.tsv
Rscript all_character/input.R 10000 25 all_character/input.tsv
```

Expected: Three `input.tsv` files created, each ~few MB.

**Step 2: Run one benchmark to test the runner**

```bash
cd inst/bench2
./run-bench.R mixed/vroom-dplyr.R mixed/vroom-dplyr.tsv mixed/input.tsv
```

Expected: TSV output file with columns: exprs, process, real, size, rows, cols, max_memory. Should have 9 rows (one per operation).

**Step 3: Run one benchmark per package to verify all work**

```bash
./run-bench.R mixed/data.table-data.table.R mixed/data.table-data.table.tsv mixed/input.tsv
./run-bench.R mixed/arrow-dplyr.R mixed/arrow-dplyr.tsv mixed/input.tsv
```

Expected: Both produce TSV output without errors.

**Step 4: Test the CRAN vroom setup and benchmark**

```bash
Rscript setup.R
./run-bench.R mixed/vroom_cran-dplyr.R mixed/vroom_cran-dplyr.tsv mixed/input.tsv
```

Expected: CRAN vroom installed to `lib/`, benchmark runs successfully.

**Step 5: Test summarise-benchmarks.R**

```bash
Rscript summarise-benchmarks.R
```

Expected: `mixed.tsv`, `all_numeric.tsv`, `all_character.tsv` created in `inst/bench2/`.

**Step 6: Test vignette builds**

```bash
cd /path/to/vroom
Rscript -e 'rmarkdown::render("vignettes/benchmarks-libvroom.Rmd")'
```

Expected: HTML output generated without errors.

---

### Task 8: Run full benchmarks and commit results

**Step 1: Run full benchmarks with Make**

```bash
cd inst/bench2
make BENCH_ROWS=1000000 BENCH_COLS=25
```

Expected: All benchmarks run, summary TSV files generated.

**Step 2: Commit summary TSV files**

The summary TSV files (`mixed.tsv`, `all_numeric.tsv`, `all_character.tsv`, `session_info.tsv`) are committed to the repo so the vignette can read them during package build. Individual per-package TSV files in subdirectories are gitignored by `.Rbuildignore`.

```bash
git add inst/bench2/mixed.tsv inst/bench2/all_numeric.tsv inst/bench2/all_character.tsv inst/bench2/session_info.tsv
git commit -m "data: add benchmark results for libvroom comparison (issue #73)"
```

---

### Task 9: Final verification

**Step 1: Run `air format .`**

```bash
air format .
```

Expected: No R files modified (benchmark scripts are not package R code, but run formatter anyway).

**Step 2: Run full test suite**

```bash
Rscript -e 'devtools::test()'
```

Expected: All tests pass, no regressions.

**Step 3: Build and check vignette**

```bash
Rscript -e 'devtools::build_vignettes()'
```

Expected: Both vignettes build successfully.

**Step 4: Final commit if any formatting changes**

```bash
git add -A
git commit -m "style: format benchmark scripts (issue #73)"
```
