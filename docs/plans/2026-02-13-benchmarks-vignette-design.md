# Benchmarks Vignette Design

**Issue:** #73 - New benchmarks vignette to demonstrate improved performance
**Date:** 2026-02-13

## Summary

Create a new benchmarks vignette (`vignettes/benchmarks-libvroom.Rmd`) that demonstrates the performance of vroom's new libvroom SIMD-based backend, compared against the CRAN version of vroom, arrow, and data.table.

## Architecture

Benchmark execution is separated from visualization:

- **`inst/bench2/`** contains benchmark scripts, a Makefile, and a runner. Each package runs in its own Rscript process via `run-bench.R`, which uses `bench::workout_expressions()` to capture timing. Results are written to TSV files.
- **`vignettes/benchmarks-libvroom.Rmd`** reads the summary TSV files and produces plots/tables. Pure visualization, no benchmarking.

### Directory Layout

```
inst/bench2/
├── GNUmakefile
├── run-bench.R               # Runner: executes .R file, captures timing
├── summarise-benchmarks.R    # Aggregates per-package .tsv into summary .tsv
├── session_info.R            # Captures package versions
├── setup.R                   # Installs CRAN vroom into lib/
├── lib/                      # CRAN vroom installed here (gitignored)
├── mixed/
│   ├── input.R               # Generates synthetic mixed-type data
│   ├── vroom-dplyr.R
│   ├── vroom_cran-dplyr.R
│   ├── arrow-dplyr.R
│   └── data.table-data.table.R
├── all_numeric/
│   ├── input.R
│   ├── vroom-dplyr.R
│   ├── vroom_cran-dplyr.R
│   ├── arrow-dplyr.R
│   └── data.table-data.table.R
└── all_character/
    ├── input.R
    ├── vroom-dplyr.R
    ├── vroom_cran-dplyr.R
    ├── arrow-dplyr.R
    └── data.table-data.table.R
```

## Packages Compared

| Package | Reader | Manipulator | Notes |
|---------|--------|-------------|-------|
| vroom | `vroom::vroom()` (libvroom) | dplyr | Current dev version |
| vroom_cran | `vroom::vroom()` (CRAN) | dplyr | Installed in `lib/` via `setup.R` |
| arrow | `arrow::read_csv_arrow()` | dplyr | |
| data.table | `data.table::fread()` | data.table | |

### CRAN vroom Isolation

`setup.R` installs the CRAN version of vroom into `inst/bench2/lib/`. The `vroom_cran-*.R` scripts prepend this path:

```r
.libPaths(c("lib", .libPaths()))
library(vroom)
```

## Benchmark Scenarios

Three scenarios with synthetic data (~1M rows x 25 cols):

1. **Mixed-type** (`mixed/`): Strings + integers + doubles. Typical real-world CSV.
2. **All-numeric** (`all_numeric/`): Pure doubles. Traditionally vroom's weakest case; libvroom's SIMD parsing changes this.
3. **All-character** (`all_character/`): Pure strings. Best case for Altrep lazy reading.

## Operations Measured

Each benchmark script runs these operations in sequence, timed individually by `bench::workout_expressions()`:

1. **setup** - Library loading
2. **read** - Read the file
3. **materialize** - `vroom::vroom_materialize(x, replace = FALSE)` for vroom; no-op for arrow/data.table (already materialized on read)
4. **print** - Print the tibble/data.frame
5. **head** / **tail** - First/last rows
6. **sample** - Random 100 rows
7. **filter** - Filter on a column value
8. **aggregate** - Grouped mean

## Makefile

```makefile
BENCH_ROWS := 1000000
BENCH_COLS := 25

# lib/ target installs CRAN vroom
lib/vroom: setup.R
    Rscript setup.R

# input.R generates test data
%/input.tsv: %/input.R
    Rscript $< $(BENCH_ROWS) $(BENCH_COLS) $@

# Per-package benchmarks
%.tsv: %.R
    ./run-bench.R $< $@ $(@D)/input.tsv

# Aggregate summaries
all: mixed.tsv all_numeric.tsv all_character.tsv session_info.tsv
```

## Vignette Structure

`vignettes/benchmarks-libvroom.Rmd`:

1. **Introduction** - libvroom SIMD backend, what we're comparing
2. **Setup chunk** (hidden) - Helper functions: `read_benchmark()`, `plot_benchmark()`, `make_table()`
3. **Mixed-type data** - Plot + table + prose
4. **All-numeric data** - Plot + table + prose
5. **All-character data** - Plot + table + prose
6. **Session info** - Package versions

### Visualization Style

- Horizontal stacked bar charts: operation breakdown (read, materialize, print, head, tail, sample, filter, aggregate)
- Side panel: max memory usage
- `scale_fill_brewer(palette = "Set2")` + patchwork layout
- kable HTML tables with pretty-printed times

## Build Ignore

Add to `.Rbuildignore`:
```
^inst/bench2/.+/.*\.tsv$
^inst/bench2/lib$
```
