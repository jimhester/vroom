# Read a delimited file into an Arrow Table

Uses the libvroom SIMD parser to read a CSV file and return the result
as an Arrow Table via zero-copy Arrow C Data Interface export. This
avoids R's global string pool entirely, making it particularly efficient
for string-heavy files.

## Usage

``` r
vroom_arrow(
  file,
  delim = NULL,
  quote = "\"",
  col_names = TRUE,
  comment = "",
  skip_empty_rows = TRUE,
  na = c("", "NA"),
  num_threads = vroom_threads()
)
```

## Arguments

- file:

  Path to a delimited file.

- delim:

  Single character used to separate fields. If `NULL`, the delimiter is
  auto-detected.

- quote:

  Single character used to quote strings.

- col_names:

  If `TRUE`, the first row is used as column names.

- comment:

  A string used to identify comments.

- skip_empty_rows:

  Should blank rows be ignored?

- na:

  Character vector of strings to interpret as missing values.

- num_threads:

  Number of threads to use for parsing.

## Value

An Arrow Table.
