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
