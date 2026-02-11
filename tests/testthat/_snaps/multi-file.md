# vroom errors if numbers of columns are inconsistent

    Code
      vroom::vroom(files, col_types = list())
    Condition
      Error in `vroom::vroom()`:
      ! Files have different number of columns.
      i First file has 2 columns.
      i File multi-file/baz has 3 columns.

# vroom errors if column names are inconsistent

    Code
      vroom::vroom(files, col_types = list())
    Condition
      Error in `vroom::vroom()`:
      ! Files have different column names.
      i First file has columns: "A" and "B".
      i File multi-file/bar has columns: "C" and "D".

