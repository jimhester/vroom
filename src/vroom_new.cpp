#include <cpp11.hpp>
#include <libvroom/encoding.h>
#include <libvroom/error.h>
#include <libvroom/format_parser.h>
#include <libvroom/vroom.h>

#include <limits>

#include "arrow_to_r.h"
#include "libvroom_helpers.h"
#include "vroom_arrow_chr.h"
#include "vroom_arrow_dbl.h"
#include "vroom_arrow_int.h"
#include "vroom_arrow_lgl.h"
#include "vroom_rle.h"

// Translate libvroom type coercion messages to R-friendly expected values.
// libvroom produces: "Cannot convert to FLOAT64 in column 'x'"
// R tests expect:    "a double"
std::string translate_expected(const std::string& msg) {
  const std::string prefix = "Cannot convert to ";
  auto pos = msg.find(prefix);
  if (pos == std::string::npos)
    return msg;
  auto type_start = pos + prefix.size();
  auto type_end = msg.find(" in column", type_start);
  if (type_end == std::string::npos)
    return msg;
  std::string type = msg.substr(type_start, type_end - type_start);

  if (type == "FLOAT64")
    return "a double";
  if (type == "INT32")
    return "an integer";
  if (type == "INT64")
    return "a big integer";
  if (type == "DATE")
    return "date in ISO8601";
  if (type == "TIMESTAMP")
    return "date in ISO8601";
  if (type == "TIME")
    return "time in ISO8601";
  if (type == "BOOL")
    return "1/0, T/F, TRUE/FALSE";
  return msg;
}

// Convert libvroom ParseErrors to an R data frame (tibble-compatible).
// Returns a list with vectors: row (integer), col (integer),
// expected (character), actual (character).
static cpp11::writable::list
errors_to_r_problems(const std::vector<libvroom::ParseError>& errors) {
  R_xlen_t n = static_cast<R_xlen_t>(errors.size());
  cpp11::writable::integers rows(n);
  cpp11::writable::integers cols(n);
  cpp11::writable::strings expected(n);
  cpp11::writable::strings actual(n);

  for (R_xlen_t i = 0; i < n; i++) {
    const auto& err = errors[static_cast<size_t>(i)];
    rows[i] = err.line > 0 ? static_cast<int>(err.line) : NA_INTEGER;
    cols[i] = err.column > 0 ? static_cast<int>(err.column) : NA_INTEGER;
    expected[i] = translate_expected(err.message);
    actual[i] = err.context;
  }

  cpp11::writable::list df(
      {cpp11::named_arg("row") = rows, cpp11::named_arg("col") = cols,
       cpp11::named_arg("expected") = expected,
       cpp11::named_arg("actual") = actual});

  df.attr("class") =
      cpp11::writable::strings({"tbl_df", "tbl", "data.frame"});
  df.attr("row.names") =
      cpp11::writable::integers({NA_INTEGER, -static_cast<int>(n)});

  return df;
}

// Version that includes a file column for multi-file error tracking
static cpp11::writable::list errors_to_r_problems_with_files(
    const std::vector<libvroom::ParseError>& errors,
    const std::vector<std::string>& file_paths) {
  R_xlen_t n = static_cast<R_xlen_t>(errors.size());
  cpp11::writable::integers rows(n);
  cpp11::writable::integers cols(n);
  cpp11::writable::strings expected(n);
  cpp11::writable::strings actual(n);
  cpp11::writable::strings files(n);

  for (R_xlen_t i = 0; i < n; i++) {
    const auto& err = errors[static_cast<size_t>(i)];
    rows[i] = err.line > 0 ? static_cast<int>(err.line) : NA_INTEGER;
    cols[i] = err.column > 0 ? static_cast<int>(err.column) : NA_INTEGER;
    expected[i] = translate_expected(err.message);
    actual[i] = err.context;
    files[i] = static_cast<size_t>(i) < file_paths.size()
                   ? file_paths[static_cast<size_t>(i)]
                   : "";
  }

  cpp11::writable::list df(
      {cpp11::named_arg("row") = rows, cpp11::named_arg("col") = cols,
       cpp11::named_arg("expected") = expected,
       cpp11::named_arg("actual") = actual,
       cpp11::named_arg("file") = files});

  df.attr("class") =
      cpp11::writable::strings({"tbl_df", "tbl", "data.frame"});
  df.attr("row.names") =
      cpp11::writable::integers({NA_INTEGER, -static_cast<int>(n)});

  return df;
}

[[cpp11::register]] cpp11::sexp vroom_libvroom_(
    SEXP input,
    const std::string& delim,
    char quote,
    bool has_header,
    int skip,
    const std::string& comment,
    bool skip_empty_rows,
    bool trim_ws,
    const std::string& na_values,
    int num_threads,
    bool strings_as_factors,
    bool use_altrep,
    const std::vector<int>& col_types,
    const cpp11::strings& col_type_names,
    const cpp11::strings& col_formats,
    int default_col_type,
    bool escape_backslash,
    const cpp11::strings& locale_mon_ab,
    const cpp11::strings& locale_mon,
    const cpp11::strings& locale_day_ab,
    const cpp11::strings& locale_am_pm,
    const std::string& locale_date_format,
    const std::string& locale_time_format,
    const std::string& locale_decimal_mark,
    const std::string& locale_tz,
    int guess_max) {

  libvroom::CsvOptions opts;
  opts.decimal_mark = locale_decimal_mark.empty() ? '.' : locale_decimal_mark[0];
  opts.escape_backslash = escape_backslash;
  opts.guess_integer = false; // vroom defaults to guessing doubles, not integers
  if (!delim.empty())
    opts.separator = delim;
  opts.quote = quote;
  opts.has_header = has_header;
  opts.skip_empty_rows = skip_empty_rows;
  opts.trim_ws = trim_ws;
  if (skip > 0)
    opts.skip = static_cast<size_t>(skip);
  if (!comment.empty())
    opts.comment = comment;
  opts.null_values = na_values;
  if (num_threads > 0)
    opts.num_threads = static_cast<size_t>(num_threads);

  // Skip full-file encoding detection (simdutf::validate_utf8 scans entire
  // file). R already handles encoding at the connection level.
  opts.encoding = libvroom::CharEncoding::UTF8;

  opts.error_mode = libvroom::ErrorMode::PERMISSIVE;

  if (guess_max > 0)
    opts.sample_rows = static_cast<size_t>(guess_max);
  else if (guess_max == 0)
    opts.sample_rows = 0;
  else if (guess_max < 0)
    opts.sample_rows = SIZE_MAX;

  libvroom::CsvReader reader(opts);

  open_input_source(reader, input);
  apply_schema_overrides(reader, col_types, col_type_names);

  // Build FormatLocale from R locale parameters
  libvroom::FormatLocale fmt_locale = libvroom::FormatLocale::english();
  if (locale_mon_ab.size() >= 12) {
    for (R_xlen_t i = 0; i < 12; ++i)
      fmt_locale.month_abbrev[static_cast<size_t>(i)] = std::string(locale_mon_ab[i]);
  }
  if (locale_mon.size() >= 12) {
    for (R_xlen_t i = 0; i < 12; ++i)
      fmt_locale.month_names[static_cast<size_t>(i)] = std::string(locale_mon[i]);
  }
  if (locale_day_ab.size() >= 7) {
    for (R_xlen_t i = 0; i < 7; ++i)
      fmt_locale.day_abbrev[static_cast<size_t>(i)] = std::string(locale_day_ab[i]);
  }
  if (locale_am_pm.size() >= 2) {
    fmt_locale.am = std::string(locale_am_pm[0]);
    fmt_locale.pm = std::string(locale_am_pm[1]);
  }
  if (!locale_date_format.empty()) {
    fmt_locale.date_format = locale_date_format;
  }
  if (!locale_time_format.empty()) {
    fmt_locale.time_format = locale_time_format;
  }

  reader.set_format_locale(fmt_locale);

  // Apply format strings from R col_types to the schema
  if (col_formats.size() > 0) {
    auto schema_copy = reader.schema();
    if (col_type_names.size() > 0) {
      // Named matching
      for (size_t i = 0; i < schema_copy.size(); ++i) {
        for (R_xlen_t j = 0; j < col_type_names.size(); ++j) {
          if (schema_copy[i].name == std::string(col_type_names[j])) {
            if (j < col_formats.size()) {
              schema_copy[i].format = std::string(col_formats[j]);
            }
            break;
          }
        }
      }
    } else {
      // Positional matching
      for (R_xlen_t j = 0; j < col_formats.size() &&
                            static_cast<size_t>(j) < schema_copy.size(); ++j) {
        schema_copy[static_cast<size_t>(j)].format = std::string(col_formats[j]);
      }
    }
    (void)reader.set_schema(schema_copy);
  }

  // Apply default column type to columns not explicitly typed
  if (default_col_type > 0) {
    auto schema_copy = reader.schema();
    for (size_t i = 0; i < schema_copy.size(); ++i) {
      bool has_explicit = false;
      if (!col_types.empty()) {
        if (col_type_names.size() > 0) {
          // Named: check if this column was in the named list
          for (R_xlen_t j = 0; j < col_type_names.size(); ++j) {
            if (schema_copy[i].name == std::string(col_type_names[j])) {
              has_explicit = true;
              break;
            }
          }
        } else {
          // Positional: columns within col_types range are explicit
          has_explicit = (i < col_types.size());
        }
      }
      if (!has_explicit) {
        schema_copy[i].type = static_cast<libvroom::DataType>(default_col_type);
      }
    }
    (void)reader.set_schema(schema_copy);
  }

  const auto& schema = reader.schema();

  // Start streaming: runs SIMD analysis synchronously, dispatches parse tasks
  auto stream_result = reader.start_streaming();
  if (!stream_result) {
    cpp11::stop("Failed to start streaming: %s", stream_result.error.c_str());
  }

  // Capture detected dialect (if auto-detected) for spec()$delim
  auto detected = reader.detected_dialect();

  auto attach_problems = [&reader, &detected](cpp11::sexp result) -> cpp11::sexp {
    const auto& errors = reader.errors();
    if (!errors.empty()) {
      Rf_setAttrib(result, Rf_install("problems"), errors_to_r_problems(errors));
    }
    // Attach detected delimiter so R can use it for spec()$delim
    if (detected.has_value()) {
      std::string det_delim(1, detected->dialect.delimiter);
      Rf_setAttrib(result, Rf_install("detected_delim"),
                   Rf_mkString(det_delim.c_str()));
    }
    return result;
  };

  size_t total_rows = reader.row_count();
  size_t ncols = schema.size();

  if (total_rows == 0) {
    auto result = empty_tibble_from_schema(schema);
    // Drain any remaining chunks
    while (reader.next_chunk()) {}
    return attach_problems(result);
  }

  // ALTREP path: stream chunks incrementally.
  // Pre-allocate R vectors for numerics, accumulate string builders for ALTREP.
  if (use_altrep && !strings_as_factors) {
    cpp11::writable::list result(ncols);
    cpp11::writable::strings names(ncols);

    // Pre-allocate numeric R vectors and string builder accumulators
    std::vector<SEXP> numeric_vecs(ncols, R_NilValue);
    std::vector<std::vector<std::shared_ptr<libvroom::ArrowStringColumnBuilder>>>
        string_accumulators(ncols);

    for (size_t i = 0; i < ncols; i++) {
      names[static_cast<R_xlen_t>(i)] = schema[i].name;
      switch (schema[i].type) {
      case libvroom::DataType::INT32: {
        cpp11::writable::integers v(total_rows);
        numeric_vecs[i] = v;
        result[static_cast<R_xlen_t>(i)] = v; // GC-protect
        break;
      }
      case libvroom::DataType::INT64:
      case libvroom::DataType::FLOAT64:
      case libvroom::DataType::DATE:
      case libvroom::DataType::TIMESTAMP:
      case libvroom::DataType::TIME: {
        cpp11::writable::doubles v(total_rows);
        numeric_vecs[i] = v;
        result[static_cast<R_xlen_t>(i)] = v; // GC-protect
        break;
      }
      case libvroom::DataType::BOOL: {
        cpp11::writable::logicals v(total_rows);
        numeric_vecs[i] = v;
        result[static_cast<R_xlen_t>(i)] = v; // GC-protect
        break;
      }
      default:
        // String columns: will accumulate builders for ALTREP
        break;
      }
    }

    // Stream chunks, copying numeric data at running offset
    size_t row_offset = 0;
    while (auto chunk = reader.next_chunk()) {
      auto& columns = chunk.value();
      if (columns.empty())
        continue;
      size_t chunk_rows = columns[0]->size();

      for (size_t i = 0; i < ncols; i++) {
        auto type = columns[i]->type();

        if (type == libvroom::DataType::STRING) {
          // Accumulate string column builder for later ALTREP wrapping
          string_accumulators[i].push_back(
              std::shared_ptr<libvroom::ArrowStringColumnBuilder>(
                  static_cast<libvroom::ArrowStringColumnBuilder*>(
                      columns[i].release())));

        } else if (type == libvroom::DataType::INT32) {
          auto& col = static_cast<libvroom::ArrowInt32ColumnBuilder&>(*columns[i]);
          int* dest = INTEGER(numeric_vecs[i]) + row_offset;
          const int32_t* src = col.values().data();
          if (!col.null_bitmap().has_nulls()) {
            std::memcpy(dest, src, chunk_rows * sizeof(int32_t));
          } else {
            const auto& nulls = col.null_bitmap();
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = nulls.is_valid(r) ? src[r] : NA_INTEGER;
            }
          }

        } else if (type == libvroom::DataType::INT64) {
          auto& col = static_cast<libvroom::ArrowInt64ColumnBuilder&>(*columns[i]);
          double* dest = REAL(numeric_vecs[i]) + row_offset;
          const int64_t* src = col.values().data();
          constexpr int64_t BIT64_NA = std::numeric_limits<int64_t>::min();
          std::memcpy(dest, src, chunk_rows * sizeof(int64_t));
          if (col.null_bitmap().has_nulls()) {
            const auto& nulls = col.null_bitmap();
            for (size_t r = 0; r < chunk_rows; r++) {
              if (!nulls.is_valid(r)) {
                std::memcpy(&dest[r], &BIT64_NA, sizeof(int64_t));
              }
            }
          }

        } else if (type == libvroom::DataType::FLOAT64) {
          auto& col = static_cast<libvroom::ArrowFloat64ColumnBuilder&>(*columns[i]);
          double* dest = REAL(numeric_vecs[i]) + row_offset;
          const double* src = col.values().data();
          if (!col.null_bitmap().has_nulls()) {
            std::memcpy(dest, src, chunk_rows * sizeof(double));
          } else {
            const auto& nulls = col.null_bitmap();
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = nulls.is_valid(r) ? src[r] : NA_REAL;
            }
          }

        } else if (type == libvroom::DataType::BOOL) {
          auto& col = static_cast<libvroom::ArrowBoolColumnBuilder&>(*columns[i]);
          int* dest = LOGICAL(numeric_vecs[i]) + row_offset;
          const uint8_t* src = col.values().data();
          if (!col.null_bitmap().has_nulls()) {
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = static_cast<int>(src[r]);
            }
          } else {
            const auto& nulls = col.null_bitmap();
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = nulls.is_valid(r) ? static_cast<int>(src[r]) : NA_LOGICAL;
            }
          }

        } else if (type == libvroom::DataType::DATE) {
          auto& col = static_cast<libvroom::ArrowDateColumnBuilder&>(*columns[i]);
          double* dest = REAL(numeric_vecs[i]) + row_offset;
          const int32_t* src = col.values().data();
          if (!col.null_bitmap().has_nulls()) {
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = static_cast<double>(src[r]);
            }
          } else {
            const auto& nulls = col.null_bitmap();
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = nulls.is_valid(r) ? static_cast<double>(src[r]) : NA_REAL;
            }
          }

        } else if (type == libvroom::DataType::TIMESTAMP) {
          auto& col = static_cast<libvroom::ArrowTimestampColumnBuilder&>(*columns[i]);
          double* dest = REAL(numeric_vecs[i]) + row_offset;
          const int64_t* src = col.values().data();
          if (!col.null_bitmap().has_nulls()) {
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = static_cast<double>(src[r]) / 1e6;
            }
          } else {
            const auto& nulls = col.null_bitmap();
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = nulls.is_valid(r) ? static_cast<double>(src[r]) / 1e6 : NA_REAL;
            }
          }

        } else if (type == libvroom::DataType::TIME) {
          auto& col = static_cast<libvroom::ArrowTimeColumnBuilder&>(*columns[i]);
          double* dest = REAL(numeric_vecs[i]) + row_offset;
          const int64_t* src = col.values().data();
          if (!col.null_bitmap().has_nulls()) {
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = static_cast<double>(src[r]) / 1e6;
            }
          } else {
            const auto& nulls = col.null_bitmap();
            for (size_t r = 0; r < chunk_rows; r++) {
              dest[r] = nulls.is_valid(r) ? static_cast<double>(src[r]) / 1e6 : NA_REAL;
            }
          }

        } else {
          // Unknown type: try as string (same accumulator path)
          auto* str_col = dynamic_cast<libvroom::ArrowStringColumnBuilder*>(
              columns[i].get());
          if (str_col) {
            (void)columns[i].release();
            string_accumulators[i].push_back(
                std::shared_ptr<libvroom::ArrowStringColumnBuilder>(str_col));
          }
        }
      }

      row_offset += chunk_rows;
    }

    // Set Date/Timestamp/Time class attributes on numeric vectors
    for (size_t i = 0; i < ncols; i++) {
      if (schema[i].type == libvroom::DataType::DATE) {
        Rf_setAttrib(numeric_vecs[i], R_ClassSymbol, Rf_mkString("Date"));
      } else if (schema[i].type == libvroom::DataType::TIMESTAMP) {
        cpp11::writable::strings cls({"POSIXct", "POSIXt"});
        Rf_setAttrib(numeric_vecs[i], R_ClassSymbol, cls);
        Rf_setAttrib(numeric_vecs[i], Rf_install("tzone"), Rf_mkString("UTC"));
      } else if (schema[i].type == libvroom::DataType::TIME) {
        cpp11::writable::strings cls({"hms", "difftime"});
        Rf_setAttrib(numeric_vecs[i], R_ClassSymbol, cls);
        Rf_setAttrib(numeric_vecs[i], Rf_install("units"), Rf_mkString("secs"));
      } else if (schema[i].type == libvroom::DataType::INT64) {
        Rf_setAttrib(numeric_vecs[i], R_ClassSymbol, Rf_mkString("integer64"));
      }
    }

    // Wrap string columns in multi-chunk ALTREP
    for (size_t i = 0; i < ncols; i++) {
      if (!string_accumulators[i].empty()) {
        result[static_cast<R_xlen_t>(i)] =
            vroom_arrow_chr::Make(std::move(string_accumulators[i]), total_rows);
      }
    }

    result.attr("names") = names;
    result.attr("class") =
        cpp11::writable::strings({"tbl_df", "tbl", "data.frame"});
    result.attr("row.names") =
        cpp11::writable::integers({NA_INTEGER, -static_cast<int>(total_rows)});
    return attach_problems(result);
  }

  // Non-ALTREP paths: collect all chunks, then use existing conversion.
  // This unifies factor and non-ALTREP paths on the streaming API.
  std::vector<std::vector<std::unique_ptr<libvroom::ArrowColumnBuilder>>> chunks;
  while (auto chunk = reader.next_chunk()) {
    chunks.push_back(std::move(chunk.value()));
  }

  if (chunks.empty()) {
    // Edge case: no data chunks despite non-zero row_count
    cpp11::writable::list result(ncols);
    cpp11::writable::strings names(ncols);
    for (size_t i = 0; i < ncols; i++) {
      result[static_cast<R_xlen_t>(i)] = Rf_allocVector(STRSXP, 0);
      names[static_cast<R_xlen_t>(i)] = schema[i].name;
    }
    result.attr("names") = names;
    result.attr("class") =
        cpp11::writable::strings({"tbl_df", "tbl", "data.frame"});
    result.attr("row.names") =
        cpp11::writable::integers({NA_INTEGER, 0});
    return attach_problems(result);
  }

  // Fast path: direct chunked copy for non-factor, all-numeric data.
  // columns_to_r_chunked wraps strings in ALTREP, so only safe when
  // there are no string columns (or when ALTREP is acceptable).
  if (!strings_as_factors) {
    bool has_string_cols = false;
    for (size_t i = 0; i < chunks[0].size(); i++) {
      if (chunks[0][i]->type() == libvroom::DataType::STRING) {
        has_string_cols = true;
        break;
      }
    }
    if (!has_string_cols) {
      return attach_problems(
          columns_to_r_chunked(chunks, schema, total_rows));
    }
  }

  // Merge path: needed for factors (dict building) and non-ALTREP strings
  std::vector<std::unique_ptr<libvroom::ArrowColumnBuilder>>& merged = chunks[0];
  for (size_t c = 1; c < chunks.size(); c++) {
    for (size_t col = 0; col < merged.size(); col++) {
      merged[col]->merge_from(*chunks[c][col]);
    }
  }

  return attach_problems(columns_to_r(merged, schema, total_rows, strings_as_factors,
                                      use_altrep));
}

// Multi-file entry point: reads multiple CSV files and returns a single R data
// frame with multi-chunk Altrep vectors spanning all files (zero-copy for all
// column types).
[[cpp11::register]] cpp11::sexp vroom_libvroom_multi_(
    const cpp11::strings& files,
    const std::string& delim,
    char quote,
    bool has_header,
    int skip,
    const std::string& comment,
    bool skip_empty_rows,
    bool trim_ws,
    const std::string& na_values,
    int num_threads,
    bool use_altrep,
    const std::vector<int>& col_types,
    const cpp11::strings& col_type_names,
    int default_col_type,
    bool escape_backslash,
    const std::string& id_col_name) {

  // Build shared CSV options
  libvroom::CsvOptions opts;
  opts.escape_backslash = escape_backslash;
  if (!delim.empty())
    opts.separator = delim[0];
  opts.quote = quote;
  opts.has_header = has_header;
  opts.skip_empty_rows = skip_empty_rows;
  opts.trim_ws = trim_ws;
  if (skip > 0)
    opts.skip = static_cast<size_t>(skip);
  if (!comment.empty())
    opts.comment = comment[0];
  if (!na_values.empty())
    opts.null_values = na_values;
  if (num_threads > 0)
    opts.num_threads = static_cast<size_t>(num_threads);
  opts.encoding = libvroom::CharEncoding::UTF8;
  opts.error_mode = libvroom::ErrorMode::PERMISSIVE;

  R_xlen_t n_files = files.size();
  if (n_files == 0) {
    cpp11::stop("No files provided");
  }

  // Phase 1: Open all files, set up schema, start streaming
  struct FileInfo {
    std::unique_ptr<libvroom::CsvReader> reader;
    std::string path;
    size_t row_count;
  };
  std::vector<FileInfo> file_infos;
  file_infos.reserve(static_cast<size_t>(n_files));

  std::vector<libvroom::ColumnSchema> master_schema;
  bool schema_established = false;
  size_t total_rows = 0;

  for (R_xlen_t fi = 0; fi < n_files; fi++) {
    std::string path = std::string(files[fi]);
    auto reader = std::make_unique<libvroom::CsvReader>(opts);

    auto open_result = reader->open(path);
    if (!open_result) {
      cpp11::stop("Failed to open file '%s': %s", path.c_str(),
                  open_result.error.c_str());
    }

    if (!schema_established) {
      // First file with data: apply schema overrides + default col types
      apply_schema_overrides(*reader, col_types, col_type_names);

      if (default_col_type > 0) {
        auto schema_copy = reader->schema();
        for (size_t i = 0; i < schema_copy.size(); ++i) {
          bool has_explicit = false;
          if (!col_types.empty()) {
            if (col_type_names.size() > 0) {
              for (R_xlen_t j = 0; j < col_type_names.size(); ++j) {
                if (schema_copy[i].name == std::string(col_type_names[j])) {
                  has_explicit = true;
                  break;
                }
              }
            } else {
              has_explicit = (i < col_types.size());
            }
          }
          if (!has_explicit) {
            schema_copy[i].type =
                static_cast<libvroom::DataType>(default_col_type);
          }
        }
        reader->set_schema(schema_copy);
      }
    } else {
      // Subsequent files: enforce consistent schema from first file with data
      reader->set_schema(master_schema);
    }

    auto stream_result = reader->start_streaming();
    if (!stream_result) {
      cpp11::stop("Failed to start streaming for '%s': %s", path.c_str(),
                  stream_result.error.c_str());
    }

    size_t row_count = reader->row_count();
    total_rows += row_count;

    // Establish master schema from first file with actual data rows
    // (files with only headers have unreliable type inference)
    if (!schema_established && row_count > 0) {
      master_schema = reader->schema();
      schema_established = true;
    }

    file_infos.push_back({std::move(reader), std::move(path), row_count});
  }

  // If no file had data, use schema from first file (for column names)
  if (!schema_established && !file_infos.empty()) {
    master_schema = file_infos[0].reader->schema();
  }

  const auto& schema = master_schema;
  size_t ncols = schema.size();

  // Phase 2: Stream chunks from all files into per-column accumulators
  // We use typed accumulators so we can create multi-chunk Altrep for all types.
  std::vector<std::vector<std::shared_ptr<libvroom::ArrowStringColumnBuilder>>>
      string_accums(ncols);
  std::vector<std::vector<std::shared_ptr<libvroom::ArrowInt32ColumnBuilder>>>
      int_accums(ncols);
  std::vector<std::vector<std::shared_ptr<libvroom::ArrowColumnBuilder>>>
      dbl_accums(ncols); // For FLOAT64, INT64, DATE, TIMESTAMP
  std::vector<std::vector<std::shared_ptr<libvroom::ArrowBoolColumnBuilder>>>
      lgl_accums(ncols);

  // Track file paths and row counts for id column
  std::vector<std::string> id_file_paths;
  std::vector<size_t> id_row_counts;

  // Collect all errors with their source file paths
  std::vector<libvroom::ParseError> all_errors;
  std::vector<std::string> all_error_files;

  for (auto& fi : file_infos) {
    // Track for id column even if file has 0 rows (skip accumulation below)
    if (!id_col_name.empty() && fi.row_count > 0) {
      id_file_paths.push_back(fi.path);
      id_row_counts.push_back(fi.row_count);
    }

    while (auto chunk = fi.reader->next_chunk()) {
      auto& columns = chunk.value();
      if (columns.empty())
        continue;

      for (size_t i = 0; i < ncols && i < columns.size(); i++) {
        auto type = columns[i]->type();

        switch (type) {
        case libvroom::DataType::STRING: {
          string_accums[i].push_back(
              std::shared_ptr<libvroom::ArrowStringColumnBuilder>(
                  static_cast<libvroom::ArrowStringColumnBuilder*>(
                      columns[i].release())));
          break;
        }
        case libvroom::DataType::INT32: {
          int_accums[i].push_back(
              std::shared_ptr<libvroom::ArrowInt32ColumnBuilder>(
                  static_cast<libvroom::ArrowInt32ColumnBuilder*>(
                      columns[i].release())));
          break;
        }
        case libvroom::DataType::FLOAT64:
        case libvroom::DataType::INT64:
        case libvroom::DataType::DATE:
        case libvroom::DataType::TIMESTAMP: {
          dbl_accums[i].push_back(std::shared_ptr<libvroom::ArrowColumnBuilder>(
              columns[i].release()));
          break;
        }
        case libvroom::DataType::BOOL: {
          lgl_accums[i].push_back(
              std::shared_ptr<libvroom::ArrowBoolColumnBuilder>(
                  static_cast<libvroom::ArrowBoolColumnBuilder*>(
                      columns[i].release())));
          break;
        }
        default: {
          // Unknown type: try as string
          auto* str_col = dynamic_cast<libvroom::ArrowStringColumnBuilder*>(
              columns[i].get());
          if (str_col) {
            (void)columns[i].release();
            string_accums[i].push_back(
                std::shared_ptr<libvroom::ArrowStringColumnBuilder>(str_col));
          }
          break;
        }
        }
      }
    }

    // Collect errors from this file, tagging each with the file path
    const auto& file_errors = fi.reader->errors();
    if (!file_errors.empty()) {
      all_errors.insert(all_errors.end(), file_errors.begin(),
                        file_errors.end());
      all_error_files.insert(all_error_files.end(), file_errors.size(),
                             fi.path);
    }
  }

  // Phase 3: Build multi-chunk Altrep result
  bool has_id = !id_col_name.empty();
  size_t result_ncols = ncols + (has_id ? 1 : 0);

  // Handle empty result
  if (total_rows == 0) {
    cpp11::writable::list result(static_cast<R_xlen_t>(result_ncols));
    cpp11::writable::strings names(static_cast<R_xlen_t>(result_ncols));
    R_xlen_t col_offset = 0;

    if (has_id) {
      result[col_offset] = Rf_allocVector(STRSXP, 0);
      names[col_offset] = id_col_name;
      col_offset++;
    }
    for (size_t i = 0; i < ncols; i++) {
      result[col_offset + static_cast<R_xlen_t>(i)] =
          Rf_allocVector(STRSXP, 0);
      names[col_offset + static_cast<R_xlen_t>(i)] = schema[i].name;
    }
    result.attr("names") = names;
    result.attr("class") =
        cpp11::writable::strings({"tbl_df", "tbl", "data.frame"});
    result.attr("row.names") =
        cpp11::writable::integers({NA_INTEGER, 0});

    if (!all_errors.empty()) {
      Rf_setAttrib(result, Rf_install("problems"),
                   errors_to_r_problems_with_files(all_errors, all_error_files));
    }
    return result;
  }

  cpp11::writable::list result(static_cast<R_xlen_t>(result_ncols));
  cpp11::writable::strings names(static_cast<R_xlen_t>(result_ncols));
  R_xlen_t col_offset = 0;

  // Build id column if requested
  if (has_id) {
    cpp11::writable::integers rle(static_cast<R_xlen_t>(id_file_paths.size()));
    cpp11::writable::strings rle_names(
        static_cast<R_xlen_t>(id_file_paths.size()));
    for (R_xlen_t i = 0; i < static_cast<R_xlen_t>(id_file_paths.size());
         i++) {
      rle[i] = static_cast<int>(id_row_counts[static_cast<size_t>(i)]);
      rle_names[i] = id_file_paths[static_cast<size_t>(i)];
    }
    rle.names() = rle_names;

    result[col_offset] = vroom_rle::Make(rle);
    names[col_offset] = id_col_name;
    col_offset++;
  }

  // Build data columns using multi-chunk Altrep
  for (size_t i = 0; i < ncols; i++) {
    R_xlen_t ri = col_offset + static_cast<R_xlen_t>(i);
    names[ri] = schema[i].name;

    if (use_altrep) {
      switch (schema[i].type) {
      case libvroom::DataType::STRING: {
        if (!string_accums[i].empty()) {
          result[ri] =
              vroom_arrow_chr::Make(std::move(string_accums[i]), total_rows);
        } else {
          result[ri] = Rf_allocVector(STRSXP, static_cast<R_xlen_t>(total_rows));
        }
        break;
      }
      case libvroom::DataType::INT32: {
        if (!int_accums[i].empty()) {
          result[ri] =
              vroom_arrow_int::Make(std::move(int_accums[i]), total_rows);
        } else {
          result[ri] = Rf_allocVector(INTSXP, static_cast<R_xlen_t>(total_rows));
        }
        break;
      }
      case libvroom::DataType::FLOAT64:
      case libvroom::DataType::INT64: {
        if (!dbl_accums[i].empty()) {
          result[ri] = vroom_arrow_dbl::Make(std::move(dbl_accums[i]),
                                             total_rows, schema[i].type);
        } else {
          result[ri] = Rf_allocVector(REALSXP, static_cast<R_xlen_t>(total_rows));
        }
        break;
      }
      case libvroom::DataType::DATE: {
        SEXP vec;
        if (!dbl_accums[i].empty()) {
          vec = vroom_arrow_dbl::Make(std::move(dbl_accums[i]), total_rows,
                                     libvroom::DataType::DATE);
        } else {
          vec = Rf_allocVector(REALSXP, static_cast<R_xlen_t>(total_rows));
        }
        Rf_setAttrib(vec, R_ClassSymbol, Rf_mkString("Date"));
        result[ri] = vec;
        break;
      }
      case libvroom::DataType::TIMESTAMP: {
        SEXP vec;
        if (!dbl_accums[i].empty()) {
          vec = vroom_arrow_dbl::Make(std::move(dbl_accums[i]), total_rows,
                                     libvroom::DataType::TIMESTAMP);
        } else {
          vec = Rf_allocVector(REALSXP, static_cast<R_xlen_t>(total_rows));
        }
        cpp11::writable::strings cls({"POSIXct", "POSIXt"});
        Rf_setAttrib(vec, R_ClassSymbol, cls);
        Rf_setAttrib(vec, Rf_install("tzone"), Rf_mkString("UTC"));
        result[ri] = vec;
        break;
      }
      case libvroom::DataType::BOOL: {
        if (!lgl_accums[i].empty()) {
          result[ri] =
              vroom_arrow_lgl::Make(std::move(lgl_accums[i]), total_rows);
        } else {
          result[ri] = Rf_allocVector(LGLSXP, static_cast<R_xlen_t>(total_rows));
        }
        break;
      }
      default: {
        // Fallback: treat as string
        if (!string_accums[i].empty()) {
          result[ri] =
              vroom_arrow_chr::Make(std::move(string_accums[i]), total_rows);
        } else {
          result[ri] = Rf_allocVector(STRSXP, static_cast<R_xlen_t>(total_rows));
        }
        break;
      }
      }
    } else {
      // Non-Altrep path: not supported for multi-file, fall back to string
      // (callers should always pass use_altrep=TRUE for multi-file)
      if (!string_accums[i].empty()) {
        result[ri] =
            vroom_arrow_chr::Make(std::move(string_accums[i]), total_rows);
      } else {
        result[ri] = Rf_allocVector(STRSXP, static_cast<R_xlen_t>(total_rows));
      }
    }
  }

  result.attr("names") = names;
  result.attr("class") =
      cpp11::writable::strings({"tbl_df", "tbl", "data.frame"});
  result.attr("row.names") = cpp11::writable::integers(
      {NA_INTEGER, -static_cast<int>(total_rows)});

  if (!all_errors.empty()) {
    Rf_setAttrib(result, Rf_install("problems"),
                 errors_to_r_problems_with_files(all_errors, all_error_files));
  }
  return result;
}
