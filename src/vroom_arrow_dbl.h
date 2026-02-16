#pragma once

#include <cpp11/R.hpp>

#include <libvroom/arrow_buffer.h>
#include <libvroom/arrow_column_builder.h>
#include <libvroom/types.h>

#include "altrep.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

// Arrow-backed ALTREP double vector with multi-chunk support.
// Wraps one or more ArrowColumnBuilder chunks that produce R doubles.
// Supports FOUR source types that all map to R's REALSXP:
//   FLOAT64:   ArrowFloat64ColumnBuilder — double -> double (direct)
//   INT64:     ArrowInt64ColumnBuilder   — int64_t -> double (cast)
//   DATE:      ArrowDateColumnBuilder    — int32_t -> double (cast, days since epoch)
//   TIMESTAMP: ArrowTimestampColumnBuilder — int64_t -> double (cast + divide by 1e6)
//
// The Make() factory does NOT set Date/POSIXct class attributes —
// that's done by the caller after creating the vector.

struct ArrowDblChunk {
  std::shared_ptr<libvroom::ArrowColumnBuilder> builder; // keeps data alive
  const void* raw_data;              // pointer to values buffer
  const libvroom::NullBitmap* nulls; // pointer to null bitmap
  size_t size;                       // number of rows in chunk
};

struct ArrowDblInfo {
  libvroom::DataType source_type; // FLOAT64, INT64, DATE, or TIMESTAMP
  std::vector<ArrowDblChunk> chunks;
  // Prefix sums: chunk_offsets[i] = total rows in chunks[0..i-1]
  // chunk_offsets[0] = 0, chunk_offsets[n] = total_rows
  std::vector<size_t> chunk_offsets;
  size_t nrows;
  bool has_nulls;
};

struct vroom_arrow_dbl {
  static R_altrep_class_t class_t;

  static void Finalize(SEXP ptr) {
    auto* info = static_cast<ArrowDblInfo*>(R_ExternalPtrAddr(ptr));
    if (info) {
      delete info;
      R_ClearExternalPtr(ptr);
    }
  }

  // Extract raw data pointer from a builder based on source type.
  static const void*
  extract_raw_data(libvroom::ArrowColumnBuilder* b,
                   libvroom::DataType source_type) {
    using namespace libvroom;
    switch (source_type) {
    case DataType::FLOAT64:
      return static_cast<ArrowFloat64ColumnBuilder*>(b)->values().data();
    case DataType::INT64:
      return static_cast<ArrowInt64ColumnBuilder*>(b)->values().data();
    case DataType::DATE:
      return static_cast<ArrowDateColumnBuilder*>(b)->values().data();
    case DataType::TIMESTAMP:
      return static_cast<ArrowTimestampColumnBuilder*>(b)->values().data();
    default:
      return nullptr; // should never happen
    }
  }

  // Create ALTREP vector wrapping multiple chunks (zero-copy).
  // source_type indicates how to interpret the raw data in all chunks.
  static SEXP
  Make(std::vector<std::shared_ptr<libvroom::ArrowColumnBuilder>> builders,
       size_t total_rows, libvroom::DataType source_type) {
    auto* info = new ArrowDblInfo{};
    info->source_type = source_type;
    info->nrows = total_rows;
    info->has_nulls = false;

    // Build chunks with cached raw pointers + prefix sum offsets
    info->chunks.reserve(builders.size());
    info->chunk_offsets.reserve(builders.size() + 1);
    info->chunk_offsets.push_back(0);
    size_t offset = 0;
    for (auto& b : builders) {
      ArrowDblChunk chunk;
      chunk.raw_data = extract_raw_data(b.get(), source_type);
      chunk.nulls = &b->null_bitmap();
      chunk.size = b->size();
      chunk.builder = b; // keep alive
      if (b->null_bitmap().has_nulls()) {
        info->has_nulls = true;
      }
      offset += chunk.size;
      info->chunk_offsets.push_back(offset);
      info->chunks.push_back(std::move(chunk));
    }

    SEXP ptr = PROTECT(R_MakeExternalPtr(info, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, Finalize, FALSE);

    SEXP res = R_new_altrep(class_t, ptr, R_NilValue);
    MARK_NOT_MUTABLE(res);

    UNPROTECT(1);
    return res;
  }

  static inline ArrowDblInfo& Info(SEXP vec) {
    return *static_cast<ArrowDblInfo*>(
        R_ExternalPtrAddr(R_altrep_data1(vec)));
  }

  // Find chunk index for global row i.
  // Uses upper_bound on prefix sums. With 4-16 chunks, this is 2-4 comparisons.
  static inline void
  resolve_chunk(const ArrowDblInfo& info, size_t i, size_t& chunk_idx,
                size_t& local_idx) {
    // upper_bound finds first offset > i, subtract 1 gives the chunk
    auto it = std::upper_bound(info.chunk_offsets.begin(),
                               info.chunk_offsets.end(), i);
    chunk_idx = static_cast<size_t>(it - info.chunk_offsets.begin()) - 1;
    local_idx = i - info.chunk_offsets[chunk_idx];
  }

  // Convert a raw value to double based on source type.
  static inline double
  convert_value(const ArrowDblChunk& chunk, size_t local_idx,
                libvroom::DataType source_type) {
    switch (source_type) {
    case libvroom::DataType::FLOAT64:
      return static_cast<const double*>(chunk.raw_data)[local_idx];
    case libvroom::DataType::INT64:
      return static_cast<double>(
          static_cast<const int64_t*>(chunk.raw_data)[local_idx]);
    case libvroom::DataType::DATE:
      return static_cast<double>(
          static_cast<const int32_t*>(chunk.raw_data)[local_idx]);
    case libvroom::DataType::TIMESTAMP:
      return static_cast<double>(
                 static_cast<const int64_t*>(chunk.raw_data)[local_idx]) /
             1e6;
    default:
      return NA_REAL; // should never happen
    }
  }

  // ALTREP methods

  static R_xlen_t Length(SEXP vec) {
    SEXP data2 = R_altrep_data2(vec);
    if (data2 != R_NilValue) {
      return Rf_xlength(data2);
    }
    return static_cast<R_xlen_t>(Info(vec).nrows);
  }

  static Rboolean
  Inspect(SEXP x, int, int, int, void (*)(SEXP, int, int, int)) {
    auto& info = Info(x);
    Rprintf(
        "vroom_arrow_dbl (len=%d, chunks=%d, type=%s, materialized=%s)\n",
        (int)Length(x), (int)info.chunks.size(),
        libvroom::type_name(info.source_type),
        R_altrep_data2(x) != R_NilValue ? "T" : "F");
    return TRUE;
  }

  // ALTREAL Elt: read from correct Arrow chunk, converting based on source type
  static double real_Elt(SEXP vec, R_xlen_t i) {
    SEXP data2 = R_altrep_data2(vec);
    if (data2 != R_NilValue) {
      return REAL_ELT(data2, i);
    }

    auto& info = Info(vec);
    size_t chunk_idx, local_idx;
    resolve_chunk(info, static_cast<size_t>(i), chunk_idx, local_idx);

    auto& chunk = info.chunks[chunk_idx];
    if (info.has_nulls && !chunk.nulls->is_valid(local_idx)) {
      return NA_REAL;
    }
    return convert_value(chunk, local_idx, info.source_type);
  }

  // Materialize: build full REALSXP from all Arrow chunks
  static SEXP Materialize(SEXP vec) {
    SEXP data2 = R_altrep_data2(vec);
    if (data2 != R_NilValue) {
      return data2;
    }

    auto& info = Info(vec);
    R_xlen_t n = static_cast<R_xlen_t>(info.nrows);

    SEXP result = PROTECT(Rf_allocVector(REALSXP, n));
    double* dest = REAL(result);
    R_xlen_t dest_idx = 0;
    size_t rows_remaining = info.nrows;

    for (size_t c = 0; c < info.chunks.size() && rows_remaining > 0; c++) {
      auto& chunk = info.chunks[c];
      size_t chunk_size = std::min(chunk.size, rows_remaining);
      bool chunk_has_nulls = chunk.nulls->has_nulls();

      // For FLOAT64, we can bulk-copy since double -> double is identity
      if (info.source_type == libvroom::DataType::FLOAT64) {
        std::memcpy(
            dest + dest_idx, chunk.raw_data, chunk_size * sizeof(double));
      } else {
        // Per-element conversion for INT64, DATE, TIMESTAMP
        for (size_t j = 0; j < chunk_size; j++) {
          dest[dest_idx + static_cast<R_xlen_t>(j)] =
              convert_value(chunk, j, info.source_type);
        }
      }

      if (chunk_has_nulls) {
        // Patch NA values: scan the null bitmap and overwrite with NA_REAL
        for (size_t j = 0; j < chunk_size; j++) {
          if (!chunk.nulls->is_valid(j)) {
            dest[dest_idx + static_cast<R_xlen_t>(j)] = NA_REAL;
          }
        }
      }

      dest_idx += static_cast<R_xlen_t>(chunk_size);
      rows_remaining -= chunk_size;
    }

    // Copy attributes (class, tzone, etc.) from the ALTREP vector to the
    // materialized result so Date/POSIXct class attributes are preserved.
    SEXP attribs = ATTRIB(vec);
    if (attribs != R_NilValue) {
      SET_ATTRIB(result, Rf_duplicate(attribs));
    }

    R_set_altrep_data2(vec, result);

    // Release the Arrow chunk data now that we have the full REALSXP
    info.chunks.clear();

    UNPROTECT(1);
    return result;
  }

  static void* Dataptr(SEXP vec, Rboolean) {
    return DATAPTR_RW(Materialize(vec));
  }

  static const void* Dataptr_or_null(SEXP vec) {
    SEXP data2 = R_altrep_data2(vec);
    if (data2 == R_NilValue)
      return nullptr;
    return DATAPTR_RO(data2);
  }

  static void Init(DllInfo* dll) {
    class_t = R_make_altreal_class("vroom_arrow_dbl", "vroom", dll);

    // altrep
    R_set_altrep_Length_method(class_t, Length);
    R_set_altrep_Inspect_method(class_t, Inspect);

    // altvec
    R_set_altvec_Dataptr_method(class_t, Dataptr);
    R_set_altvec_Dataptr_or_null_method(class_t, Dataptr_or_null);

    // altreal
    R_set_altreal_Elt_method(class_t, real_Elt);
  }
};

[[cpp11::init]] void init_vroom_arrow_dbl(DllInfo* dll);
