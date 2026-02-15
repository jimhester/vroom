#pragma once

#include <cpp11/R.hpp>

#include <libvroom/arrow_buffer.h>
#include <libvroom/arrow_column_builder.h>

#include "altrep.h"

#include <algorithm>
#include <memory>
#include <vector>

// Arrow-backed ALTREP logical vector with multi-chunk support.
// Wraps one or more ArrowBoolColumnBuilder chunks directly.
// Creation is near-instant (just store pointers + compute offsets).
// Element access reads uint8_t from the correct chunk on demand and casts to int.
// Materialization builds the full LGLSXP when DATAPTR is requested.

struct ArrowLglInfo {
  // One ArrowBoolColumnBuilder per parsed chunk
  std::vector<std::shared_ptr<libvroom::ArrowBoolColumnBuilder>> chunks;
  // Prefix sums: chunk_offsets[i] = total rows in chunks[0..i-1]
  // chunk_offsets[0] = 0, chunk_offsets[n] = total_rows
  std::vector<size_t> chunk_offsets;
  size_t nrows;
  bool has_nulls;
};

struct vroom_arrow_lgl {
  static R_altrep_class_t class_t;

  static void Finalize(SEXP ptr) {
    auto* info = static_cast<ArrowLglInfo*>(R_ExternalPtrAddr(ptr));
    if (info) {
      delete info;
      R_ClearExternalPtr(ptr);
    }
  }

  // Create ALTREP vector wrapping a single ArrowBoolColumnBuilder.
  static SEXP Make(std::shared_ptr<libvroom::ArrowBoolColumnBuilder> col,
                   size_t nrows) {
    auto* info = new ArrowLglInfo{};
    info->nrows = nrows;
    info->chunk_offsets.push_back(0);
    info->chunk_offsets.push_back(nrows);
    info->has_nulls = col->null_bitmap().has_nulls();
    info->chunks.push_back(std::move(col));

    SEXP ptr = PROTECT(R_MakeExternalPtr(info, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, Finalize, FALSE);

    SEXP res = R_new_altrep(class_t, ptr, R_NilValue);
    MARK_NOT_MUTABLE(res);

    UNPROTECT(1);
    return res;
  }

  // Create ALTREP vector wrapping multiple chunks (zero-copy).
  static SEXP
  Make(std::vector<std::shared_ptr<libvroom::ArrowBoolColumnBuilder>> chunks,
       size_t total_rows) {
    auto* info = new ArrowLglInfo{};
    info->nrows = total_rows;
    info->has_nulls = false;

    // Build prefix sum offsets
    info->chunk_offsets.reserve(chunks.size() + 1);
    info->chunk_offsets.push_back(0);
    size_t offset = 0;
    for (auto& chunk : chunks) {
      offset += chunk->size();
      info->chunk_offsets.push_back(offset);
      if (chunk->null_bitmap().has_nulls()) {
        info->has_nulls = true;
      }
    }
    info->chunks = std::move(chunks);

    SEXP ptr = PROTECT(R_MakeExternalPtr(info, R_NilValue, R_NilValue));
    R_RegisterCFinalizerEx(ptr, Finalize, FALSE);

    SEXP res = R_new_altrep(class_t, ptr, R_NilValue);
    MARK_NOT_MUTABLE(res);

    UNPROTECT(1);
    return res;
  }

  static inline ArrowLglInfo& Info(SEXP vec) {
    return *static_cast<ArrowLglInfo*>(
        R_ExternalPtrAddr(R_altrep_data1(vec)));
  }

  // Find chunk index for global row i.
  // Uses upper_bound on prefix sums. With 4-16 chunks, this is 2-4 comparisons.
  static inline void
  resolve_chunk(const ArrowLglInfo& info, size_t i, size_t& chunk_idx,
                size_t& local_idx) {
    // upper_bound finds first offset > i, subtract 1 gives the chunk
    auto it = std::upper_bound(info.chunk_offsets.begin(),
                               info.chunk_offsets.end(), i);
    chunk_idx = static_cast<size_t>(it - info.chunk_offsets.begin()) - 1;
    local_idx = i - info.chunk_offsets[chunk_idx];
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
        "vroom_arrow_lgl (len=%d, chunks=%d, materialized=%s)\n",
        (int)Length(x), (int)info.chunks.size(),
        R_altrep_data2(x) != R_NilValue ? "T" : "F");
    return TRUE;
  }

  // ALTLOGICAL Elt: read from correct Arrow chunk's NumericBuffer<uint8_t>
  // Returns int (not Rboolean): 1=TRUE, 0=FALSE, NA_LOGICAL=NA
  static int logical_Elt(SEXP vec, R_xlen_t i) {
    SEXP data2 = R_altrep_data2(vec);
    if (data2 != R_NilValue) {
      return LOGICAL_ELT(data2, i);
    }

    auto& info = Info(vec);
    size_t chunk_idx, local_idx;
    resolve_chunk(info, static_cast<size_t>(i), chunk_idx, local_idx);

    auto& chunk = *info.chunks[chunk_idx];
    if (info.has_nulls && !chunk.null_bitmap().is_valid(local_idx)) {
      return NA_LOGICAL;
    }
    // Arrow bool is uint8_t (0 or 1), cast to int for R logical
    return static_cast<int>(chunk.values().get(local_idx));
  }

  // Materialize: build full LGLSXP from all Arrow chunks
  static SEXP Materialize(SEXP vec) {
    SEXP data2 = R_altrep_data2(vec);
    if (data2 != R_NilValue) {
      return data2;
    }

    auto& info = Info(vec);
    R_xlen_t n = static_cast<R_xlen_t>(info.nrows);

    SEXP result = PROTECT(Rf_allocVector(LGLSXP, n));
    int* dest = LOGICAL(result);
    R_xlen_t dest_idx = 0;
    size_t rows_remaining = info.nrows;

    for (size_t c = 0; c < info.chunks.size() && rows_remaining > 0; c++) {
      auto& chunk = *info.chunks[c];
      const libvroom::NumericBuffer<uint8_t>& buf = chunk.values();
      const libvroom::NullBitmap& nulls = chunk.null_bitmap();
      size_t chunk_size = std::min(chunk.size(), rows_remaining);
      bool chunk_has_nulls = nulls.has_nulls();

      // Cannot use memcpy: uint8_t != int, need per-element cast
      const uint8_t* src = buf.data();
      for (size_t j = 0; j < chunk_size; j++) {
        if (chunk_has_nulls && !nulls.is_valid(j)) {
          dest[dest_idx + static_cast<R_xlen_t>(j)] = NA_LOGICAL;
        } else {
          dest[dest_idx + static_cast<R_xlen_t>(j)] = static_cast<int>(src[j]);
        }
      }

      dest_idx += static_cast<R_xlen_t>(chunk_size);
      rows_remaining -= chunk_size;
    }

    R_set_altrep_data2(vec, result);

    // Release the Arrow chunk data now that we have the full LGLSXP
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
    class_t = R_make_altlogical_class("vroom_arrow_lgl", "vroom", dll);

    // altrep
    R_set_altrep_Length_method(class_t, Length);
    R_set_altrep_Inspect_method(class_t, Inspect);

    // altvec
    R_set_altvec_Dataptr_method(class_t, Dataptr);
    R_set_altvec_Dataptr_or_null_method(class_t, Dataptr_or_null);

    // altlogical
    R_set_altlogical_Elt_method(class_t, logical_Elt);
  }
};

[[cpp11::init]] void init_vroom_arrow_lgl(DllInfo* dll);
