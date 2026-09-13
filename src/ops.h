// CPU compute kernels. Weights stay in their on-disk (possibly quantized) layout and every
// kernel that reads a weight matrix dispatches on its type. Activations are always f32.
#pragma once

#include <dispatch/dispatch.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "gguf.h"

namespace ops {

// Runs f(i) for i in [0, n) across all cores using Grand Central Dispatch. Blocks until done.
template <typename F>
void parallel_for(size_t n, F&& f) {
    dispatch_apply_f(n, DISPATCH_APPLY_AUTO, &f, [](void* ctx, size_t i) { (*static_cast<F*>(ctx))(i); });
}

bool type_supported(gguf::TensorType t);

// Dequantize row `row` (0 <= row < ne[1]) of t into out[ne[0]].
void dequant_row(const gguf::TensorInfo& t, size_t row, float* out);
// Dot product of row `row` of t with x[ne[0]].
float dot_row(const gguf::TensorInfo& t, size_t row, const float* x);
// out[r] = dot(row r of W, x) for every row, in parallel. W has ne = [n_in, n_out].
void matvec(float* out, const gguf::TensorInfo& W, const float* x);

void rmsnorm(float* out, const float* x, const float* weight, size_t n, float eps);
void softmax(float* x, size_t n);
// Rotary position embedding in ggml "normal" mode: rotates adjacent pairs (x[2i], x[2i+1])
// inside each head by angle pos * inv_freq[i]. inv_freq has head_dim/2 entries.
void rope(float* x, int n_heads, int head_dim, int pos, const float* inv_freq);
inline float silu(float x) { return x / (1.0f + expf(-x)); }

}  // namespace ops
