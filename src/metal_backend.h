// Metal GPU backend. Kernels live in shaders/kernels.metal and are compiled at runtime, so only
// the Command Line Tools are needed, not Xcode.
#pragma once

#include <memory>

#include "backend.h"
#include "gguf.h"
#include "model.h"

// Throws std::runtime_error if Metal is unavailable or the shaders fail to compile.
// n_batch is the most tokens one forward_batch call may process at once.
std::unique_ptr<Backend> make_metal_backend(const gguf::File& file, const Model& model, int n_ctx, int n_batch);

// Microbenchmark: runs the linear layer `tensor` on n_tok tokens `iters` times inside one command
// buffer and returns the achieved GFLOP/s. Used to tune the matmul kernels in isolation.
double metal_linear_bench(const gguf::File& file, const Model& model, const std::string& tensor, int n_tok, int iters);

// Diagnostics: compares the tensor-API matmul against the SIMD-group matmul on random input.
void metal_linear_check(const gguf::File& file, const Model& model, const std::string& tensor, int n_tok);

// Diagnostics: GPU read bandwidth (GB/s) over a shared buffer, a private buffer, and the model file.
void metal_bandwidth_test(const gguf::File& file, const Model& model, size_t bytes);
// Diagnostics: seconds per token spent in decode attention across all layers at position pos.
double metal_attention_bench(const gguf::File& file, const Model& model, int pos, int iters);
