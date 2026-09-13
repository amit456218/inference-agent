// Reference CPU implementation of the Llama forward pass, one token at a time, with a KV cache.
#pragma once

#include <cstdint>
#include <vector>

#include "backend.h"
#include "model.h"

class CpuBackend : public Backend {
public:
    CpuBackend(const Model& m, int n_ctx);
    // Runs the model on `token` at position `pos` (0-based, must be < n_ctx) and returns the
    // logits for the next token, an array of n_vocab floats owned by the backend.
    const float* forward(int32_t token, int pos) override;
    int n_ctx() const override { return n_ctx_; }
    const char* name() const override { return "cpu"; }

private:
    const Model& m_;
    int n_ctx_;
    std::vector<float> x_, h_, q_, k_, v_, attn_, tmp_, gate_, up_, logits_, scores_;
    std::vector<float> kcache_, vcache_;  // [n_layer][n_ctx][n_head_kv * head_dim]
};
