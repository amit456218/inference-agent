// Model config and weight table for the Llama architecture, resolved from a GGUF file.
#pragma once

#include <vector>

#include "gguf.h"

struct Config {
    int n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab, n_ctx_train;
    float rope_theta, rms_eps;
};

struct Layer {
    std::vector<float> attn_norm, ffn_norm;  // small, dequantized at load
    const gguf::TensorInfo *wq, *wk, *wv, *wo, *w_gate, *w_up, *w_down;
};

struct Model {
    Config cfg{};
    const gguf::TensorInfo* tok_embd = nullptr;
    const gguf::TensorInfo* output = nullptr;  // == tok_embd when embeddings are tied
    std::vector<float> output_norm;
    std::vector<float> rope_inv_freq;  // head_dim/2 entries: angle for pair i at position p is p * rope_inv_freq[i]
    std::vector<Layer> layers;

    static Model load(const gguf::File& f);
};
