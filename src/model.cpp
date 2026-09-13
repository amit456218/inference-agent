#include "model.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "ops.h"

namespace {

const gguf::TensorInfo& matrix(const gguf::File& f, const std::string& name, int64_t n_in, int64_t n_out) {
    const gguf::TensorInfo& t = f.tensor(name);
    if (t.ne.size() != 2 || int64_t(t.ne[0]) != n_in || int64_t(t.ne[1]) != n_out)
        throw std::runtime_error("model: " + name + " has unexpected shape (want [" + std::to_string(n_in) +
                                 ", " + std::to_string(n_out) + "])");
    if (!ops::type_supported(t.type))
        throw std::runtime_error("model: " + name + " has unsupported type " + gguf::type_name(t.type));
    return t;
}

std::vector<float> vector(const gguf::File& f, const std::string& name, int64_t n) {
    const gguf::TensorInfo& t = f.tensor(name);
    if (t.ne.size() != 1 || int64_t(t.ne[0]) != n)
        throw std::runtime_error("model: " + name + " has unexpected shape (want [" + std::to_string(n) + "])");
    std::vector<float> v(n);
    ops::dequant_row(t, 0, v.data());
    return v;
}

}  // namespace

Model Model::load(const gguf::File& f) {
    std::string arch = f.get_str("general.architecture", "");
    if (arch != "llama") throw std::runtime_error("model: architecture '" + arch + "' not supported (only llama)");

    Model m;
    Config& c = m.cfg;
    c.n_layer = int(f.get_int("llama.block_count"));
    c.n_embd = int(f.get_int("llama.embedding_length"));
    c.n_head = int(f.get_int("llama.attention.head_count"));
    c.n_head_kv = int(f.get_int("llama.attention.head_count_kv", c.n_head));
    c.head_dim = int(f.get_int("llama.attention.key_length", c.n_embd / c.n_head));
    c.n_ff = int(f.get_int("llama.feed_forward_length"));
    c.n_ctx_train = int(f.get_int("llama.context_length"));
    c.rope_theta = float(f.get_float("llama.rope.freq_base", 10000.0));
    c.rms_eps = float(f.get_float("llama.attention.layer_norm_rms_epsilon", 1e-5));
    c.n_vocab = int(f.get_int("llama.vocab_size", int64_t(f.get("tokenizer.ggml.tokens").array_size())));
    if (c.n_head % c.n_head_kv != 0) throw std::runtime_error("model: head_count not divisible by head_count_kv");
    if (int(f.get_int("llama.attention.value_length", c.head_dim)) != c.head_dim)
        throw std::runtime_error("model: value_length != key_length is not supported");
    if (int(f.get_int("llama.rope.dimension_count", c.head_dim)) != c.head_dim)
        throw std::runtime_error("model: partial rotary dimensions are not supported");

    const int q_dim = c.n_head * c.head_dim, kv_dim = c.n_head_kv * c.head_dim;

    m.tok_embd = &matrix(f, "token_embd.weight", c.n_embd, c.n_vocab);
    m.output = f.find_tensor("output.weight") ? &matrix(f, "output.weight", c.n_embd, c.n_vocab) : m.tok_embd;
    m.output_norm = vector(f, "output_norm.weight", c.n_embd);
    // Rotary frequencies, computed the way ggml does (iterated multiply), optionally divided by
    // the per-frequency factors that Llama 3.1+ use for long-context scaling.
    std::vector<float> factors;
    if (f.find_tensor("rope_freqs.weight")) factors = vector(f, "rope_freqs.weight", c.head_dim / 2);
    m.rope_inv_freq.resize(c.head_dim / 2);
    const float theta_scale = powf(c.rope_theta, -2.0f / float(c.head_dim));
    float freq = 1.0f;
    for (int i = 0; i < c.head_dim / 2; i++) {
        m.rope_inv_freq[i] = factors.empty() ? freq : freq / factors[i];
        freq *= theta_scale;
    }

    m.layers.resize(c.n_layer);
    for (int l = 0; l < c.n_layer; l++) {
        Layer& L = m.layers[l];
        std::string p = "blk." + std::to_string(l) + ".";
        L.attn_norm = vector(f, p + "attn_norm.weight", c.n_embd);
        L.ffn_norm = vector(f, p + "ffn_norm.weight", c.n_embd);
        L.wq = &matrix(f, p + "attn_q.weight", c.n_embd, q_dim);
        L.wk = &matrix(f, p + "attn_k.weight", c.n_embd, kv_dim);
        L.wv = &matrix(f, p + "attn_v.weight", c.n_embd, kv_dim);
        L.wo = &matrix(f, p + "attn_output.weight", q_dim, c.n_embd);
        L.w_gate = &matrix(f, p + "ffn_gate.weight", c.n_embd, c.n_ff);
        L.w_up = &matrix(f, p + "ffn_up.weight", c.n_embd, c.n_ff);
        L.w_down = &matrix(f, p + "ffn_down.weight", c.n_ff, c.n_embd);
    }
    return m;
}
