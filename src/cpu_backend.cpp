#include "cpu_backend.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "ops.h"

CpuBackend::CpuBackend(const Model& m, int n_ctx) : m_(m), n_ctx_(n_ctx) {
    const Config& c = m.cfg;
    const size_t kv_dim = size_t(c.n_head_kv) * c.head_dim;
    x_.resize(c.n_embd); h_.resize(c.n_embd); tmp_.resize(c.n_embd);
    q_.resize(size_t(c.n_head) * c.head_dim); attn_.resize(q_.size());
    k_.resize(kv_dim); v_.resize(kv_dim);
    gate_.resize(c.n_ff); up_.resize(c.n_ff);
    logits_.resize(c.n_vocab);
    scores_.resize(size_t(c.n_head) * n_ctx);
    kcache_.resize(size_t(c.n_layer) * n_ctx * kv_dim);
    vcache_.resize(kcache_.size());
}

const float* CpuBackend::forward(int32_t token, int pos) {
    const Config& c = m_.cfg;
    if (pos < 0 || pos >= n_ctx_) throw std::runtime_error("context window exceeded");
    if (token < 0 || token >= c.n_vocab) throw std::runtime_error("token id out of range");

    const int hd = c.head_dim;
    const size_t kv_dim = size_t(c.n_head_kv) * hd;
    const int group = c.n_head / c.n_head_kv;  // query heads per kv head
    const float scale = 1.0f / sqrtf(float(hd));
    const float* inv_freq = m_.rope_inv_freq.data();

    ops::dequant_row(*m_.tok_embd, size_t(token), x_.data());

    for (int l = 0; l < c.n_layer; l++) {
        const Layer& L = m_.layers[l];

        // --- attention ---
        ops::rmsnorm(h_.data(), x_.data(), L.attn_norm.data(), c.n_embd, c.rms_eps);
        ops::matvec(q_.data(), *L.wq, h_.data());
        ops::matvec(k_.data(), *L.wk, h_.data());
        ops::matvec(v_.data(), *L.wv, h_.data());
        ops::rope(q_.data(), c.n_head, hd, pos, inv_freq);
        ops::rope(k_.data(), c.n_head_kv, hd, pos, inv_freq);

        float* kl = kcache_.data() + size_t(l) * n_ctx_ * kv_dim;
        float* vl = vcache_.data() + size_t(l) * n_ctx_ * kv_dim;
        std::memcpy(kl + size_t(pos) * kv_dim, k_.data(), kv_dim * sizeof(float));
        std::memcpy(vl + size_t(pos) * kv_dim, v_.data(), kv_dim * sizeof(float));

        ops::parallel_for(size_t(c.n_head), [&](size_t h) {
            const float* qh = q_.data() + h * hd;
            const size_t kvh = h / group;
            float* sc = scores_.data() + h * n_ctx_;
            for (int t = 0; t <= pos; t++) {
                const float* kt = kl + size_t(t) * kv_dim + kvh * hd;
                float s = 0;
                for (int i = 0; i < hd; i++) s += qh[i] * kt[i];
                sc[t] = s * scale;
            }
            ops::softmax(sc, size_t(pos) + 1);
            float* out = attn_.data() + h * hd;
            std::memset(out, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float* vt = vl + size_t(t) * kv_dim + kvh * hd;
                const float w = sc[t];
                for (int i = 0; i < hd; i++) out[i] += w * vt[i];
            }
        });

        ops::matvec(tmp_.data(), *L.wo, attn_.data());
        for (int i = 0; i < c.n_embd; i++) x_[i] += tmp_[i];

        // --- feed-forward (SwiGLU) ---
        ops::rmsnorm(h_.data(), x_.data(), L.ffn_norm.data(), c.n_embd, c.rms_eps);
        ops::matvec(gate_.data(), *L.w_gate, h_.data());
        ops::matvec(up_.data(), *L.w_up, h_.data());
        for (int i = 0; i < c.n_ff; i++) gate_[i] = ops::silu(gate_[i]) * up_[i];
        ops::matvec(tmp_.data(), *L.w_down, gate_.data());
        for (int i = 0; i < c.n_embd; i++) x_[i] += tmp_[i];
    }

    ops::rmsnorm(h_.data(), x_.data(), m_.output_norm.data(), c.n_embd, c.rms_eps);
    ops::matvec(logits_.data(), *m_.output, h_.data());
    return logits_.data();
}
