// Token sampling: repetition penalty, temperature, top-k, top-p, or greedy when temperature is 0.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

struct SamplerParams {
    float temperature = 0.0f;  // 0 = greedy
    int top_k = 40;            // 0 = disabled
    float top_p = 0.95f;       // 1 = disabled
    float repeat_penalty = 1.0f;
    int repeat_last_n = 64;
    uint64_t seed = 0;         // 0 = random seed
};

class Sampler {
public:
    Sampler(const SamplerParams& p, int n_vocab);
    // Picks the next token. `logits` is modified in place. `recent` is the token history used
    // for the repetition penalty (only the last repeat_last_n entries matter).
    int32_t sample(float* logits, const std::vector<int32_t>& recent);
    const SamplerParams& params() const { return p_; }

private:
    SamplerParams p_;
    int n_vocab_;
    std::mt19937_64 rng_;
    std::vector<int32_t> idx_;
    std::vector<float> probs_;
};
