#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

Sampler::Sampler(const SamplerParams& p, int n_vocab) : p_(p), n_vocab_(n_vocab) {
    uint64_t seed = p.seed ? p.seed : std::random_device{}();
    rng_.seed(seed);
    idx_.resize(n_vocab);
    probs_.resize(n_vocab);
}

int32_t Sampler::sample(float* logits, const std::vector<int32_t>& recent) {
    if (p_.repeat_penalty != 1.0f && p_.repeat_last_n > 0 && !recent.empty()) {
        size_t start = recent.size() > size_t(p_.repeat_last_n) ? recent.size() - p_.repeat_last_n : 0;
        std::unordered_set<int32_t> seen(recent.begin() + start, recent.end());
        for (int32_t t : seen) {
            if (t < 0 || t >= n_vocab_) continue;
            logits[t] = logits[t] > 0 ? logits[t] / p_.repeat_penalty : logits[t] * p_.repeat_penalty;
        }
    }
    if (p_.temperature <= 0.0f) return int32_t(std::max_element(logits, logits + n_vocab_) - logits);

    // Candidates sorted by logit, descending. top-k trims the list before the softmax.
    for (int i = 0; i < n_vocab_; i++) idx_[i] = i;
    int k = (p_.top_k > 0 && p_.top_k < n_vocab_) ? p_.top_k : n_vocab_;
    std::partial_sort(idx_.begin(), idx_.begin() + k, idx_.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    const float inv_t = 1.0f / p_.temperature;
    const float mx = logits[idx_[0]];
    double sum = 0;
    for (int i = 0; i < k; i++) {
        probs_[i] = expf((logits[idx_[i]] - mx) * inv_t);
        sum += probs_[i];
    }
    for (int i = 0; i < k; i++) probs_[i] = float(probs_[i] / sum);

    // top-p: keep the smallest prefix whose probability mass reaches top_p.
    int n = k;
    if (p_.top_p < 1.0f) {
        double cum = 0;
        for (int i = 0; i < k; i++) {
            cum += probs_[i];
            if (cum >= p_.top_p) { n = i + 1; break; }
        }
    }
    std::discrete_distribution<int> dist(probs_.begin(), probs_.begin() + n);
    return idx_[dist(rng_)];
}
