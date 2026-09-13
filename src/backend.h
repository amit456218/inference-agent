// A backend runs the model forward and owns the KV cache.
#pragma once

#include <cstdint>
#include <memory>

struct Model;

class Backend {
public:
    virtual ~Backend() = default;
    // Runs `token` at position `pos` (0-based, < n_ctx); returns logits[n_vocab] owned by the backend.
    virtual const float* forward(int32_t token, int pos) = 0;
    // Runs n tokens at positions pos..pos+n-1 and returns the logits after the last one.
    // Backends that can batch override this; the default runs the tokens one at a time.
    virtual const float* forward_batch(const int32_t* tokens, int n, int pos) {
        const float* logits = nullptr;
        for (int i = 0; i < n; i++) logits = forward(tokens[i], pos + i);
        return logits;
    }
    virtual int n_ctx() const = 0;
    virtual const char* name() const = 0;
};
