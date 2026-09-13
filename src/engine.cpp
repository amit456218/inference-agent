#include "engine.h"

#include <algorithm>
#include <stdexcept>

#include "cpu_backend.h"
#include "metal_backend.h"

std::unique_ptr<Backend> make_backend(const std::string& name, const gguf::File& f, const Model& model, int n_ctx, int n_batch) {
    if (name == "cpu") return std::make_unique<CpuBackend>(model, n_ctx);
    if (name == "metal") return make_metal_backend(f, model, n_ctx, n_batch);
    throw std::runtime_error("unknown backend " + name + " (use metal or cpu)");
}

const float* prefill(Backend& backend, const std::vector<int32_t>& ids, int& pos, int n_batch) {
    const float* logits = nullptr;
    for (size_t i = 0; i < ids.size(); i += n_batch) {
        int n = int(std::min(size_t(n_batch), ids.size() - i));
        logits = backend.forward_batch(ids.data() + i, n, pos);
        pos += n;
    }
    return logits;
}

void take_complete_utf8(std::string& buf, std::string& out) {
    size_t keep = 0;
    for (size_t i = buf.size(); i > 0 && i + 4 > buf.size(); i--) {
        unsigned char c = buf[i - 1];
        if ((c & 0xC0) == 0x80) continue;
        size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        if (buf.size() - (i - 1) < need) keep = buf.size() - (i - 1);
        break;
    }
    out.append(buf, 0, buf.size() - keep);
    buf.erase(0, buf.size() - keep);
}
