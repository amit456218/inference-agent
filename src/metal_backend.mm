#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach-o/dyld.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "metal_backend.h"
#include "ops.h"

namespace {

// Parameter structs, laid out exactly as in shaders/kernels.metal.
struct MatvecParams { uint32_t n_in, n_out, out_offset, accumulate; };
struct NormParams { uint32_t n; float eps; };
struct RopeParams { uint32_t n_heads, head_dim, pos, offset, row_stride; };
struct MatmulParams { uint32_t n_in, n_out, n_tok, out_offset, accumulate; };
struct AttnParams { uint32_t n_head, n_head_kv, head_dim, pos, kv_offset, q_stride; float scale; };
struct RopeKParams { uint32_t n_heads, head_dim, pos, dst_offset, row_stride; };
struct AttnDecParams { uint32_t n_head, n_head_kv, head_dim, n_keys, kv_offset, n_splits; float scale; };
struct AttnPrefillParams { uint32_t n_head, n_head_kv, head_dim, pos, n_tok, kv_offset, q_stride; float scale; };

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Finds shaders/kernels.metal next to the working directory or the executable.
std::string load_shader_source() {
    std::vector<std::string> candidates = {"shaders/kernels.metal"};
    char exe[4096];
    uint32_t size = sizeof(exe);
    if (_NSGetExecutablePath(exe, &size) == 0) {
        std::string dir = exe;
        dir = dir.substr(0, dir.find_last_of('/'));
        candidates.push_back(dir + "/shaders/kernels.metal");
        candidates.push_back(dir + "/../shaders/kernels.metal");
    }
    for (const std::string& c : candidates) {
        std::string src = read_file(c);
        if (!src.empty()) return src;
    }
    throw std::runtime_error("metal: cannot find shaders/kernels.metal");
}

const char* matvec_kernel_name(gguf::TensorType t) {
    switch (t) {
        case gguf::TensorType::F32: return "matvec_f32";
        case gguf::TensorType::F16: return "matvec_f16";
        case gguf::TensorType::Q8_0: return "matvec_q8_0";
        case gguf::TensorType::Q4_0: return "matvec_q4_0";
        case gguf::TensorType::Q4_1: return "matvec_q4_1";
        case gguf::TensorType::Q6_K: return "matvec_q6_k";
        default: throw std::runtime_error(std::string("metal: no matvec kernel for type ") + gguf::type_name(t));
    }
}

const char* matmul_kernel_name(gguf::TensorType t) {
    switch (t) {
        case gguf::TensorType::F32: return "matmul_f32";
        case gguf::TensorType::F16: return "matmul_f16";
        case gguf::TensorType::Q8_0: return "matmul_q8_0";
        case gguf::TensorType::Q4_0: return "matmul_q4_0";
        case gguf::TensorType::Q4_1: return "matmul_q4_1";
        case gguf::TensorType::Q6_K: return "matmul_q6_k";
        default: throw std::runtime_error(std::string("metal: no matmul kernel for type ") + gguf::type_name(t));
    }
}

class MetalBackend : public Backend {
public:
    MetalBackend(const gguf::File& file, const Model& model, int n_ctx, int n_batch);
    ~MetalBackend() override {
        if (profile_ && calls_)
            fprintf(stderr, "[metal profile] %d forwards: gpu %.1f ms/call, wall %.1f ms/call, overhead %.1f ms/call; %d single-token: gpu %.2f ms/call\n",
                    calls_, 1e3 * gpu_secs_ / calls_, 1e3 * wall_secs_ / calls_, 1e3 * (wall_secs_ - gpu_secs_) / calls_,
                    calls1_, calls1_ ? 1e3 * gpu1_secs_ / calls1_ : 0.0);
    }
    const float* forward(int32_t token, int pos) override { return forward_batch(&token, 1, pos); }
    const float* forward_batch(const int32_t* tokens, int n, int pos) override;
    double linear_bench(const gguf::TensorInfo& W, int n_tok, int iters);
    void linear_check(const gguf::TensorInfo& W, int n_tok);
    void bandwidth_test(size_t bytes);
    double attention_bench(int pos, int iters);
    int n_ctx() const override { return n_ctx_; }
    const char* name() const override { return "metal"; }

private:
    struct WeightRef { id<MTLBuffer> buf; NSUInteger offset; };
    struct Chunk { id<MTLBuffer> buf; size_t file_offset, length; };

    void wrap_weights(const gguf::File& file);
    WeightRef weight(const gguf::TensorInfo& t) const;
    id<MTLComputePipelineState> pipeline(const char* name);
    id<MTLBuffer> new_buffer(size_t bytes);
    id<MTLBuffer> buffer_from(const std::vector<float>& v);

    // Linear layer: out = W x (+ out). n_tok == 1 uses the bandwidth-bound matvec kernels,
    // larger batches use the tiled matmul kernels.
    void encode_linear(id<MTLComputeCommandEncoder> enc, const gguf::TensorInfo& W, id<MTLBuffer> x,
                       id<MTLBuffer> out, uint32_t out_offset, bool accumulate, int n_tok);
    void encode_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, uint32_t x_offset, id<MTLBuffer> w,
                        id<MTLBuffer> out, int n_rows);
    void encode_rope(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, uint32_t offset, uint32_t row_stride,
                     int n_heads, int pos, int n_tok);
    void encode_attention(id<MTLComputeCommandEncoder> enc, int layer, int pos, int n_tok);
    void encode_kv_store(id<MTLComputeCommandEncoder> enc, uint32_t slot, int pos, int n_tok);
    void encode_silu_mul(id<MTLComputeCommandEncoder> enc, int n_tok);
    // Returns the buffer a linear layer should read: src itself for one token, else xh_ = half(src).
    id<MTLBuffer> linear_input(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, size_t n_elems, int n_tok);

    const Model& m_;
    int n_ctx_, n_batch_;
    bool has_tensor_ = false;
    int mv_nr_ = 4, mv_nsg_ = 2;
    // LLM_PROFILE=1: accumulate GPU time vs wall time per forward and report at exit.
    bool profile_ = getenv("LLM_PROFILE") != nullptr;
    std::string skip_ = getenv("LLM_SKIP") ? getenv("LLM_SKIP") : "";
    bool skip(const char* k) const { return skip_.find(k) != std::string::npos; }
    double gpu_secs_ = 0, wall_secs_ = 0, gpu1_secs_ = 0;
    int calls_ = 0, calls1_ = 0;
    size_t kv_dim_;
    id<MTLDevice> dev_;
    id<MTLCommandQueue> queue_;
    id<MTLLibrary> lib_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> psos_;
    const uint8_t* file_base_ = nullptr;
    std::vector<Chunk> chunks_;
    id<MTLBuffer> x_, h_, q_, attn_, gate_, up_, logits_, kcache_, vcache_, inv_freq_, output_norm_;
    id<MTLBuffer> xh_;  // half-precision copy of the current linear-layer input (batched path)
    id<MTLBuffer> kf32_, vf32_;  // K/V projections before they go into the f16 cache
    id<MTLBuffer> part_;         // flash-decoding partials
    static constexpr int kMaxSplits = 64;
    int attn_splits_ = getenv("LLM_ATTN_SPLITS") ? atoi(getenv("LLM_ATTN_SPLITS")) : 8;
    std::vector<id<MTLBuffer>> attn_norm_, ffn_norm_;
};

MetalBackend::MetalBackend(const gguf::File& file, const Model& model, int n_ctx, int n_batch)
    : m_(model), n_ctx_((n_ctx + 31) / 32 * 32), n_batch_(n_batch) {
    const Config& c = m_.cfg;
    kv_dim_ = size_t(c.n_head_kv) * c.head_dim;
    if (n_batch < 1) throw std::runtime_error("metal: --batch must be at least 1");
    if (c.head_dim > 1024) throw std::runtime_error("metal: head_dim > 1024 not supported");
    if (n_ctx > 7000 && (c.head_dim != 128 || getenv("LLM_NAIVE_ATTN")))
        throw std::runtime_error("metal: --ctx above 7000 needs the flash attention kernels (head_dim 128)");

    dev_ = MTLCreateSystemDefaultDevice();
    if (!dev_) throw std::runtime_error("metal: no GPU device");
    queue_ = [dev_ newCommandQueue];

    // Metal 4 tensor ops (the M5's matrix units) need language version 4.0 and the Metal4 GPU family.
    has_tensor_ = [dev_ supportsFamily:(MTLGPUFamily)5002] && getenv("LLM_NO_TENSOR") == nullptr;
    NSString* src = [NSString stringWithUTF8String:load_shader_source().c_str()];
    MTLCompileOptions* opts = [MTLCompileOptions new];
    mv_nr_ = getenv("LLM_MV_NR") ? atoi(getenv("LLM_MV_NR")) : 4;    // matvec rows per SIMD group
    mv_nsg_ = getenv("LLM_MV_NSG") ? atoi(getenv("LLM_MV_NSG")) : 2;  // matvec SIMD groups per threadgroup
    NSMutableDictionary* macros = [NSMutableDictionary new];
    macros[@"MV_NR"] = @(mv_nr_);
    macros[@"MV_NSG"] = @(mv_nsg_);
    if (getenv("LLM_ATTN_VARIANT")) macros[@"ATTN_VARIANT"] = @(atoi(getenv("LLM_ATTN_VARIANT")));
    if (getenv("LLM_PROBE_MODE")) macros[@"PROBE_MODE"] = @(atoi(getenv("LLM_PROBE_MODE")));
    if (has_tensor_) {
        opts.languageVersion = (MTLLanguageVersion)(4 << 16);
        macros[@"HAS_TENSOR"] = @"1";
    }
    opts.preprocessorMacros = macros;
    NSError* err = nil;
    lib_ = [dev_ newLibraryWithSource:src options:opts error:&err];
    if (!lib_) throw std::runtime_error(std::string("metal: shader compile failed:\n") + err.localizedDescription.UTF8String);

    wrap_weights(file);

    const size_t nb = size_t(n_batch);
    x_ = new_buffer(nb * c.n_embd * 4);
    h_ = new_buffer(nb * c.n_embd * 4);
    q_ = new_buffer(nb * c.n_head * c.head_dim * 4);
    attn_ = new_buffer(nb * c.n_head * c.head_dim * 4);
    gate_ = new_buffer(nb * c.n_ff * 4);
    up_ = new_buffer(nb * c.n_ff * 4);
    xh_ = new_buffer(nb * size_t(std::max(c.n_ff, std::max(c.n_embd, c.n_head * c.head_dim))) * 2);
    logits_ = new_buffer(size_t(c.n_vocab) * 4);
    kcache_ = new_buffer(size_t(c.n_layer) * n_ctx * kv_dim_ * 2);  // f16
    vcache_ = new_buffer(size_t(c.n_layer) * n_ctx * kv_dim_ * 2);
    kf32_ = new_buffer(nb * kv_dim_ * 4);
    vf32_ = new_buffer(nb * kv_dim_ * 4);
    part_ = new_buffer(size_t(kMaxSplits) * c.n_head * 130 * 4);
    inv_freq_ = buffer_from(m_.rope_inv_freq);
    output_norm_ = buffer_from(m_.output_norm);
    for (const Layer& L : m_.layers) {
        attn_norm_.push_back(buffer_from(L.attn_norm));
        ffn_norm_.push_back(buffer_from(L.ffn_norm));
    }
}

// Wraps the mmap'd model file in Metal buffers without copying. Buffers must start on a page
// boundary and be at most maxBufferLength long, so the file is covered by one or more chunks,
// each holding whole tensors.
void MetalBackend::wrap_weights(const gguf::File& file) {
    file_base_ = file.base();
    const size_t page = size_t(getpagesize());
    const size_t max_len = dev_.maxBufferLength;
    auto floor_page = [&](size_t v) { return v / page * page; };
    auto ceil_page = [&](size_t v) { return (v + page - 1) / page * page; };

    struct Span { size_t begin, end; };
    std::vector<Span> spans;
    for (const gguf::TensorInfo& t : file.tensors()) {
        size_t begin = size_t(t.data - file_base_);
        spans.push_back({begin, begin + t.nbytes()});
    }
    std::sort(spans.begin(), spans.end(), [](const Span& a, const Span& b) { return a.begin < b.begin; });

    size_t start = floor_page(spans[0].begin), end = ceil_page(spans[0].end);
    auto flush = [&]() {
        id<MTLBuffer> buf = [dev_ newBufferWithBytesNoCopy:(void*)(file_base_ + start)
                                                    length:end - start
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        if (!buf) throw std::runtime_error("metal: failed to wrap model file in a GPU buffer");
        chunks_.push_back({buf, start, end - start});
    };
    for (size_t i = 1; i < spans.size(); i++) {
        size_t e = ceil_page(spans[i].end);
        if (e - start > max_len) {
            flush();
            start = floor_page(spans[i].begin);
        }
        end = std::max(end, e);
    }
    flush();
}

MetalBackend::WeightRef MetalBackend::weight(const gguf::TensorInfo& t) const {
    size_t off = size_t(t.data - file_base_);
    for (const Chunk& ch : chunks_) {
        if (off >= ch.file_offset && off + t.nbytes() <= ch.file_offset + ch.length)
            return {ch.buf, NSUInteger(off - ch.file_offset)};
    }
    throw std::runtime_error("metal: tensor " + t.name + " not covered by any GPU buffer");
}

id<MTLComputePipelineState> MetalBackend::pipeline(const char* name) {
    auto it = psos_.find(name);
    if (it != psos_.end()) return it->second;
    id<MTLFunction> fn = [lib_ newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) throw std::runtime_error(std::string("metal: kernel not found: ") + name);
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [dev_ newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) throw std::runtime_error(std::string("metal: pipeline failed for ") + name + ": " + err.localizedDescription.UTF8String);
    if (profile_) fprintf(stderr, "[metal pipeline] %-22s max threads/threadgroup %lu\n", name, (unsigned long)pso.maxTotalThreadsPerThreadgroup);
    psos_[name] = pso;
    return pso;
}

id<MTLBuffer> MetalBackend::new_buffer(size_t bytes) {
    id<MTLBuffer> b = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("metal: buffer allocation of " + std::to_string(bytes) + " bytes failed");
    return b;
}

id<MTLBuffer> MetalBackend::buffer_from(const std::vector<float>& v) {
    id<MTLBuffer> b = new_buffer(v.size() * 4);
    memcpy(b.contents, v.data(), v.size() * 4);
    return b;
}

void MetalBackend::encode_linear(id<MTLComputeCommandEncoder> enc, const gguf::TensorInfo& W, id<MTLBuffer> x,
                                 id<MTLBuffer> out, uint32_t out_offset, bool accumulate, int n_tok) {
    if (skip("linear")) return;
    WeightRef w = weight(W);
    const uint32_t n_in = uint32_t(W.ne[0]), n_out = uint32_t(W.ne[1]);
    if (n_tok == 1) {
        MatvecParams p{n_in, n_out, out_offset, accumulate ? 1u : 0u};
        // 32-value block kernels: 2 SIMD groups x 4 rows; the others: 4 SIMD groups x 2 rows.
        const bool q4q8 = W.type == gguf::TensorType::Q4_0 || W.type == gguf::TensorType::Q4_1 || W.type == gguf::TensorType::Q8_0;
        const uint32_t threads = q4q8 ? 32 * mv_nsg_ : 128, rows_per_tg = q4q8 ? mv_nsg_ * mv_nr_ : 8;
        [enc setComputePipelineState:pipeline(matvec_kernel_name(W.type))];
        [enc setBuffer:w.buf offset:w.offset atIndex:0];
        [enc setBuffer:x offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n_out + rows_per_tg - 1) / rows_per_tg, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    } else {
        MatmulParams p{n_in, n_out, uint32_t(n_tok), out_offset, accumulate ? 1u : 0u};
        std::string kname = matmul_kernel_name(W.type);
        if (has_tensor_) kname.insert(strlen("matmul"), "_tensor");
        [enc setComputePipelineState:pipeline(kname.c_str())];
        [enc setBuffer:w.buf offset:w.offset atIndex:0];
        [enc setBuffer:x offset:0 atIndex:1];
        [enc setBuffer:out offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((n_out + 63) / 64, (n_tok + 31) / 32, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
}

void MetalBackend::encode_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, uint32_t x_offset, id<MTLBuffer> w,
                                  id<MTLBuffer> out, int n_rows) {
    if (skip("norm")) return;
    NormParams p{uint32_t(m_.cfg.n_embd), m_.cfg.rms_eps};
    [enc setComputePipelineState:pipeline("rmsnorm")];
    [enc setBuffer:x offset:size_t(x_offset) * 4 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void MetalBackend::encode_rope(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, uint32_t offset, uint32_t row_stride,
                               int n_heads, int pos, int n_tok) {
    if (skip("rope")) return;
    RopeParams p{uint32_t(n_heads), uint32_t(m_.cfg.head_dim), uint32_t(pos), offset, row_stride};
    [enc setComputePipelineState:pipeline("rope")];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:inv_freq_ offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(size_t(n_heads) * m_.cfg.head_dim / 2, n_tok, 1)
        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

void MetalBackend::encode_attention(id<MTLComputeCommandEncoder> enc, int layer, int pos, int n_tok) {
    if (skip("attn")) return;
    const Config& c = m_.cfg;
    const int group = c.n_head / c.n_head_kv;
    if (n_tok == 1 && c.head_dim == 128 && group <= 4 && getenv("LLM_NAIVE_ATTN") == nullptr) {
        // Flash-decoding: split the keys across threadgroups, then merge the partials.
        const uint32_t n_keys = uint32_t(pos) + 1;
        const uint32_t n_splits = std::max(1u, std::min(uint32_t(std::min(attn_splits_, kMaxSplits)), n_keys / 64));
        AttnDecParams dp{uint32_t(c.n_head), uint32_t(c.n_head_kv), uint32_t(c.head_dim), n_keys,
                         uint32_t(size_t(layer) * n_ctx_ * kv_dim_), n_splits, 1.0f / sqrtf(float(c.head_dim))};
        const std::string kname = "attn_decode_g" + std::to_string(group);
        [enc setComputePipelineState:pipeline(kname.c_str())];
        [enc setBuffer:q_ offset:0 atIndex:0];
        [enc setBuffer:kcache_ offset:0 atIndex:1];
        [enc setBuffer:vcache_ offset:0 atIndex:2];
        [enc setBuffer:part_ offset:0 atIndex:3];
        [enc setBytes:&dp length:sizeof(dp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(c.n_head_kv, n_splits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [enc setComputePipelineState:pipeline("attn_combine")];
        [enc setBuffer:part_ offset:0 atIndex:0];
        [enc setBuffer:attn_ offset:0 atIndex:1];
        [enc setBytes:&dp length:sizeof(dp) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(size_t(c.n_head) * c.head_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        return;
    }
    if (n_tok > 1 && c.head_dim == 128 && getenv("LLM_NAIVE_ATTN") == nullptr) {
        AttnPrefillParams fp{uint32_t(c.n_head), uint32_t(c.n_head_kv), uint32_t(c.head_dim), uint32_t(pos), uint32_t(n_tok),
                             uint32_t(size_t(layer) * n_ctx_ * kv_dim_), uint32_t(c.n_head * c.head_dim), 1.0f / sqrtf(float(c.head_dim))};
        [enc setComputePipelineState:pipeline("attn_prefill")];
        [enc setBuffer:q_ offset:0 atIndex:0];
        [enc setBuffer:kcache_ offset:0 atIndex:1];
        [enc setBuffer:vcache_ offset:0 atIndex:2];
        [enc setBuffer:attn_ offset:0 atIndex:3];
        [enc setBytes:&fp length:sizeof(fp) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(c.n_head, (n_tok + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        return;
    }
    AttnParams p{uint32_t(c.n_head), uint32_t(c.n_head_kv), uint32_t(c.head_dim), uint32_t(pos),
                 uint32_t(size_t(layer) * n_ctx_ * kv_dim_), uint32_t(c.n_head * c.head_dim), 1.0f / sqrtf(float(c.head_dim))};
    [enc setComputePipelineState:pipeline("attention")];
    [enc setBuffer:q_ offset:0 atIndex:0];
    [enc setBuffer:kcache_ offset:0 atIndex:1];
    [enc setBuffer:vcache_ offset:0 atIndex:2];
    [enc setBuffer:attn_ offset:0 atIndex:3];
    [enc setBytes:&p length:sizeof(p) atIndex:4];
    [enc setThreadgroupMemoryLength:((size_t(pos) + n_tok) * 4 + 15) / 16 * 16 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(c.n_head, n_tok, 1) threadsPerThreadgroup:MTLSizeMake(c.head_dim, 1, 1)];
}

id<MTLBuffer> MetalBackend::linear_input(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, size_t n_elems, int n_tok) {
    if (n_tok == 1) return src;
    [enc setComputePipelineState:pipeline("to_half")];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:xh_ offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(n_elems / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    return xh_;
}

// K: rope + convert into the cache. V: convert into the cache.
void MetalBackend::encode_kv_store(id<MTLComputeCommandEncoder> enc, uint32_t slot, int pos, int n_tok) {
    const Config& c = m_.cfg;
    RopeKParams rp{uint32_t(c.n_head_kv), uint32_t(c.head_dim), uint32_t(pos), slot, uint32_t(kv_dim_)};
    [enc setComputePipelineState:pipeline("rope_k_to_cache")];
    [enc setBuffer:kf32_ offset:0 atIndex:0];
    [enc setBuffer:kcache_ offset:0 atIndex:1];
    [enc setBuffer:inv_freq_ offset:0 atIndex:2];
    [enc setBytes:&rp length:sizeof(rp) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(kv_dim_ / 2, n_tok, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    [enc setComputePipelineState:pipeline("to_half")];
    [enc setBuffer:vf32_ offset:0 atIndex:0];
    [enc setBuffer:vcache_ offset:size_t(slot) * 2 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(size_t(n_tok) * kv_dim_ / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void MetalBackend::encode_silu_mul(id<MTLComputeCommandEncoder> enc, int n_tok) {
    if (skip("silu")) return;
    [enc setComputePipelineState:pipeline("silu_mul")];
    [enc setBuffer:gate_ offset:0 atIndex:0];
    [enc setBuffer:up_ offset:0 atIndex:1];
    [enc dispatchThreads:MTLSizeMake(size_t(m_.cfg.n_ff) * n_tok, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

const float* MetalBackend::forward_batch(const int32_t* tokens, int n, int pos) {
    const Config& c = m_.cfg;
    if (n < 1 || n > n_batch_) throw std::runtime_error("metal: batch size out of range");
    if (pos < 0 || pos + n > n_ctx_) throw std::runtime_error("context window exceeded");
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= c.n_vocab) throw std::runtime_error("token id out of range");

    // Embedding lookups are single rows: do them on the CPU straight into the shared buffer.
    float* x = static_cast<float*>(x_.contents);
    for (int i = 0; i < n; i++) ops::dequant_row(*m_.tok_embd, size_t(tokens[i]), x + size_t(i) * c.n_embd);

    const uint32_t q_dim = uint32_t(c.n_head) * c.head_dim;
    auto t0 = std::chrono::steady_clock::now();
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (int l = 0; l < c.n_layer; l++) {
            const Layer& L = m_.layers[l];
            const uint32_t slot = uint32_t((size_t(l) * n_ctx_ + pos) * kv_dim_);  // K/V rows for pos..pos+n-1
            encode_rmsnorm(enc, x_, 0, attn_norm_[l], h_, n);
            id<MTLBuffer> in = linear_input(enc, h_, size_t(n) * c.n_embd, n);
            encode_linear(enc, *L.wq, in, q_, 0, false, n);
            encode_linear(enc, *L.wk, in, kf32_, 0, false, n);
            encode_linear(enc, *L.wv, in, vf32_, 0, false, n);
            encode_rope(enc, q_, 0, q_dim, c.n_head, pos, n);
            encode_kv_store(enc, slot, pos, n);
            encode_attention(enc, l, pos, n);
            in = linear_input(enc, attn_, size_t(n) * q_dim, n);
            encode_linear(enc, *L.wo, in, x_, 0, true, n);
            encode_rmsnorm(enc, x_, 0, ffn_norm_[l], h_, n);
            in = linear_input(enc, h_, size_t(n) * c.n_embd, n);
            encode_linear(enc, *L.w_gate, in, gate_, 0, false, n);
            encode_linear(enc, *L.w_up, in, up_, 0, false, n);
            encode_silu_mul(enc, n);
            in = linear_input(enc, gate_, size_t(n) * c.n_ff, n);
            encode_linear(enc, *L.w_down, in, x_, 0, true, n);
        }
        // Logits only for the last token: normalize its row into h_ row 0, then one matvec.
        encode_rmsnorm(enc, x_, uint32_t(n - 1) * c.n_embd, output_norm_, h_, 1);
        encode_linear(enc, *m_.output, h_, logits_, 0, false, 1);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) throw std::runtime_error(std::string("metal: command buffer failed: ") + cb.error.localizedDescription.UTF8String);
        if (profile_) {
            if (n == 1) { gpu1_secs_ += cb.GPUEndTime - cb.GPUStartTime; calls1_++; }
            gpu_secs_ += cb.GPUEndTime - cb.GPUStartTime;
            wall_secs_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            calls_++;
        }
    }
    return static_cast<const float*>(logits_.contents);
}

double MetalBackend::linear_bench(const gguf::TensorInfo& W, int n_tok, int iters) {
    id<MTLBuffer> out = new_buffer(size_t(n_tok) * W.ne[1] * 4);
    auto run = [&](int reps) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLBuffer> in = linear_input(enc, h_, size_t(n_tok) * W.ne[0], n_tok);
            for (int i = 0; i < reps; i++) encode_linear(enc, W, in, out, 0, false, n_tok);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            return double(cb.GPUEndTime - cb.GPUStartTime);
        }
    };
    run(2);  // warm-up
    double secs = run(iters);
    double flops = 2.0 * double(W.ne[0]) * double(W.ne[1]) * n_tok * iters;
    return flops / secs / 1e9;
}

// Runs the tensor-API matmul and the SIMD-group matmul on the same pseudo-random input and
// prints, per token, the largest difference between them (they should agree to f16 rounding).
void MetalBackend::linear_check(const gguf::TensorInfo& W, int n_tok) {
    const size_t n_in = W.ne[0], n_out = W.ne[1];
    float* h = static_cast<float*>(h_.contents);
    uint32_t seed = 12345;
    for (size_t i = 0; i < size_t(n_tok) * n_in; i++) {
        seed = seed * 1664525u + 1013904223u;
        h[i] = float(int32_t(seed >> 8) % 2001 - 1000) / 1000.0f;
    }
    id<MTLBuffer> outs[2] = {new_buffer(size_t(n_tok) * n_out * 4), new_buffer(size_t(n_tok) * n_out * 4)};
    const bool saved = has_tensor_;
    for (int v = 0; v < 2; v++) {
        has_tensor_ = (v == 0) && saved;
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            id<MTLBuffer> in = linear_input(enc, h_, size_t(n_tok) * n_in, n_tok);
            encode_linear(enc, W, in, outs[v], 0, false, n_tok);
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    has_tensor_ = saved;
    const float* a = static_cast<const float*>(outs[0].contents);
    const float* b = static_cast<const float*>(outs[1].contents);
    printf("n_tok=%d tensor=%s: per-token max |tensor - simd| (max |simd|)\n", n_tok, saved ? "yes" : "no");
    for (int t = 0; t < n_tok; t++) {
        float md = 0, mv = 0; size_t worst = 0;
        for (size_t r = 0; r < n_out; r++) {
            float d = fabsf(a[size_t(t) * n_out + r] - b[size_t(t) * n_out + r]);
            if (d > md) { md = d; worst = r; }
            mv = std::max(mv, fabsf(b[size_t(t) * n_out + r]));
        }
        printf("  tok %2d: %10.5f (%.3f)%s  worst row %zu: %.4f vs %.4f\n", t, md, mv, md > 0.05f * mv ? "  <-- MISMATCH" : "",
               worst, a[size_t(t) * n_out + worst], b[size_t(t) * n_out + worst]);
    }
}

void MetalBackend::bandwidth_test(size_t bytes) {
    id<MTLBuffer> out = new_buffer(65536 * 4);
    // Times one bw_sum dispatch over `nbytes` starting at `offset`; returns seconds (best of 5).
    const char* kernel_name = "bw_sum";
    uint32_t elem_bytes = 16;
    auto time_read = [&](id<MTLBuffer> buf, NSUInteger offset, size_t nbytes) {
        const uint32_t n4 = uint32_t(nbytes / elem_bytes);
        double best = 1e9;
        for (int rep = 0; rep < 5; rep++) {
            @autoreleasepool {
                id<MTLCommandBuffer> cb = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipeline(kernel_name)];
                [enc setBuffer:buf offset:offset atIndex:0];
                [enc setBuffer:out offset:0 atIndex:1];
                [enc setBytes:&n4 length:4 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(65536, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                best = std::min(best, double(cb.GPUEndTime - cb.GPUStartTime));
            }
        }
        return best;
    };
    auto fill = [&](id<MTLBuffer> buf) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline("bw_fill")];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(buf.length / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    };
    id<MTLBuffer> priv = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    fill(priv);
    fill(kcache_);
    struct Src { const char* name; id<MTLBuffer> buf; };
    std::vector<Src> srcs = {{"private buffer", priv}, {"KV cache (shared)", kcache_}, {"model file (no-copy)", chunks_[0].buf}};
    fprintf(stderr, "%-22s %10s %10s %10s %10s\n", "read size ->", "4 MB", "16 MB", "64 MB", "256 MB");
    struct Kern { const char* name; uint32_t bytes; };
    for (Kern k : {Kern{"bw_sum", 16}, Kern{"bw_sum_half4", 8}, Kern{"bw_sum_half8", 16}}) {
      kernel_name = k.name; elem_bytes = k.bytes;
      fprintf(stderr, "-- %s (%u-byte loads)\n", k.name, k.bytes);
      for (const Src& src : srcs) {
        for (NSUInteger offset : {NSUInteger(0), NSUInteger(bytes / 2)}) {
            fprintf(stderr, "%-14s @%4zu MB", src.name, size_t(offset / 1000000));
            for (size_t n : {size_t(4e6), size_t(16e6), size_t(64e6), size_t(256e6)}) {
                if (offset + n > src.buf.length) { fprintf(stderr, " %10s", "-"); continue; }
                double t = time_read(src.buf, offset, n);
                fprintf(stderr, " %5.0f GB/s", n / t / 1e9);
            }
            fprintf(stderr, "\n");
        }
      }
    }
}

double MetalBackend::attention_bench(int pos, int iters) {
    // Fill every buffer so no page is an untouched zero page, then time decode attention for all layers.
    for (id<MTLBuffer> buf : {kcache_, vcache_, q_, attn_, part_, x_, h_}) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline("bw_fill")];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc dispatchThreads:MTLSizeMake(buf.length / 16, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
    auto run = [&](int reps) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            const bool probe = getenv("LLM_PROBE") != nullptr;
            for (int i = 0; i < reps; i++)
                for (int l = 0; l < m_.cfg.n_layer; l++) {
                    if (!probe) { encode_attention(enc, l, pos, 1); continue; }
                    const Config& c = m_.cfg;
                    const uint32_t n_keys = uint32_t(pos) + 1;
                    const uint32_t n_splits = std::max(1u, std::min(uint32_t(std::min(attn_splits_, kMaxSplits)), n_keys / 64));
                    AttnDecParams pp{uint32_t(c.n_head), uint32_t(c.n_head_kv), uint32_t(c.head_dim), n_keys,
                                     uint32_t(size_t(l) * n_ctx_ * kv_dim_), n_splits, 1.0f};
                    if (getenv("LLM_PROBE_LAYOUT")) {
                        [enc setComputePipelineState:pipeline("attn_probe_layout")];
                        [enc setBuffer:q_ offset:0 atIndex:0];
                        [enc setBuffer:kcache_ offset:0 atIndex:1];
                        [enc setBuffer:vcache_ offset:0 atIndex:2];
                        [enc setBuffer:part_ offset:0 atIndex:3];
                        [enc setBytes:&pp length:sizeof(pp) atIndex:4];
                    } else {
                    [enc setComputePipelineState:pipeline("attn_probe")];
                    [enc setBuffer:kcache_ offset:0 atIndex:0];
                    [enc setBuffer:vcache_ offset:0 atIndex:1];
                    [enc setBuffer:part_ offset:0 atIndex:2];
                    [enc setBytes:&pp length:sizeof(pp) atIndex:3];
                    [enc setBuffer:q_ offset:0 atIndex:4];
                    }
                    [enc dispatchThreadgroups:MTLSizeMake(c.n_head_kv, n_splits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            return double(cb.GPUEndTime - cb.GPUStartTime);
        }
    };
    for (const char* k : {"attn_decode_g3", "attn_probe", "attn_combine", "matvec_q4_0"})
        fprintf(stderr, "  pipeline %-16s max threads/threadgroup %4lu  (1024 = no register pressure)\n", k,
                (unsigned long)pipeline(k).maxTotalThreadsPerThreadgroup);
    run(1);
    return run(iters) / iters;  // seconds per token (all layers)
}

}  // namespace

double metal_attention_bench(const gguf::File& file, const Model& model, int pos, int iters) {
    MetalBackend b(file, model, pos + 8, 1);
    return b.attention_bench(pos, iters);
}

void metal_bandwidth_test(const gguf::File& file, const Model& model, size_t bytes) {
    MetalBackend b(file, model, 6000, 1);
    b.bandwidth_test(bytes);
}

double metal_linear_bench(const gguf::File& file, const Model& model, const std::string& tensor, int n_tok, int iters) {
    MetalBackend b(file, model, 64, n_tok);
    return b.linear_bench(file.tensor(tensor), n_tok, iters);
}

void metal_linear_check(const gguf::File& file, const Model& model, const std::string& tensor, int n_tok) {
    MetalBackend b(file, model, 64, n_tok);
    b.linear_check(file.tensor(tensor), n_tok);
}

std::unique_ptr<Backend> make_metal_backend(const gguf::File& file, const Model& model, int n_ctx, int n_batch) {
    return std::make_unique<MetalBackend>(file, model, n_ctx, n_batch);
}
