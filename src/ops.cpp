#include "ops.h"

#include <arm_neon.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ops {

using gguf::TensorType;

namespace {

inline uint16_t load_u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline float fp16_to_f32(uint16_t h) { __fp16 v; std::memcpy(&v, &h, 2); return float(v); }
inline float bf16_to_f32(uint16_t h) { uint32_t u = uint32_t(h) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// ---- dequantization of n elements (n = multiple of the block size) starting at p ----

void dequant_f32(const uint8_t* p, float* out, size_t n) { std::memcpy(out, p, n * sizeof(float)); }

void dequant_f16(const uint8_t* p, float* out, size_t n) {
    const __fp16* w = reinterpret_cast<const __fp16*>(p);
    for (size_t i = 0; i < n; i++) out[i] = float(w[i]);
}

void dequant_bf16(const uint8_t* p, float* out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = bf16_to_f32(load_u16(p + 2 * i));
}

// Q8_0: 32 elements per 34-byte block: f16 scale d, then 32 int8. value = d * q.
void dequant_q8_0(const uint8_t* p, float* out, size_t n) {
    for (size_t b = 0; b < n / 32; b++, p += 34, out += 32) {
        float d = fp16_to_f32(load_u16(p));
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
        for (int i = 0; i < 32; i++) out[i] = d * float(q[i]);
    }
}

// Q4_0: 32 elements per 18-byte block: f16 scale d, then 16 bytes of nibbles. The low nibble
// of byte i is element i, the high nibble is element i+16. value = d * (nibble - 8).
void dequant_q4_0(const uint8_t* p, float* out, size_t n) {
    for (size_t b = 0; b < n / 32; b++, p += 18, out += 32) {
        float d = fp16_to_f32(load_u16(p));
        const uint8_t* q = p + 2;
        for (int i = 0; i < 16; i++) {
            out[i] = d * float(int(q[i] & 0xF) - 8);
            out[i + 16] = d * float(int(q[i] >> 4) - 8);
        }
    }
}

// Q4_1: like Q4_0 with an f16 offset m after d (20-byte block). value = d * nibble + m.
void dequant_q4_1(const uint8_t* p, float* out, size_t n) {
    for (size_t b = 0; b < n / 32; b++, p += 20, out += 32) {
        float d = fp16_to_f32(load_u16(p)), m = fp16_to_f32(load_u16(p + 2));
        const uint8_t* q = p + 4;
        for (int i = 0; i < 16; i++) {
            out[i] = d * float(q[i] & 0xF) + m;
            out[i + 16] = d * float(q[i] >> 4) + m;
        }
    }
}

// Q6_K: 256 elements per 210-byte block: ql[128] low 4 bits, qh[64] high 2 bits,
// scales[16] int8 (one per 16 elements), f16 d. value = d * scale * (q6 - 32).
void dequant_q6_k(const uint8_t* p, float* out, size_t n) {
    for (size_t b = 0; b < n / 256; b++, p += 210, out += 256) {
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(p + 192);
        float d = fp16_to_f32(load_u16(p + 208));
        float* y = out;
        for (int half = 0; half < 2; half++) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * float(sc[is]) * float(q1);
                y[l + 32] = d * float(sc[is + 2]) * float(q2);
                y[l + 64] = d * float(sc[is + 4]) * float(q3);
                y[l + 96] = d * float(sc[is + 6]) * float(q4);
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

void dequant(TensorType t, const uint8_t* p, float* out, size_t n) {
    switch (t) {
        case TensorType::F32: dequant_f32(p, out, n); break;
        case TensorType::F16: dequant_f16(p, out, n); break;
        case TensorType::BF16: dequant_bf16(p, out, n); break;
        case TensorType::Q8_0: dequant_q8_0(p, out, n); break;
        case TensorType::Q4_0: dequant_q4_0(p, out, n); break;
        case TensorType::Q4_1: dequant_q4_1(p, out, n); break;
        case TensorType::Q6_K: dequant_q6_k(p, out, n); break;
        default: throw std::runtime_error(std::string("ops: unsupported tensor type ") + gguf::type_name(t));
    }
}

// ---- NEON dot products for the hot types ----

inline float hsum(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d) {
    return vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
}

float dot_f32(const uint8_t* p, const float* x, size_t n) {
    const float* w = reinterpret_cast<const float*>(p);
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(w + i), vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, vld1q_f32(w + i + 4), vld1q_f32(x + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(w + i + 8), vld1q_f32(x + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(w + i + 12), vld1q_f32(x + i + 12));
    }
    float s = hsum(a0, a1, a2, a3);
    for (; i < n; i++) s += w[i] * x[i];
    return s;
}

float dot_f16(const uint8_t* p, const float* x, size_t n) {
    const __fp16* w = reinterpret_cast<const __fp16*>(p);
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        float16x8_t w0 = vld1q_f16(w + i), w1 = vld1q_f16(w + i + 8);
        a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(w0)), vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, vcvt_f32_f16(vget_high_f16(w0)), vld1q_f32(x + i + 4));
        a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(w1)), vld1q_f32(x + i + 8));
        a3 = vfmaq_f32(a3, vcvt_f32_f16(vget_high_f16(w1)), vld1q_f32(x + i + 12));
    }
    float s = hsum(a0, a1, a2, a3);
    for (; i < n; i++) s += float(w[i]) * x[i];
    return s;
}

// Multiply-accumulate 16 int8 values (as two 8-lane halves) against 16 floats of x.
inline void mac_i8x16(float32x4_t& acc, int8x16_t q, const float* x) {
    int16x8_t lo = vmovl_s8(vget_low_s8(q)), hi = vmovl_s8(vget_high_s8(q));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(x));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(x + 4));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(x + 8));
    acc = vfmaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(x + 12));
}

float dot_q8_0(const uint8_t* p, const float* x, size_t n) {
    float32x4_t acc = vdupq_n_f32(0);
    for (size_t b = 0; b < n / 32; b++, p += 34, x += 32) {
        float d = fp16_to_f32(load_u16(p));
        const int8_t* q = reinterpret_cast<const int8_t*>(p + 2);
        float32x4_t blk = vdupq_n_f32(0);
        mac_i8x16(blk, vld1q_s8(q), x);
        mac_i8x16(blk, vld1q_s8(q + 16), x + 16);
        acc = vfmaq_f32(acc, blk, vdupq_n_f32(d));
    }
    return vaddvq_f32(acc);
}

float dot_q4_0(const uint8_t* p, const float* x, size_t n) {
    float32x4_t acc = vdupq_n_f32(0);
    const uint8x16_t mask = vdupq_n_u8(0xF);
    const int8x16_t eight = vdupq_n_s8(8);
    for (size_t b = 0; b < n / 32; b++, p += 18, x += 32) {
        float d = fp16_to_f32(load_u16(p));
        uint8x16_t v = vld1q_u8(p + 2);
        int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight);
        int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
        float32x4_t blk = vdupq_n_f32(0);
        mac_i8x16(blk, lo, x);
        mac_i8x16(blk, hi, x + 16);
        acc = vfmaq_f32(acc, blk, vdupq_n_f32(d));
    }
    return vaddvq_f32(acc);
}

size_t row_bytes(const gguf::TensorInfo& t) {
    return t.ne[0] / gguf::type_block_size(t.type) * gguf::type_block_bytes(t.type);
}

}  // namespace

bool type_supported(TensorType t) {
    switch (t) {
        case TensorType::F32: case TensorType::F16: case TensorType::BF16: case TensorType::Q8_0:
        case TensorType::Q4_0: case TensorType::Q4_1: case TensorType::Q6_K: return true;
        default: return false;
    }
}

void dequant_row(const gguf::TensorInfo& t, size_t row, float* out) {
    dequant(t.type, t.data + row * row_bytes(t), out, t.ne[0]);
}

float dot_row(const gguf::TensorInfo& t, size_t row, const float* x) {
    const uint8_t* p = t.data + row * row_bytes(t);
    size_t n = t.ne[0];
    switch (t.type) {
        case TensorType::F32: return dot_f32(p, x, n);
        case TensorType::F16: return dot_f16(p, x, n);
        case TensorType::Q8_0: return dot_q8_0(p, x, n);
        case TensorType::Q4_0: return dot_q4_0(p, x, n);
        default: {
            thread_local std::vector<float> buf;
            buf.resize(n);
            dequant(t.type, p, buf.data(), n);
            return dot_f32(reinterpret_cast<const uint8_t*>(buf.data()), x, n);
        }
    }
}

void matvec(float* out, const gguf::TensorInfo& W, const float* x) {
    const size_t n_out = W.dim(1);
    const size_t chunk = 16;
    const size_t n_chunks = (n_out + chunk - 1) / chunk;
    parallel_for(n_chunks, [&](size_t c) {
        size_t end = std::min(n_out, (c + 1) * chunk);
        for (size_t r = c * chunk; r < end; r++) out[r] = dot_row(W, r, x);
    });
}

void rmsnorm(float* out, const float* x, const float* weight, size_t n, float eps) {
    float ss = 0;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / float(n) + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

void softmax(float* x, size_t n) {
    float mx = x[0];
    for (size_t i = 1; i < n; i++) mx = std::max(mx, x[i]);
    float sum = 0;
    for (size_t i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

void rope(float* x, int n_heads, int head_dim, int pos, const float* inv_freq) {
    const int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* v = x + h * head_dim;
        for (int i = 0; i < half; i++) {
            float angle = float(pos) * inv_freq[i];
            float c = cosf(angle), s = sinf(angle);
            float x0 = v[2 * i], x1 = v[2 * i + 1];
            v[2 * i] = x0 * c - x1 * s;
            v[2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

}  // namespace ops
