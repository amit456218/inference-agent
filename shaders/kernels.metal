// GPU kernels for the Llama forward pass. Compiled at runtime by the Metal backend.
// Activations are f32. Weights are read in their GGUF layout; block structs below must match
// src/ops.cpp byte for byte.
#include <metal_stdlib>
#ifdef HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

struct BlockQ8_0 { half d; int8_t qs[32]; };                                        // 34 B, 32 values
struct BlockQ4_0 { half d; uint8_t qs[16]; };                                       // 18 B, 32 values
struct BlockQ4_1 { half d; half m; uint8_t qs[16]; };                               // 20 B, 32 values
struct BlockQ6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; };   // 210 B, 256 values
static_assert(sizeof(BlockQ8_0) == 34, "");
static_assert(sizeof(BlockQ4_0) == 18, "");
static_assert(sizeof(BlockQ4_1) == 20, "");
static_assert(sizeof(BlockQ6_K) == 210, "");

struct MatvecParams { uint n_in; uint n_out; uint out_offset; uint accumulate; };
struct NormParams   { uint n; float eps; };
struct RopeParams   { uint n_heads; uint head_dim; uint pos; uint offset; uint row_stride; };
struct AttnParams   { uint n_head; uint n_head_kv; uint head_dim; uint pos; uint kv_offset; uint q_stride; float scale; };
struct RopeKParams  { uint n_heads; uint head_dim; uint pos; uint dst_offset; uint row_stride; };
struct AttnDecParams { uint n_head; uint n_head_kv; uint head_dim; uint n_keys; uint kv_offset; uint n_splits; float scale; };

// ---------------------------------------------------------------- matvec: out[row] = W[row] . x
// Decode is memory-bound: each weight byte is read once, so the kernels are shaped to keep the
// load instruction count low. One SIMD group handles MV_ROWS rows at a time and loads each
// 32-element chunk of x once (as float4s) for all of them; weights are read with 16-bit loads
// because quantized blocks are only 2-byte aligned. 4 SIMD groups per threadgroup.

#define MV_ROWS 2

// Q4_0 / Q4_1: 2 threads per block, each owning half of it (16 values = 8 ushorts of nibbles).
// The nibble is never shifted: the masked ushort is multiplied by a pre-scaled x (x/256 for
// the byte's high half, x/16 and x/4096 for the high nibbles), and the -8 offset of Q4_0 is
// applied once per half-block through the sum of x. 4 rows per SIMD group share the x loads.
#ifndef MV_NR
#define MV_NR 4     // rows per SIMD group for the 32-value block kernels
#endif
#ifndef MV_NSG
#define MV_NSG 2    // SIMD groups per threadgroup
#endif

inline float half_block_dot(device const BlockQ4_0& b, float sumy, thread const float* yl, uint il) {
    device const ushort* qs = (device const ushort*)((device const uchar*)&b + 2) + il / 2;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < 8; i += 2) {
        const uint q = qs[i / 2];
        a0 += yl[i] * float(q & 0x000F);
        a1 += yl[i + 1] * float(q & 0x0F00);
        a2 += yl[i + 8] * float(q & 0x00F0);
        a3 += yl[i + 9] * float(q & 0xF000);
    }
    return float(b.d) * (sumy * -8.0f + a0 + a1 + a2 + a3);
}

inline float half_block_dot(device const BlockQ4_1& b, float sumy, thread const float* yl, uint il) {
    device const ushort* qs = (device const ushort*)((device const uchar*)&b + 4) + il / 2;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (int i = 0; i < 8; i += 2) {
        const uint q = qs[i / 2];
        a0 += yl[i] * float(q & 0x000F);
        a1 += yl[i + 1] * float(q & 0x0F00);
        a2 += yl[i + 8] * float(q & 0x00F0);
        a3 += yl[i + 9] * float(q & 0xF000);
    }
    return float(b.d) * (a0 + a1 + a2 + a3) + float(b.m) * sumy;
}

template <typename Block>
kernel void matvec_q4(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                      device float* out [[buffer(2)]], constant MatvecParams& p [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * MV_NSG + sgid) * MV_NR;
    if (row0 >= p.n_out) return;
    const uint nb = p.n_in / 32;
    device const Block* rows[MV_NR];
    float sum[MV_NR];
    for (uint r = 0; r < MV_NR; r++) {
        rows[r] = (device const Block*)(W + ulong(min(row0 + r, p.n_out - 1)) * nb * sizeof(Block));
        sum[r] = 0;
    }
    const uint ix = lane / 2;        // block within each group of 16
    const uint il = (lane % 2) * 8;  // which half of the block: values il..il+7 and il+16..il+23
    device const float* yb = x + ix * 32 + il;
    float yl[16];
    for (uint ib = ix; ib < nb; ib += 16) {
        float sumy = 0;
        for (int i = 0; i < 8; i += 2) {
            sumy += yb[i] + yb[i + 1] + yb[i + 16] + yb[i + 17];
            yl[i] = yb[i];
            yl[i + 1] = yb[i + 1] / 256.0f;
            yl[i + 8] = yb[i + 16] / 16.0f;
            yl[i + 9] = yb[i + 17] / 4096.0f;
        }
        for (uint r = 0; r < MV_NR; r++) sum[r] += half_block_dot(rows[r][ib], sumy, yl, il);
        yb += 32 * 16;
    }
    for (uint r = 0; r < MV_NR; r++) {
        const float s = simd_sum(sum[r]);
        if (lane == 0 && row0 + r < p.n_out) {
            device float* o = out + p.out_offset + row0 + r;
            *o = p.accumulate ? *o + s : s;
        }
    }
}
template [[host_name("matvec_q4_0")]] kernel void matvec_q4<BlockQ4_0>(device const uchar*, device const float*, device float*, constant MatvecParams&, uint, uint, uint);
template [[host_name("matvec_q4_1")]] kernel void matvec_q4<BlockQ4_1>(device const uchar*, device const float*, device float*, constant MatvecParams&, uint, uint, uint);

// Q8_0: 4 threads per block, 8 values each, 4 rows per SIMD group.
kernel void matvec_q8_0(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* out [[buffer(2)]], constant MatvecParams& p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * MV_NSG + sgid) * MV_NR;
    if (row0 >= p.n_out) return;
    const uint nb = p.n_in / 32;
    device const BlockQ8_0* rows[MV_NR];
    float sum[MV_NR];
    for (uint r = 0; r < MV_NR; r++) {
        rows[r] = (device const BlockQ8_0*)(W + ulong(min(row0 + r, p.n_out - 1)) * nb * sizeof(BlockQ8_0));
        sum[r] = 0;
    }
    const uint ix = lane / 4, il = (lane % 4) * 8;
    device const float* yb = x + ix * 32 + il;
    for (uint ib = ix; ib < nb; ib += 8) {
        float yl[8];
        for (int i = 0; i < 8; i++) yl[i] = yb[i];
        for (uint r = 0; r < MV_NR; r++) {
            device const int8_t* qs = rows[r][ib].qs + il;
            float acc = 0;
            for (int i = 0; i < 8; i++) acc += float(qs[i]) * yl[i];
            sum[r] += acc * float(rows[r][ib].d);
        }
        yb += 32 * 8;
    }
    for (uint r = 0; r < MV_NR; r++) {
        const float s = simd_sum(sum[r]);
        if (lane == 0 && row0 + r < p.n_out) {
            device float* o = out + p.out_offset + row0 + r;
            *o = p.accumulate ? *o + s : s;
        }
    }
}

// Q6_K blocks hold 256 values, too many to split across lanes by block, so all 32 lanes work
// on the same block: lane l owns elements l, l+32, l+64, l+96 of each 128-value half.
kernel void matvec_q6_k(device const uchar* W [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* out [[buffer(2)]], constant MatvecParams& p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * 4 + sgid) * MV_ROWS;
    if (row0 >= p.n_out) return;
    const uint nb = p.n_in / 256;
    const uint is = lane / 16;
    float sum[MV_ROWS];
    device const BlockQ6_K* rows[MV_ROWS];
    for (uint r = 0; r < MV_ROWS; r++) {
        sum[r] = 0;
        rows[r] = (device const BlockQ6_K*)(W + ulong(min(row0 + r, p.n_out - 1)) * nb * sizeof(BlockQ6_K));
    }
    for (uint b = 0; b < nb; b++) {
        device const float* xb = x + b * 256 + lane;
        float xs[8];
        for (int h = 0; h < 2; h++)
            for (int k = 0; k < 4; k++) xs[h * 4 + k] = xb[h * 128 + k * 32];
        for (uint r = 0; r < MV_ROWS; r++) {
            device const BlockQ6_K& blk = rows[r][b];
            float acc = 0;
            for (int h = 0; h < 2; h++) {
                device const uchar* ql = blk.ql + h * 64;
                device const uchar* qh = blk.qh + h * 32;
                device const int8_t* sc = blk.scales + h * 8;
                const uint qll = ql[lane], qlh = ql[lane + 32], qhh = qh[lane];
                const float q1 = float(int((qll & 0xF) | ((qhh & 3) << 4)) - 32);
                const float q2 = float(int((qlh & 0xF) | (((qhh >> 2) & 3) << 4)) - 32);
                const float q3 = float(int((qll >> 4) | (((qhh >> 4) & 3) << 4)) - 32);
                const float q4 = float(int((qlh >> 4) | (((qhh >> 6) & 3) << 4)) - 32);
                acc += float(sc[is]) * q1 * xs[h * 4] + float(sc[is + 2]) * q2 * xs[h * 4 + 1] +
                       float(sc[is + 4]) * q3 * xs[h * 4 + 2] + float(sc[is + 6]) * q4 * xs[h * 4 + 3];
            }
            sum[r] += float(blk.d) * acc;
        }
    }
    for (uint r = 0; r < MV_ROWS; r++) {
        const float s = simd_sum(sum[r]);
        if (lane == 0 && row0 + r < p.n_out) {
            device float* o = out + p.out_offset + row0 + r;
            *o = p.accumulate ? *o + s : s;
        }
    }
}

template <typename T>
kernel void matvec_float(device const T* W [[buffer(0)]], device const float* x [[buffer(1)]],
                         device float* out [[buffer(2)]], constant MatvecParams& p [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * 4 + sgid) * MV_ROWS;
    if (row0 >= p.n_out) return;
    float sum[MV_ROWS];
    device const T* rows[MV_ROWS];
    for (uint r = 0; r < MV_ROWS; r++) {
        sum[r] = 0;
        rows[r] = W + ulong(min(row0 + r, p.n_out - 1)) * p.n_in;
    }
    for (uint i = lane * 8; i < p.n_in; i += 256) {
        const float4 x0 = *(device const float4*)(x + i), x1 = *(device const float4*)(x + i + 4);
        for (uint r = 0; r < MV_ROWS; r++) {
            device const vec<T, 4>* w = (device const vec<T, 4>*)(rows[r] + i);
            sum[r] += dot(float4(w[0]), x0) + dot(float4(w[1]), x1);
        }
    }
    for (uint r = 0; r < MV_ROWS; r++) {
        const float s = simd_sum(sum[r]);
        if (lane == 0 && row0 + r < p.n_out) {
            device float* o = out + p.out_offset + row0 + r;
            *o = p.accumulate ? *o + s : s;
        }
    }
}
template [[host_name("matvec_f16")]] kernel void matvec_float<half>(device const half*, device const float*, device float*, constant MatvecParams&, uint, uint, uint);
template [[host_name("matvec_f32")]] kernel void matvec_float<float>(device const float*, device const float*, device float*, constant MatvecParams&, uint, uint, uint);

// ---------------------------------------------------------------- rmsnorm (one threadgroup)

kernel void rmsnorm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                    device float* out [[buffer(2)]], constant NormParams& p [[buffer(3)]],
                    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint tg_size [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    x += row * p.n;
    out += row * p.n;
    const uint n_sg = (tg_size + 31) / 32;
    float ss = 0;
    for (uint i = tid; i < p.n; i += tg_size) ss += x[i] * x[i];
    ss = simd_sum(ss);
    if (lane == 0) partial[sgid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = lane < n_sg ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = 1.0f / sqrt(partial[0] / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += tg_size) out[i] = x[i] * scale * w[i];
}

// ---------------------------------------------------------------- rope (one thread per pair)

kernel void rope(device float* x [[buffer(0)]], device const float* inv_freq [[buffer(1)]],
                 constant RopeParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    const uint half_dim = p.head_dim / 2;
    const uint h = gid.x / half_dim, i = gid.x % half_dim, tok = gid.y;
    if (h >= p.n_heads) return;
    const float angle = float(p.pos + tok) * inv_freq[i];
    const float c = precise::cos(angle), s = precise::sin(angle);
    device float* v = x + p.offset + tok * p.row_stride + h * p.head_dim + 2 * i;
    const float x0 = v[0], x1 = v[1];
    v[0] = x0 * c - x1 * s;
    v[1] = x0 * s + x1 * c;
}

// RoPE applied to the f32 K projection, written as f16 into the KV cache.
kernel void rope_k_to_cache(device const float* k [[buffer(0)]], device half* cache [[buffer(1)]],
                            device const float* inv_freq [[buffer(2)]], constant RopeKParams& p [[buffer(3)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint half_dim = p.head_dim / 2;
    const uint h = gid.x / half_dim, i = gid.x % half_dim, tok = gid.y;
    if (h >= p.n_heads) return;
    const float angle = float(p.pos + tok) * inv_freq[i];
    const float c = precise::cos(angle), s = precise::sin(angle);
    device const float* v = k + tok * p.row_stride + h * p.head_dim + 2 * i;
    device half* o = cache + p.dst_offset + tok * p.row_stride + h * p.head_dim + 2 * i;
    o[0] = half(v[0] * c - v[1] * s);
    o[1] = half(v[0] * s + v[1] * c);
}

// ---------------------------------------------------------------- decode attention (flash-decoding)
// One token. Grid (n_head_kv, n_splits), 128 threads. A threadgroup handles every query head that
// shares its KV head (GQA group G <= 4) over one slice of the keys, so each K/V row is read once
// for all G heads. Lanes own 4 of the 128 dims; a score is a simd_sum of per-lane dot products.
// Softmax is done online (running max m and sum l); the 4 SIMD groups are merged in threadgroup
// memory and the unnormalized partial (o[128], m, l) per (split, head) goes to `part` for
// attn_combine. Specialized for head_dim == 128.
#define AD_STRIDE 130
#ifndef ATTN_VARIANT
#define ATTN_VARIANT 0
#endif
#if ATTN_VARIANT == 1 || ATTN_VARIANT == 7
#define AEXP(x) exp(x)
#elif ATTN_VARIANT == 2 || ATTN_VARIANT == 8 || ATTN_VARIANT == 9
#define AEXP(x) (x)
#else
#define AEXP(x) precise::exp(x)
#endif
#if ATTN_VARIANT == 3 || ATTN_VARIANT == 7 || ATTN_VARIANT == 8 || ATTN_VARIANT == 9
#define ASUM(x) (x)
#else
#define ASUM(x) simd_sum(x)
#endif
// G (query heads per KV head) is a template parameter so every per-head array has a
// compile-time size: dynamically indexed thread arrays get demoted to memory on Apple GPUs.
template <uint G>
kernel void attn_decode(device const float* q [[buffer(0)]], device const half* K [[buffer(1)]],
                        device const half* V [[buffer(2)]], device float* part [[buffer(3)]],
                        constant AttnDecParams& p [[buffer(4)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * G * AD_STRIDE];
    constexpr uint hd = 128;
    constexpr float kNoMax = -1e30f;  // finite "no key seen yet" so exp(m - m_new) never sees inf - inf
    const uint kvh = tg.x, split = tg.y;
    const uint kv_dim = p.n_head_kv * hd;
    const uint per_split = (p.n_keys + p.n_splits - 1) / p.n_splits;
    const uint k0 = split * per_split, k1 = min(p.n_keys, k0 + per_split);
    device const half* Kb = K + p.kv_offset + kvh * hd + lane * 4;
    device const half* Vb = V + p.kv_offset + kvh * hd + lane * 4;

    float4 qv[G], o[G];
    float m[G], l[G];
#pragma unroll
    for (uint g = 0; g < G; g++) {
#if ATTN_VARIANT == 5
        qv[g] = float4(0.01f * float(g + 1));
#else
        qv[g] = *(device const float4*)(q + (kvh * G + g) * hd + lane * 4) * p.scale;
#endif
        o[g] = float4(0.0f);
        m[g] = kNoMax;
        l[g] = 0;
    }

    // Keys go to the 4 SIMD groups in interleaved blocks of 4. Each key is folded into the
    // running softmax immediately; keys past the slice get a -inf score and contribute nothing.
    for (uint kb = k0 + sgid * 4; kb < k1; kb += 16) {
        float4 k4[4], v4[4];
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            const uint key = min(kb + j, k1 - 1);
            k4[j] = float4(*(device const half4*)(Kb + key * kv_dim));
            v4[j] = float4(*(device const half4*)(Vb + key * kv_dim));
        }
#pragma unroll
        for (uint j = 0; j < 4; j++) {
            const bool valid = kb + j < k1;
#pragma unroll
            for (uint g = 0; g < G; g++) {
                const float s = ASUM(dot(qv[g], k4[j]));
                const float sc = valid ? s : -INFINITY;
                const float m_new = max(m[g], sc);
                const float sf = AEXP(m[g] - m_new);
                const float pj = AEXP(sc - m_new);
                l[g] = l[g] * sf + pj;
                o[g] = o[g] * sf + pj * v4[j];
                m[g] = m_new;
            }
        }
    }

#if ATTN_VARIANT == 6
    if (sgid == 0) {
#pragma unroll
        for (uint g = 0; g < G; g++) {
            device float* dst = part + (split * p.n_head + kvh * G + g) * AD_STRIDE;
            *(device float4*)(dst + lane * 4) = o[g];
            if (lane == 0) { dst[128] = m[g]; dst[129] = l[g]; }
        }
    }
    return;
#endif
#pragma unroll
    for (uint g = 0; g < G; g++) {
        threadgroup float* r = red + (sgid * G + g) * AD_STRIDE;
        *(threadgroup float4*)(r + lane * 4) = o[g];
        if (lane == 0) { r[128] = m[g]; r[129] = l[g]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
#pragma unroll
        for (uint g = 0; g < G; g++) {
            float M = kNoMax;
            for (uint i = 0; i < 4; i++) M = max(M, red[(i * G + g) * AD_STRIDE + 128]);
            float L = 0;
            float4 O = float4(0.0f);
            for (uint i = 0; i < 4; i++) {
                threadgroup const float* r = red + (i * G + g) * AD_STRIDE;
                const float w = precise::exp(r[128] - M);
                L += r[129] * w;
                O += *(threadgroup const float4*)(r + lane * 4) * w;
            }
            device float* dst = part + (split * p.n_head + kvh * G + g) * AD_STRIDE;
            *(device float4*)(dst + lane * 4) = O;
            if (lane == 0) { dst[128] = M; dst[129] = L; }
        }
    }
}
template [[host_name("attn_decode_g1")]] kernel void attn_decode<1>(device const float*, device const half*, device const half*, device float*, constant AttnDecParams&, uint2, uint, uint);
template [[host_name("attn_decode_g2")]] kernel void attn_decode<2>(device const float*, device const half*, device const half*, device float*, constant AttnDecParams&, uint2, uint, uint);

// Hand-unrolled version for G == 3 (Llama 3.2 3B: 24 query heads, 8 KV heads). No arrays at all:
// the compiler must keep every value in registers, since spilled arrays cost more memory
// traffic than the KV cache itself.
#if ATTN_VARIANT == 11 || ATTN_VARIANT == 12
#define AD_INIT(g) float4 o##g = float4(0.0f); float m##g = kNoMax, l##g = 0.0f; \
    const float4 q##g = float4(0.01f * float(g + 1));
#else
#define AD_INIT(g) float4 o##g = float4(0.0f); float m##g = kNoMax, l##g = 0.0f; \
    const float4 q##g = *(device const float4*)(q + (kvh * 3 + g) * hd + lane * 4) * p.scale;
#endif
#define AD_UPDATE(g, k4, v4, valid) { \
    const float s_ = ASUM(dot(q##g, k4)); \
    const float sc_ = (ATTN_VARIANT == 9 || valid) ? s_ : -INFINITY; \
    const float mn_ = max(m##g, sc_); \
    const float sf_ = AEXP(m##g - mn_); \
    const float pj_ = AEXP(sc_ - mn_); \
    l##g = l##g * sf_ + pj_; \
    o##g = o##g * sf_ + pj_ * v4; \
    m##g = mn_; }
#define AD_STORE(g, base) { threadgroup float* r_ = (base) + (g) * AD_STRIDE; \
    *(threadgroup float4*)(r_ + lane * 4) = o##g; if (lane == 0) { r_[128] = m##g; r_[129] = l##g; } }

kernel void attn_decode_g3(device const float* q [[buffer(0)]], device const half* K [[buffer(1)]],
                           device const half* V [[buffer(2)]], device float* part [[buffer(3)]],
                           constant AttnDecParams& p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4 * 3 * AD_STRIDE];
    constexpr uint hd = 128;
    constexpr float kNoMax = -1e30f;
    const uint kvh = tg.x, split = tg.y;
    const uint kv_dim = p.n_head_kv * hd;
    const uint per_split = (p.n_keys + p.n_splits - 1) / p.n_splits;
    const uint k0 = split * per_split, k1 = min(p.n_keys, k0 + per_split);
    device const half* Kb = K + p.kv_offset + kvh * hd + lane * 4;
    device const half* Vb = V + p.kv_offset + kvh * hd + lane * 4;

    AD_INIT(0) AD_INIT(1) AD_INIT(2)
#if ATTN_VARIANT == 10 || ATTN_VARIANT == 11 || ATTN_VARIANT == 12
    // probe body: loads and plain accumulation only, through the real dispatch path
    {
        float4 acc = float4(0.0f);
        for (uint kb = k0 + sgid * 8; kb < k1; kb += 32) {
            for (uint j = 0; j < 8; j++) {
                const uint key = min(kb + j, k1 - 1);
                acc += q0 * float4(*(device const half4*)(Kb + key * kv_dim));
            }
            for (uint j = 0; j < 8; j++) {
                const uint key = min(kb + j, k1 - 1);
                acc += float4(*(device const half4*)(Vb + key * kv_dim));
            }
        }
        o0 = acc;
    }
    if (false)
#endif
    // Two keys per iteration so the loads of the second overlap the math of the first.
    for (uint kb = k0 + sgid * 2; kb < k1; kb += 8) {
        const uint keyA = kb, keyB = min(kb + 1, k1 - 1);
        const float4 kA = float4(*(device const half4*)(Kb + keyA * kv_dim));
        const float4 vA = float4(*(device const half4*)(Vb + keyA * kv_dim));
        const float4 kB = float4(*(device const half4*)(Kb + keyB * kv_dim));
        const float4 vB = float4(*(device const half4*)(Vb + keyB * kv_dim));
        const bool validB = kb + 1 < k1;
        AD_UPDATE(0, kA, vA, true) AD_UPDATE(1, kA, vA, true) AD_UPDATE(2, kA, vA, true)
        AD_UPDATE(0, kB, vB, validB) AD_UPDATE(1, kB, vB, validB) AD_UPDATE(2, kB, vB, validB)
    }

#if ATTN_VARIANT == 12
    if (sgid == 0) { device float* dst = part + (split * p.n_head + kvh * 3) * AD_STRIDE; *(device float4*)(dst + lane * 4) = o0 + o1 + o2; }
    return;
#endif
    threadgroup float* mine = red + sgid * 3 * AD_STRIDE;
    AD_STORE(0, mine) AD_STORE(1, mine) AD_STORE(2, mine)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        for (uint g = 0; g < 3; g++) {
            float M = kNoMax;
            for (uint i = 0; i < 4; i++) M = max(M, red[(i * 3 + g) * AD_STRIDE + 128]);
            float L = 0;
            float4 O = float4(0.0f);
            for (uint i = 0; i < 4; i++) {
                threadgroup const float* r = red + (i * 3 + g) * AD_STRIDE;
                const float w = precise::exp(r[128] - M);
                L += r[129] * w;
                O += *(threadgroup const float4*)(r + lane * 4) * w;
            }
            device float* dst = part + (split * p.n_head + kvh * 3 + g) * AD_STRIDE;
            *(device float4*)(dst + lane * 4) = O;
            if (lane == 0) { dst[128] = M; dst[129] = L; }
        }
    }
}
template [[host_name("attn_decode_g4")]] kernel void attn_decode<4>(device const float*, device const half*, device const half*, device float*, constant AttnDecParams&, uint2, uint, uint);

// Merges the per-split partials: out[h][d] = sum_s o_s[d] e^(m_s - M) / sum_s l_s e^(m_s - M).
kernel void attn_combine(device const float* part [[buffer(0)]], device float* out [[buffer(1)]],
                         constant AttnDecParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    const uint h = gid / 128, d = gid % 128;
    if (h >= p.n_head) return;
    float M = -1e30f;
    for (uint s = 0; s < p.n_splits; s++) M = max(M, part[(s * p.n_head + h) * AD_STRIDE + 128]);
    float L = 0, O = 0;
    for (uint s = 0; s < p.n_splits; s++) {
        device const float* src = part + (s * p.n_head + h) * AD_STRIDE;
        const float w = precise::exp(src[128] - M);
        L += src[129] * w;
        O += src[d] * w;
    }
    out[gid] = L > 0 ? O / L : 0.0f;
}

// ---------------------------------------------------------------- prefill attention (flash attention)
// One SIMD group per (head, block of 8 queries). S = Q K^T and O += P V run as 8x8 tile
// products on the SIMD-group matrix units (Q, K, V, P in f16, S and O accumulate in f32).
// Softmax is online: per row a running max m and sum l; when m grows, O is rescaled by
// multiplying with a diagonal matrix. Keys past the query's position are masked (causal).
struct AttnPrefillParams { uint n_head; uint n_head_kv; uint head_dim; uint pos; uint n_tok; uint kv_offset; uint q_stride; float scale; };

kernel void attn_prefill(device const float* q [[buffer(0)]], device const half* K [[buffer(1)]],
                         device const half* V [[buffer(2)]], device float* out [[buffer(3)]],
                         constant AttnPrefillParams& p [[buffer(4)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint HD = 128, BQ = 8, BK = 32, DT = HD / 8;
    threadgroup half qs[BQ * HD];      // Q block, scaled, as half
    threadgroup float ss[BQ * BK];     // S tile
    threadgroup half ps[BQ * BK];      // P tile
    threadgroup float ds[64];          // diagonal rescale matrix
    threadgroup float os[BQ * HD];     // O staging for the guarded copy-out
    const uint h = tg.x, tok0 = tg.y * BQ;
    const uint kvh = h / (p.n_head / p.n_head_kv);
    const uint kv_dim = p.n_head_kv * HD;
    device const half* Kh = K + p.kv_offset + kvh * HD;
    device const half* Vh = V + p.kv_offset + kvh * HD;

    for (uint i = lane; i < BQ * HD; i += 32) {
        const uint r = i / HD, d = i % HD, tok = tok0 + r;
        qs[i] = tok < p.n_tok ? half(q[tok * p.q_stride + h * HD + d] * p.scale) : half(0.0h);
    }
    for (uint i = lane; i < 64; i += 32) ds[i] = 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_half8x8 Qt[DT];
    for (uint d = 0; d < DT; d++) simdgroup_load(Qt[d], qs + d * 8, HD);
    simdgroup_float8x8 O[DT];
    for (uint d = 0; d < DT; d++) O[d] = simdgroup_float8x8(0.0f);

    // Each row of the 8x32 S tile is owned by 4 lanes (8 columns each).
    const uint row = lane / 4, col0 = (lane % 4) * 8;
    const uint qpos = p.pos + tok0 + row;
    const uint n_keys = min(p.pos + p.n_tok, p.pos + tok0 + BQ);  // last key any query in this block may see, + 1
    float m_r = -1e30f, l_r = 0.0f;

    for (uint kb = 0; kb < n_keys; kb += BK) {
        for (uint t = 0; t < 4; t++) {
            simdgroup_float8x8 S = simdgroup_float8x8(0.0f);
            const uint key0 = kb + t * 8;
            if (key0 < n_keys) {
                for (uint d = 0; d < DT; d++) {
                    simdgroup_half8x8 Kt;
                    simdgroup_load(Kt, Kh + key0 * kv_dim + d * 8, kv_dim, ulong2(0, 0), true);  // (dims x keys)
                    simdgroup_multiply_accumulate(S, Qt[d], Kt, S);
                }
            }
            simdgroup_store(S, ss + t * 8, BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float s8[8];
        float rmax = -INFINITY;
        for (uint j = 0; j < 8; j++) {
            const uint key = kb + col0 + j;
            const float v = ss[row * BK + col0 + j];
            s8[j] = (key <= qpos && key < n_keys) ? v : -INFINITY;
            rmax = max(rmax, s8[j]);
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 2));
        const float m_new = max(m_r, rmax);
        const float alpha = precise::exp(m_r - m_new);
        float rsum = 0.0f;
        for (uint j = 0; j < 8; j++) {
            const float pj = precise::exp(s8[j] - m_new);
            ps[row * BK + col0 + j] = half(pj);
            rsum += pj;
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 2);
        l_r = l_r * alpha + rsum;
        m_r = m_new;
        if (lane % 4 == 0) ds[row * 8 + row] = alpha;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 D;
        simdgroup_load(D, ds, 8);
        for (uint d = 0; d < DT; d++) {
            simdgroup_float8x8 tmp;
            simdgroup_multiply(tmp, D, O[d]);
            O[d] = tmp;
        }
        for (uint t = 0; t < 4; t++) {
            const uint key0 = kb + t * 8;
            if (key0 >= n_keys) break;
            simdgroup_half8x8 Pt;
            simdgroup_load(Pt, ps + t * 8, BK);
            for (uint d = 0; d < DT; d++) {
                simdgroup_half8x8 Vt;
                simdgroup_load(Vt, Vh + key0 * kv_dim + d * 8, kv_dim);  // (keys x dims)
                simdgroup_multiply_accumulate(O[d], Pt, Vt, O[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane % 4 == 0) ds[row * 8 + row] = 1.0f / l_r;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 D;
    simdgroup_load(D, ds, 8);
    for (uint d = 0; d < DT; d++) {
        simdgroup_float8x8 tmp;
        simdgroup_multiply(tmp, D, O[d]);
        simdgroup_store(tmp, os + d * 8, HD);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < BQ * HD; i += 32) {
        const uint r = i / HD, tok = tok0 + r;
        if (tok < p.n_tok) out[tok * p.q_stride + h * HD + i % HD] = os[i];
    }
}

// ---------------------------------------------------------------- attention (one threadgroup per head)
// scores: threadgroup memory with room for pos+1 floats. Threads split the timesteps for the
// score/softmax phases and split the head dimensions for the weighted sum of V.

kernel void attention(device const float* q [[buffer(0)]], device const half* K [[buffer(1)]],
                      device const half* V [[buffer(2)]], device float* out [[buffer(3)]],
                      constant AttnParams& p [[buffer(4)]], threadgroup float* scores [[threadgroup(0)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint2 tg_dim [[threads_per_threadgroup]], uint sgid [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    const uint tg_size = tg_dim.x;
    const uint n_sg = (tg_size + 31) / 32;
    const uint hd = p.head_dim, kv_dim = p.n_head_kv * hd;
    const uint h = tg.x, tok = tg.y;
    const uint kvh = h / (p.n_head / p.n_head_kv);
    const uint n = p.pos + tok + 1;
    device const float* qh = q + tok * p.q_stride + h * hd;
    out += tok * p.q_stride;
    device const half* Kl = K + p.kv_offset + kvh * hd;
    device const half* Vl = V + p.kv_offset + kvh * hd;

    float local_max = -INFINITY;
    for (uint t = tid; t < n; t += tg_size) {
        device const half* kt = Kl + t * kv_dim;
        float s = 0;
        for (uint i = 0; i < hd; i++) s += qh[i] * float(kt[i]);
        s *= p.scale;
        scores[t] = s;
        local_max = max(local_max, s);
    }
    local_max = simd_max(local_max);
    if (lane == 0) partial[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = lane < n_sg ? partial[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mx = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0;
    for (uint t = tid; t < n; t += tg_size) {
        const float e = precise::exp(scores[t] - mx);
        scores[t] = e;
        local_sum += e;
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) partial[sgid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = lane < n_sg ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / partial[0];
    for (uint t = tid; t < n; t += tg_size) scores[t] *= inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < hd; i += tg_size) {
        float acc = 0;
        for (uint t = 0; t < n; t++) acc += scores[t] * float(Vl[t * kv_dim + i]);
        out[h * hd + i] = acc;
    }
}

// ---------------------------------------------------------------- SwiGLU: gate = silu(gate) * up

kernel void silu_mul(device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    const float g = gate[i];
    gate[i] = g / (1.0f + precise::exp(-g)) * up[i];
}

// ================================================================ batched prefill kernels
// For N tokens at once the weight matrix is read once per tile of 32 tokens instead of once
// per token, and the multiply runs on the GPU's SIMD-group matrix units (8x8 tiles).

struct MatmulParams { uint n_in; uint n_out; uint n_tok; uint out_offset; uint accumulate; };

// Elements per quantization block for each row element type.
template <typename T> struct Blk { static constant constexpr uint N = 1; };
template <> struct Blk<BlockQ8_0> { static constant constexpr uint N = 32; };
template <> struct Blk<BlockQ4_0> { static constant constexpr uint N = 32; };
template <> struct Blk<BlockQ4_1> { static constant constexpr uint N = 32; };
template <> struct Blk<BlockQ6_K> { static constant constexpr uint N = 256; };

// Dequantize the 16 consecutive elements of a row that start at element k (k % 16 == 0).
inline void dequant16(device const half* row, uint k, thread half* out) {
    for (int i = 0; i < 16; i++) out[i] = row[k + i];
}
inline void dequant16(device const float* row, uint k, thread half* out) {
    for (int i = 0; i < 16; i++) out[i] = half(row[k + i]);
}
inline void dequant16(device const BlockQ8_0* row, uint k, thread half* out) {
    device const BlockQ8_0& b = row[k / 32];
    const uint off = k % 32;
    const float d = float(b.d);
    for (int i = 0; i < 16; i++) out[i] = half(d * float(b.qs[off + i]));
}
// Nibbles are masked but not shifted: the high nibble is worth 16x, so its scale is d/16.
inline void dequant16(device const BlockQ4_0* row, uint k, thread half* out) {
    device const BlockQ4_0& b = row[k / 32];
    const bool hi = (k % 32) != 0;
    const float d = hi ? float(b.d) / 16.0f : float(b.d);
    const float md = -8.0f * float(b.d);
    const uint mask = hi ? 0xF0 : 0x0F;
    for (int i = 0; i < 16; i++) out[i] = half(d * float(b.qs[i] & mask) + md);
}
inline void dequant16(device const BlockQ4_1* row, uint k, thread half* out) {
    device const BlockQ4_1& b = row[k / 32];
    const bool hi = (k % 32) != 0;
    const float d = hi ? float(b.d) / 16.0f : float(b.d);
    const float m = float(b.m);
    const uint mask = hi ? 0xF0 : 0x0F;
    for (int i = 0; i < 16; i++) out[i] = half(d * float(b.qs[i] & mask) + m);
}
inline void dequant16(device const BlockQ6_K* row, uint k, thread half* out) {
    device const BlockQ6_K& b = row[k / 256];
    const uint r = k % 256, h = r / 128, g = (r % 128) / 32, l0 = r % 32;
    device const uint8_t* ql = b.ql + h * 64;
    device const uint8_t* qh = b.qh + h * 32;
    device const int8_t* sc = b.scales + h * 8;
    const float d = float(b.d);
    for (int i = 0; i < 16; i++) {
        const uint l = l0 + i;
        int q;
        if (g == 0)      q = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4));
        else if (g == 1) q = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4));
        else if (g == 2) q = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
        else             q = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
        out[i] = half(d * float(sc[l / 16 + 2 * g]) * float(q - 32));
    }
}

#define MM_ROWS 64   // weight rows (output features) per threadgroup
#define MM_TOKS 32   // tokens per threadgroup
#define MM_K    32   // reduction step
#define TILE    64   // elements in one 8x8 tile

// f32 -> f16 conversion of activations, 4 elements per thread.
kernel void to_half(device const float4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    y[i] = half4(x[i]);
}

// out[tok][row] = sum_k X[tok][k] * W[row][k], X already in half.
// Threadgroup = 128 threads = 4 SIMD groups; each SIMD group owns 16 rows x 32 tokens = 8 tiles.
// Threadgroup tiles are stored as arrays of contiguous 8x8 blocks so every simdgroup_load reads
// 128 contiguous bytes: block (i, j) lives at (i * 4 + j) * TILE, element (r, c) at r * 8 + c.
template <typename RowT>
kernel void matmul_tiles(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]],
                         device float* out [[buffer(2)]], constant MatmulParams& p [[buffer(3)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint sgid [[simdgroup_index_in_threadgroup]]) {
    threadgroup half Wt[MM_ROWS * MM_K];      // 8 row-tiles x 4 k-tiles
    threadgroup half Xt[MM_TOKS * MM_K];      // 4 tok-tiles x 4 k-tiles
    threadgroup float Ct[MM_TOKS * MM_ROWS];  // [tok][row]
    const uint row0 = tg.x * MM_ROWS, tok0 = tg.y * MM_TOKS;
    const ulong row_bytes = ulong(p.n_in / Blk<RowT>::N) * sizeof(RowT);

    simdgroup_float8x8 acc[8];
    for (int i = 0; i < 8; i++) acc[i] = simdgroup_float8x8(0.0f);
    const uint sg_rt = sgid * 2;  // first of this SIMD group's two row-tiles

    // Load assignments: 2 threads per W row (16 k each), 4 threads per X token (8 k each).
    const uint w_row = tid / 2, w_k = (tid % 2) * 16;
    const uint x_tok = tid / 4, x_k = (tid % 4) * 8;
    const bool w_valid = row0 + w_row < p.n_out;
    const bool x_valid = tok0 + x_tok < p.n_tok;
    device const RowT* wrow = (device const RowT*)(W + ulong(min(row0 + w_row, p.n_out - 1)) * row_bytes);
    device const half* xrow = X + ulong(min(tok0 + x_tok, p.n_tok - 1)) * p.n_in + x_k;
    threadgroup half* wdst = Wt + ((w_row / 8) * 4 + w_k / 8) * TILE + (w_row % 8) * 8;  // next k-tile at +TILE
    threadgroup half* xdst = Xt + ((x_tok / 8) * 4 + x_k / 8) * TILE + (x_tok % 8) * 8;

    for (uint k0 = 0; k0 < p.n_in; k0 += MM_K) {
        thread half tmp[16];
        if (w_valid) dequant16(wrow, k0 + w_k, tmp);
        else for (int i = 0; i < 16; i++) tmp[i] = 0;
        *(threadgroup half4*)(wdst) = half4(tmp[0], tmp[1], tmp[2], tmp[3]);
        *(threadgroup half4*)(wdst + 4) = half4(tmp[4], tmp[5], tmp[6], tmp[7]);
        *(threadgroup half4*)(wdst + TILE) = half4(tmp[8], tmp[9], tmp[10], tmp[11]);
        *(threadgroup half4*)(wdst + TILE + 4) = half4(tmp[12], tmp[13], tmp[14], tmp[15]);
        if (x_valid) {
            *(threadgroup half4*)(xdst) = *(device const half4*)(xrow + k0);
            *(threadgroup half4*)(xdst + 4) = *(device const half4*)(xrow + k0 + 4);
        } else {
            *(threadgroup half4*)(xdst) = half4(0.0h);
            *(threadgroup half4*)(xdst + 4) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kt = 0; kt < 4; kt++) {
            simdgroup_half8x8 a[4], b[2];
            for (int t = 0; t < 4; t++) simdgroup_load(a[t], Xt + (t * 4 + kt) * TILE, 8);
            for (int r = 0; r < 2; r++) simdgroup_load(b[r], Wt + ((sg_rt + r) * 4 + kt) * TILE, 8, ulong2(0, 0), true);
            for (int t = 0; t < 4; t++)
                for (int r = 0; r < 2; r++) simdgroup_multiply_accumulate(acc[t * 2 + r], a[t], b[r], acc[t * 2 + r]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int t = 0; t < 4; t++)
        for (int r = 0; r < 2; r++) simdgroup_store(acc[t * 2 + r], Ct + (t * 8) * MM_ROWS + (sg_rt + r) * 8, MM_ROWS);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < MM_TOKS * MM_ROWS; i += 128) {
        const uint tok = tok0 + i / MM_ROWS, row = row0 + i % MM_ROWS;
        if (tok < p.n_tok && row < p.n_out) {
            device float* o = out + p.out_offset + ulong(tok) * p.n_out + row;
            *o = p.accumulate ? *o + Ct[i] : Ct[i];
        }
    }
}
template [[host_name("matmul_f16")]]  kernel void matmul_tiles<half>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint, uint);
template [[host_name("matmul_f32")]]  kernel void matmul_tiles<float>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint, uint);
template [[host_name("matmul_q8_0")]] kernel void matmul_tiles<BlockQ8_0>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint, uint);
template [[host_name("matmul_q4_0")]] kernel void matmul_tiles<BlockQ4_0>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint, uint);
template [[host_name("matmul_q4_1")]] kernel void matmul_tiles<BlockQ4_1>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint, uint);
template [[host_name("matmul_q6_k")]] kernel void matmul_tiles<BlockQ6_K>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint, uint);

#ifdef HAS_TENSOR
using namespace mpp::tensor_ops;

// Same tiling as matmul_tiles, but the 64x32x32 tile product runs on the M5's matrix units
// through the Metal 4 tensor API. Tiles are plain row-major with K contiguous; the 32 tokens
// are processed as two 16-token halves, the second skipped when the tile is short.
template <typename RowT>
kernel void matmul_tensor(device const uchar* W [[buffer(0)]], device const half* X [[buffer(1)]],
                          device float* out [[buffer(2)]], constant MatmulParams& p [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    // 8 KB of threadgroup memory: W tile [64][32] half and X tile [32][32] half (6 KB) during
    // the K loop, then the same bytes hold the output tile [32][64] float (the full 8 KB).
    threadgroup half shmem[MM_TOKS * MM_ROWS * 2];
    static_assert(MM_ROWS * MM_K + MM_TOKS * MM_K <= MM_TOKS * MM_ROWS * 2, "W and X tiles must fit in the output tile");
    threadgroup half* Wt = shmem;                     // [row][k]
    threadgroup half* Xt = shmem + MM_ROWS * MM_K;    // [tok][k]
    threadgroup float* Ct = (threadgroup float*)shmem;  // [tok][row], 32*64*4 = 8192 bytes
    const uint row0 = tg.x * MM_ROWS, tok0 = tg.y * MM_TOKS;
    const ulong row_bytes = ulong(p.n_in / Blk<RowT>::N) * sizeof(RowT);
    const bool has_hi = p.n_tok > tok0 + 16;

    const uint w_row = tid / 2, w_k = (tid % 2) * 16;
    const uint x_tok = tid / 4, x_k = (tid % 4) * 8;
    const bool w_valid = row0 + w_row < p.n_out;
    const bool x_valid = tok0 + x_tok < p.n_tok;
    device const RowT* wrow = (device const RowT*)(W + ulong(min(row0 + w_row, p.n_out - 1)) * row_bytes);
    device const half* xrow = X + ulong(min(tok0 + x_tok, p.n_tok - 1)) * p.n_in + x_k;
    threadgroup half* wdst = Wt + w_row * MM_K + w_k;
    threadgroup half* xdst = Xt + x_tok * MM_K + x_k;

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Wt, dextents<int32_t, 2>(MM_K, MM_ROWS));
    auto tB0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Xt, dextents<int32_t, 2>(MM_K, 16));
    auto tB1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(Xt + 16 * MM_K, dextents<int32_t, 2>(MM_K, 16));
    matmul2d<matmul2d_descriptor(16, MM_ROWS, MM_K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT0 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto cT1 = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();

    for (uint k0 = 0; k0 < p.n_in; k0 += MM_K) {
        thread half tmp[16];
        if (w_valid) dequant16(wrow, k0 + w_k, tmp);
        else for (int i = 0; i < 16; i++) tmp[i] = 0;
        *(threadgroup half4*)(wdst) = half4(tmp[0], tmp[1], tmp[2], tmp[3]);
        *(threadgroup half4*)(wdst + 4) = half4(tmp[4], tmp[5], tmp[6], tmp[7]);
        *(threadgroup half4*)(wdst + 8) = half4(tmp[8], tmp[9], tmp[10], tmp[11]);
        *(threadgroup half4*)(wdst + 12) = half4(tmp[12], tmp[13], tmp[14], tmp[15]);
        if (x_valid) {
            *(threadgroup half4*)(xdst) = *(device const half4*)(xrow + k0);
            *(threadgroup half4*)(xdst + 4) = *(device const half4*)(xrow + k0 + 4);
        } else {
            *(threadgroup half4*)(xdst) = half4(0.0h);
            *(threadgroup half4*)(xdst + 4) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sA = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);
        mm.run(sB0, sA, cT0);
        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);
            mm.run(sB1, sA, cT1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(Ct, dextents<int32_t, 2>(MM_ROWS, 16));
    cT0.store(tC0);
    if (has_hi) {
        auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(Ct + 16 * MM_ROWS, dextents<int32_t, 2>(MM_ROWS, 16));
        cT1.store(tC1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < MM_TOKS * MM_ROWS; i += 128) {
        const uint tok = tok0 + i / MM_ROWS, row = row0 + i % MM_ROWS;
        if (tok < p.n_tok && row < p.n_out) {
            device float* o = out + p.out_offset + ulong(tok) * p.n_out + row;
            *o = p.accumulate ? *o + Ct[i] : Ct[i];
        }
    }
}
template [[host_name("matmul_tensor_f16")]]  kernel void matmul_tensor<half>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint);
template [[host_name("matmul_tensor_f32")]]  kernel void matmul_tensor<float>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint);
template [[host_name("matmul_tensor_q8_0")]] kernel void matmul_tensor<BlockQ8_0>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint);
template [[host_name("matmul_tensor_q4_0")]] kernel void matmul_tensor<BlockQ4_0>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint);
template [[host_name("matmul_tensor_q4_1")]] kernel void matmul_tensor<BlockQ4_1>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint);
template [[host_name("matmul_tensor_q6_k")]] kernel void matmul_tensor<BlockQ6_K>(device const uchar*, device const half*, device float*, constant MatmulParams&, uint2, uint);
#endif  // HAS_TENSOR


// ---------------------------------------------------------------- bandwidth probe (diagnostics)
// Each thread sums a strided slice of the buffer; used to measure read bandwidth per buffer type.
kernel void bw_sum(device const float4* x [[buffer(0)]], device float* out [[buffer(1)]],
                   constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                   uint threads [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n; i += threads) acc += x[i];
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
kernel void bw_fill(device float4* x [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    x[gid] = float4(float(gid & 1023));
}
// Strided variant: reads `chunk` float4s out of every `stride` float4s (attention-like pattern).
kernel void bw_sum_strided(device const float4* x [[buffer(0)]], device float* out [[buffer(1)]],
                           constant uint& n [[buffer(2)]], constant uint& chunk [[buffer(3)]],
                           constant uint& stride [[buffer(4)]], uint gid [[thread_position_in_grid]],
                           uint threads [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    const uint n_chunks = n / stride;
    for (uint c = gid / chunk; c < n_chunks; c += threads / chunk) acc += x[c * stride + gid % chunk];
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
// Same as bw_sum but with 8-byte half4 loads, to compare 16-bit against 32-bit vector loads.
kernel void bw_sum_half4(device const half4* x [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                         uint threads [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n; i += threads) acc += float4(x[i]);
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
// 16-byte loads reinterpreted as 8 halves.
kernel void bw_sum_half8(device const uint4* x [[buffer(0)]], device float* out [[buffer(1)]],
                         constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]],
                         uint threads [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = gid; i < n; i += threads) {
        const uint4 u = x[i];
        const float2 a = float2(as_type<half2>(u.x)), b = float2(as_type<half2>(u.y));
        const float2 c = float2(as_type<half2>(u.z)), d = float2(as_type<half2>(u.w));
        acc += float4(a + b, c + d);
    }
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}

// Probe with the real kernel's exact argument layout, body of mode 4.
kernel void attn_probe_layout(device const float* q [[buffer(0)]], device const half* K [[buffer(1)]],
                              device const half* V [[buffer(2)]], device float* part [[buffer(3)]],
                              constant AttnDecParams& p [[buffer(4)]],
                              uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const uint hd = 128, kvh = tg.x, split = tg.y;
    const uint kv_dim = p.n_head_kv * hd;
    const uint per_split = (p.n_keys + p.n_splits - 1) / p.n_splits;
    const uint k0 = split * per_split, k1 = min(p.n_keys, k0 + per_split);
    device const half* Kb = K + p.kv_offset + kvh * hd + lane * 4;
    device const half* Vb = V + p.kv_offset + kvh * hd + lane * 4;
    float4 acc = *(device const float4*)(q + kvh * 3 * hd + lane * 4) * p.scale;
    for (uint kb = k0 + sgid * 8; kb < k1; kb += 32) {
        for (uint j = 0; j < 8; j++) {
            const uint key = min(kb + j, k1 - 1);
            acc += float4(*(device const half4*)(Kb + key * kv_dim));
        }
        for (uint j = 0; j < 8; j++) {
            const uint key = min(kb + j, k1 - 1);
            acc += float4(*(device const half4*)(Vb + key * kv_dim));
        }
    }
    if (sgid == 0) { device float* dst = part + (split * p.n_head + kvh * 3) * AD_STRIDE; *(device float4*)(dst + lane * 4) = acc; }
}

// Attention skeleton probe: same grid/threadgroup shape and K/V read pattern as attn_decode,
// but only sums what it reads. PROBE_MODE selects the read pattern.
#ifndef PROBE_MODE
#define PROBE_MODE 0
#endif
kernel void attn_probe(device const half* K [[buffer(0)]], device const half* V [[buffer(1)]],
                       device float* out [[buffer(2)]], constant AttnDecParams& p [[buffer(3)]],
                       device const float* q [[buffer(4)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sgid [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint tid [[thread_index_in_threadgroup]]) {
    const uint hd = 128, kvh = tg.x, split = tg.y;
    const uint kv_dim = p.n_head_kv * hd;
    const uint per_split = (p.n_keys + p.n_splits - 1) / p.n_splits;
    const uint k0 = split * per_split, k1 = min(p.n_keys, k0 + per_split);
    float4 acc = float4(0.0f);
#if PROBE_MODE == 0
    // as attn_decode: per simdgroup blocks of 8 keys, lane reads 4 dims of each key
    device const half* Kb = K + p.kv_offset + kvh * hd + lane * 4;
    device const half* Vb = V + p.kv_offset + kvh * hd + lane * 4;
    for (uint kb = k0 + sgid * 8; kb < k1; kb += 32) {
        for (uint j = 0; j < 8; j++) {
            const uint key = min(kb + j, k1 - 1);
            acc += float4(*(device const half4*)(Kb + key * kv_dim));
        }
        for (uint j = 0; j < 8; j++) {
            const uint key = min(kb + j, k1 - 1);
            acc += float4(*(device const half4*)(Vb + key * kv_dim));
        }
    }
#elif PROBE_MODE == 4 || PROBE_MODE == 5
    threadgroup float red[4 * 3 * AD_STRIDE];
    device const half* Kb = K + p.kv_offset + kvh * hd + lane * 4;
    device const half* Vb = V + p.kv_offset + kvh * hd + lane * 4;
#if PROBE_MODE == 5
    float4 qv0 = *(device const float4*)(q + kvh * 3 * hd + lane * 4) * p.scale;   // the real q read
#else
    float4 qv0 = *(device const float4*)(out + lane * 4);   // stand-in for the q read
#endif
    for (uint kb = k0 + sgid * 8; kb < k1; kb += 32) {
        for (uint j = 0; j < 8; j++) {
            const uint key = min(kb + j, k1 - 1);
            acc += qv0 * float4(*(device const half4*)(Kb + key * kv_dim));
        }
        for (uint j = 0; j < 8; j++) {
            const uint key = min(kb + j, k1 - 1);
            acc += float4(*(device const half4*)(Vb + key * kv_dim));
        }
    }
    threadgroup float* r = red + sgid * AD_STRIDE;
    *(threadgroup float4*)(r + lane * 4) = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) { float4 t = float4(0.0f); for (uint i = 0; i < 4; i++) t += *(threadgroup float4*)(red + i * AD_STRIDE + lane * 4); acc = t; }
#elif PROBE_MODE == 1
    // K only
    device const half* Kb = K + p.kv_offset + kvh * hd + lane * 4;
    for (uint kb = k0 + sgid * 8; kb < k1; kb += 32) {
        for (uint j = 0; j < 8; j++) {
            const uint key = min(kb + j, k1 - 1);
            acc += float4(*(device const half4*)(Kb + key * kv_dim));
        }
    }
#elif PROBE_MODE == 2
    // whole threadgroup walks consecutive keys: thread t reads dims (t%32)*4 of key kb + t/32
    device const half* Kb = K + p.kv_offset + kvh * hd + lane * 4;
    device const half* Vb = V + p.kv_offset + kvh * hd + lane * 4;
    for (uint kb = k0 + sgid; kb < k1; kb += 4) {
        acc += float4(*(device const half4*)(Kb + kb * kv_dim));
        acc += float4(*(device const half4*)(Vb + kb * kv_dim));
    }
#elif PROBE_MODE == 3
    // all 8 kv heads per threadgroup: the threadgroup reads full 2 KB rows (tid covers 512 halves = 1 KB; 2 loads per row)
    device const half* Kb = K + p.kv_offset + tid * 4;
    device const half* Vb = V + p.kv_offset + tid * 4;
    for (uint kb = k0; kb < k1; kb++) {
        acc += float4(*(device const half4*)(Kb + kb * kv_dim)) + float4(*(device const half4*)(Kb + kb * kv_dim + 512));
        acc += float4(*(device const half4*)(Vb + kb * kv_dim)) + float4(*(device const half4*)(Vb + kb * kv_dim + 512));
    }
#endif
    out[(tg.y * p.n_head_kv + tg.x) * 128 + tid] = acc.x + acc.y + acc.z + acc.w;
}
