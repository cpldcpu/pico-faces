/* Integer op layer - the C mirror of quant/int_ops.py (the specification).
 * Every function here must be bit-exact vs the numpy reference; enforced by
 * tests/gen_op_vectors.py + engine/desktop/test_ops.c.
 *
 * Width contract: matmul/conv accumulators are int32 (model sizing guarantees
 * |acc| < 2^31 with >250x headroom); requantization products are int64;
 * two's-complement int32 addition order is irrelevant to the result, which is
 * what lets the M33 SMLAD kernels accumulate pairwise yet match exactly.
 */
#ifndef RF_OPS_H
#define RF_OPS_H

#include <stddef.h>
#include <stdint.h>

#define RMS_R_SHIFT 30
#define RMS_G_Q 8 /* folded (1+scale)/s_out reaches ~64; Q8 covers +-128 */
#define RMS_B_Q 7
#define RMS_OUT_SHIFT (RMS_R_SHIFT + RMS_G_Q)

static inline int8_t rf_sat8(int64_t v) {
    return (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
}

static inline int16_t rf_sat16(int64_t v) {
    return (int16_t)(v > 32767 ? 32767 : (v < -32767 ? -32767 : v));
}

/* (acc*M + 1<<(s-1)) >> s, round half-up, arithmetic shift.
 * Fast path: every fold-emitted shift is >= 33 (observed 33..58 across all
 * models), so the rounding bit and all surviving result bits live in the
 * HIGH word of the 64-bit product - one 32-bit add + asr replaces the
 * compiler's ~35-instruction generic 64-bit variable shift. |acc*M| < 2^62
 * guarantees |hi| <= 2^30, so the round add cannot overflow. The generic
 * path remains for the strided-K final layer (s_v - s_shift can be < 33). */
static inline int64_t rf_rq(int64_t acc, int32_t M, uint8_t s) {
    int64_t p = acc * (int64_t)M;
    if (__builtin_expect(s >= 33, 1)) {
        int32_t hi = (int32_t)(p >> 32);
        return (hi + (1 << (s - 33))) >> (s - 32);
    }
    return (p + ((int64_t)1 << (s - 1))) >> s;
}

uint32_t rf_isqrt64(uint64_t v);

/* int8 dot product, the hot loop of every matmul/conv. K must be a multiple
 * of 4 and both pointers 4-byte aligned for the DSP path (true for every
 * tensor in this model: K in {16,32,128,384,512} and C in {4,8,16,32,64}).
 *
 * The M33 path unpacks 4 bytes/word with SXTB16 and accumulates pairs with
 * SMLAD. Pair order differs from the scalar loop, but two's-complement int32
 * addition is order-independent, so results are bit-exact vs the reference.
 */
#if defined(__ARM_FEATURE_DSP) && !defined(RF_FORCE_REF_DOT)
static inline int32_t rf__sxtb16(uint32_t a) {
    int32_t r;
    __asm("sxtb16 %0, %1" : "=r"(r) : "r"(a));
    return r;
}
static inline int32_t rf__sxtb16_ror8(uint32_t a) {
    int32_t r;
    __asm("sxtb16 %0, %1, ror #8" : "=r"(r) : "r"(a));
    return r;
}
static inline int32_t rf__smlad(int32_t a, int32_t b, int32_t acc) {
    int32_t r;
    __asm("smlad %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}
static inline int32_t rf__smlabb(int32_t a, int32_t b, int32_t acc) {
    int32_t r;
    __asm("smlabb %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}
static inline int32_t rf__smlatb(int32_t a, int32_t b, int32_t acc) {
    int32_t r;
    __asm("smlatb %0, %1, %2, %3" : "=r"(r) : "r"(a), "r"(b), "r"(acc));
    return r;
}
static inline int32_t rf_dot_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t acc) {
    const uint32_t *w32 = (const uint32_t *)w;
    const uint32_t *x32 = (const uint32_t *)x;
    /* K is a multiple of 8 everywhere (dims 32/64/128): 2 words/iter */
    for (int k8 = K >> 3; k8 > 0; k8--) {
        uint32_t wv = *w32++, xv = *x32++;
        acc = rf__smlad(rf__sxtb16(wv), rf__sxtb16(xv), acc);
        acc = rf__smlad(rf__sxtb16_ror8(wv), rf__sxtb16_ror8(xv), acc);
        wv = *w32++;
        xv = *x32++;
        acc = rf__smlad(rf__sxtb16(wv), rf__sxtb16(xv), acc);
        acc = rf__smlad(rf__sxtb16_ror8(wv), rf__sxtb16_ror8(xv), acc);
    }
    return acc;
}
/* 2 output rows x 2 tokens sharing every loaded word: row 1 lives at w+K,
 * token 1's activations at x+K (activation row stride == K at every dense
 * call site). 4 loads per 16 MACs vs 3 loads per 8 in rf_dot2_i8 - the
 * dense layers are SRAM-traffic-bound, not instruction-bound. */
static inline void rf_dot2x2_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t *a00, int32_t *a01, int32_t *a10,
                                int32_t *a11) {
    const uint32_t *w0 = (const uint32_t *)w;
    const uint32_t *w1 = (const uint32_t *)(w + K);
    const uint32_t *x0 = (const uint32_t *)x;
    const uint32_t *x1 = (const uint32_t *)(x + K);
    int32_t s00 = *a00, s01 = *a01, s10 = *a10, s11 = *a11;
    for (int k4 = K >> 2; k4 > 0; k4--) {
        uint32_t v = *x0++;
        int32_t xe0 = rf__sxtb16(v), xo0 = rf__sxtb16_ror8(v);
        v = *x1++;
        int32_t xe1 = rf__sxtb16(v), xo1 = rf__sxtb16_ror8(v);
        v = *w0++;
        int32_t we = rf__sxtb16(v), wo = rf__sxtb16_ror8(v);
        s00 = rf__smlad(we, xe0, s00);
        s00 = rf__smlad(wo, xo0, s00);
        s01 = rf__smlad(we, xe1, s01);
        s01 = rf__smlad(wo, xo1, s01);
        v = *w1++;
        we = rf__sxtb16(v);
        wo = rf__sxtb16_ror8(v);
        s10 = rf__smlad(we, xe0, s10);
        s10 = rf__smlad(wo, xo0, s10);
        s11 = rf__smlad(we, xe1, s11);
        s11 = rf__smlad(wo, xo1, s11);
    }
    *a00 = s00;
    *a01 = s01;
    *a10 = s10;
    *a11 = s11;
}

/* two output rows sharing each x load: ~25% fewer instructions than 2x dot */
static inline void rf_dot2_i8(const int8_t *w0, const int8_t *w1,
                              const int8_t *x, int K, int32_t *a0,
                              int32_t *a1) {
    const uint32_t *p0 = (const uint32_t *)w0, *p1 = (const uint32_t *)w1;
    const uint32_t *x32 = (const uint32_t *)x;
    int32_t s0 = *a0, s1 = *a1;
    for (int k8 = K >> 3; k8 > 0; k8--) {
        uint32_t xv = *x32++;
        int32_t xe = rf__sxtb16(xv), xo = rf__sxtb16_ror8(xv);
        uint32_t wv = *p0++;
        s0 = rf__smlad(rf__sxtb16(wv), xe, s0);
        s0 = rf__smlad(rf__sxtb16_ror8(wv), xo, s0);
        wv = *p1++;
        s1 = rf__smlad(rf__sxtb16(wv), xe, s1);
        s1 = rf__smlad(rf__sxtb16_ror8(wv), xo, s1);
        xv = *x32++;
        xe = rf__sxtb16(xv);
        xo = rf__sxtb16_ror8(xv);
        wv = *p0++;
        s0 = rf__smlad(rf__sxtb16(wv), xe, s0);
        s0 = rf__smlad(rf__sxtb16_ror8(wv), xo, s0);
        wv = *p1++;
        s1 = rf__smlad(rf__sxtb16(wv), xe, s1);
        s1 = rf__smlad(rf__sxtb16_ror8(wv), xo, s1);
    }
    *a0 = s0;
    *a1 = s1;
}
/* a[0..7] += v * w[0..7], w word-aligned. SXTB16 yields (w0,w2)/(w1,w3), so
 * SMLABB/SMLATB pick the right halves: 7 instructions per 4 MACs. */
static inline void rf_axpy8_i8(const int8_t *w, int32_t v, int32_t *a) {
    uint32_t lo = ((const uint32_t *)w)[0], hi = ((const uint32_t *)w)[1];
    int32_t e = rf__sxtb16(lo), o = rf__sxtb16_ror8(lo);
    a[0] = rf__smlabb(e, v, a[0]);
    a[1] = rf__smlabb(o, v, a[1]);
    a[2] = rf__smlatb(e, v, a[2]);
    a[3] = rf__smlatb(o, v, a[3]);
    e = rf__sxtb16(hi);
    o = rf__sxtb16_ror8(hi);
    a[4] = rf__smlabb(e, v, a[4]);
    a[5] = rf__smlabb(o, v, a[5]);
    a[6] = rf__smlatb(e, v, a[6]);
    a[7] = rf__smlatb(o, v, a[7]);
}
#else
static inline int32_t rf_dot_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t acc) {
    for (int k = 0; k < K; k++) acc += (int32_t)w[k] * x[k];
    return acc;
}
static inline void rf_dot2_i8(const int8_t *w0, const int8_t *w1,
                              const int8_t *x, int K, int32_t *a0,
                              int32_t *a1) {
    *a0 = rf_dot_i8(w0, x, K, *a0);
    *a1 = rf_dot_i8(w1, x, K, *a1);
}
static inline void rf_axpy8_i8(const int8_t *w, int32_t v, int32_t *a) {
    for (int j = 0; j < 8; j++) a[j] += v * (int32_t)w[j];
}
static inline void rf_dot2x2_i8(const int8_t *w, const int8_t *x, int K,
                                int32_t *a00, int32_t *a01, int32_t *a10,
                                int32_t *a11) {
    for (int k = 0; k < K; k++) {
        *a00 += (int32_t)w[k] * x[k];
        *a01 += (int32_t)w[k] * x[K + k];
        *a10 += (int32_t)w[K + k] * x[k];
        *a11 += (int32_t)w[K + k] * x[K + k];
    }
}
#endif

/* y[O] = sat8(rq(W@x + b)); relu clamps below at 0 */
void rf_linear_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                  const int32_t *M, const uint8_t *s, int K, int O, int relu,
                  int8_t *y);

/* res[O] = sat16(res[O] + rq(W@x + b)) - branch-final projections */
void rf_linear_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b,
                        const int32_t *M, const uint8_t *s, int K, int O,
                        int16_t *res);

/* token-pair variants: token 1's activations at x+K (row stride == K).
 * Same math per token as the single versions - byte-exact. */
void rf_linear2_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                   const int32_t *M, const uint8_t *s, int K, int O, int relu,
                   int8_t *y0, int8_t *y1);
void rf_linear2_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b,
                         const int32_t *M, const uint8_t *s, int K, int O,
                         int16_t *res0, int16_t *res1);

/* v7 relu2 square-requant fc1: y = sat8(rq(relu(W@x + b)^2 >> 12, M, s)) --
 * the activation squares the RAW accumulator (no intermediate requant/LUT);
 * per-channel output scales live in M (relu2 homogeneity, see quant/fold.py) */
void rf_relu2sq_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                   const int32_t *M, const uint8_t *s, int K, int O, int8_t *y);
void rf_relu2sq2_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                    const int32_t *M, const uint8_t *s, int K, int O,
                    int8_t *y0, int8_t *y1);

/* x: HWC i8, w: [O][3][3][C], zero pad 1. stride 1|2. upsample_in fuses NN-2x
 * (H,W are the *input* dims; output is 2H x 2W). Exactly one of y / res16.
 * out_u8: final image layer, y holds uint8 = clip(rq + 128, 0, 255). */
void rf_conv3x3_i8(const int8_t *x, int H, int W, int C, const int8_t *w,
                   const int32_t *b, const int32_t *M, const uint8_t *s, int O,
                   int relu, int stride, int upsample_in, int out_u8,
                   int8_t *y, int16_t *res16);

/* output-row-range variant for parallel execution */
void rf_conv3x3_i8_rows(const int8_t *x, int H, int W, int C, const int8_t *w,
                        const int32_t *b, const int32_t *M, const uint8_t *s,
                        int O, int relu, int stride, int upsample_in,
                        int out_u8, int8_t *y, int16_t *res16, int yo0,
                        int yo1);

void rf_requant_i16_to_i8(const int16_t *x, int n, int32_t M, uint8_t s,
                          int8_t *y);

/* G: Q14 folded gain, B: Q7 folded bias, both in output-scale units */
void rf_rmsnorm_i16_to_i8(const int16_t *x, const int16_t *G, const int16_t *B,
                          int K, int8_t *y);

void rf_softmax_i32_to_i8(const int32_t *scores, int N, int32_t M_sm,
                          uint8_t s_sm, const uint16_t *explut, int lut_max,
                          int8_t *p);

void rf_lut_i8(const int8_t *x, int n, const int8_t *lut, int8_t *y);

/* ---- sparse (zero-skipping) paths -------------------------------------
 * ReLU/ReLU2/LUT outputs are 30-55% exact zeros; skipping them is exact
 * because int32 accumulation order does not change the sum. Weights for
 * these paths are stored transposed (contiguous over the output dim) so
 * the gather reads whole words: see quant/export.py version 2. */

/* collect nonzero entries; returns count */
int rf_compact_i8(const int8_t *x, int n, uint16_t *idx, int8_t *val);

/* res[o] = sat16(res[o] + rq(b[o] + sum_i val[i]*Wt[idx[i]][o])), O % 8 == 0 */
void rf_axpy_acc16_sp(const uint16_t *idx, const int8_t *val, int m,
                      const int8_t *Wt, int O, const int32_t *b,
                      const int32_t *M, const uint8_t *s, int16_t *res);

/* worker identity for per-core scratch (weak 0; firmware: get_core_num) */
int rf_core_id(void);

/* ---- PRNG: PCG32 (XSH-RR) + CLT-12 gaussian (sigma = 4096, Q12) ---- */
typedef struct {
    uint64_t state;
} rf_pcg32_t;

void rf_pcg32_seed(rf_pcg32_t *g, uint64_t seed);
uint32_t rf_pcg32_next(rf_pcg32_t *g);
void rf_gaussian_clt12(rf_pcg32_t *g, int n, int32_t *out);

/* CRC-32 IEEE reflected, init 0xFFFFFFFF outside, final xor outside */
uint32_t rf_crc32(const uint8_t *data, size_t len);
#define RF_CRC32_INIT 0xFFFFFFFFu
uint32_t rf_crc32_acc(uint32_t crc, const uint8_t *data, size_t len);

/* buffers feeding rf_dot_i8 must be word-aligned for the DSP path */
#define RF_ALIGN4 __attribute__((aligned(4)))

/* Hot kernels: firmware places these in SRAM (.time_critical) so instruction
 * fetch does not contend with weight reads on the flash/XIP bus. */
#ifdef RF_HOT_IN_RAM
#define RF_HOT __attribute__((section(".time_critical.rfeng")))
#else
#define RF_HOT
#endif

/* Split [0,n) across workers and run fn on each range; results must only
 * depend on disjoint writes. Weak serial default in kernels_ref.c; the
 * firmware overrides it with a dual-core implementation (firmware/par.c). */
void rf_par_for(int n, void (*fn)(int i0, int i1, void *ctx), void *ctx);

/* 2 x 128KB scratch: VAE decoder ping-pong; DiT-phase weight staging arena */
#define RF_ARENA_HALF (128 * 128 * 8)
extern int8_t rf_arena[2][RF_ARENA_HALF];

/* ---- DiT weight staging ------------------------------------------------
 * Fixed slots in the decoder arena (idle during the DiT phase). Consecutive
 * ops use different slots, so the NEXT op's weights can copy while the
 * current op computes. rf_stage_start begins the copy (weak default:
 * synchronous memcpy; firmware: DMA from the uncached XIP alias, one
 * channel per slot, polled - no IRQs). rf_stage_wait blocks until the
 * slot's copy is complete and returns its pointer. Overlap changes only
 * WHEN bytes move, never what the kernels read: results stay byte-exact.
 * FC1/FC2 are adjacent so the MLP sees one contiguous 128KB block. */
enum {
    RF_SLOT_FC1,   /* arena[0] + 0     64KB */
    RF_SLOT_FC2,   /* arena[0] + 64KB  64KB */
    RF_SLOT_QKV,   /* arena[1] + 0     48KB */
    RF_SLOT_PROJ,  /* arena[1] + 48KB  16KB */
    RF_SLOT_EMB,   /* arena[1] + 64KB   4KB (staged once per generate) */
    RF_SLOT_FINAL, /* arena[1] + 68KB   4KB (staged once per generate) */
    RF_SLOT_N
};

static inline int8_t *rf_stage_slot(int slot) {
    switch (slot) {
    case RF_SLOT_FC1: return rf_arena[0];
    case RF_SLOT_FC2: return rf_arena[0] + 65536;
    case RF_SLOT_QKV: return rf_arena[1];
    case RF_SLOT_PROJ: return rf_arena[1] + 49152;
    case RF_SLOT_EMB: return rf_arena[1] + 65536;
    default: return rf_arena[1] + 69632;
    }
}

void rf_stage_start(int slot, const void *src, size_t n);
const int8_t *rf_stage_wait(int slot);
void rf_stage_drain(void); /* all slots idle (call before the decoder) */

#endif
