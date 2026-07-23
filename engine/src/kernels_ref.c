/* Portable reference kernels. See rf_ops.h for the width contract. */
#include <string.h>

#include "rf_ops.h"

int8_t rf_arena[2][RF_ARENA_HALF] RF_ALIGN4;

__attribute__((weak)) void rf_par_for(int n, void (*fn)(int, int, void *),
                                      void *ctx) {
    fn(0, n, ctx);
}

/* weak synchronous staging (desktop / no-DMA builds); firmware overrides
 * with per-slot DMA channels (firmware/stage.c) */
__attribute__((weak)) void rf_stage_start(int slot, const void *src, size_t n) {
    memcpy(rf_stage_slot(slot), src, n);
}

__attribute__((weak)) const int8_t *rf_stage_wait(int slot) {
    return rf_stage_slot(slot);
}

__attribute__((weak)) void rf_stage_drain(void) {}

RF_HOT uint32_t rf_isqrt64(uint64_t v) {
    uint64_t r = 0, bit = (uint64_t)1 << 62;
    while (bit > v) bit >>= 2;
    while (bit) {
        if (v >= r + bit) {
            v -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return (uint32_t)r;
}

RF_HOT void rf_linear_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                  const int32_t *M, const uint8_t *s, int K, int O, int relu,
                  int8_t *y) {
    for (int o = 0; o < O; o += 2) {
        int32_t a0 = b[o], a1 = b[o + 1];
        rf_dot2_i8(W + (size_t)o * K, W + (size_t)(o + 1) * K, x, K, &a0, &a1);
        int64_t v0 = rf_rq(a0, M[o], s[o]);
        int64_t v1 = rf_rq(a1, M[o + 1], s[o + 1]);
        if (relu) {
            if (v0 < 0) v0 = 0;
            if (v1 < 0) v1 = 0;
        }
        y[o] = rf_sat8(v0);
        y[o + 1] = rf_sat8(v1);
    }
}

/* v7 relu2 square-requant epilogue: h1 = sat8(rq(relu(acc)^2 >> 12, M, s))
 * on the RAW fc1 accumulator -- no intermediate requant/LUT/clip (relu2 is
 * degree-2 homogeneous, so the per-channel output scale lives entirely in M;
 * see quant/fold.py). Ranges: |acc| <= K*127^2 < 2^21, u < 2^30, u*M < 2^61. */
static inline int8_t rf_sq_rq(int32_t a, int32_t M, uint8_t s) {
    if (a <= 0) return 0; /* == rq(0) */
    int64_t u = ((int64_t)a * a) >> 12;
    return rf_sat8(rf_rq(u, M, s));
}

RF_HOT void rf_relu2sq_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                          const int32_t *M, const uint8_t *s, int K, int O,
                          int8_t *y) {
    for (int o = 0; o < O; o += 2) {
        int32_t a0 = b[o], a1 = b[o + 1];
        rf_dot2_i8(W + (size_t)o * K, W + (size_t)(o + 1) * K, x, K, &a0, &a1);
        y[o] = rf_sq_rq(a0, M[o], s[o]);
        y[o + 1] = rf_sq_rq(a1, M[o + 1], s[o + 1]);
    }
}

RF_HOT void rf_relu2sq2_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                           const int32_t *M, const uint8_t *s, int K, int O,
                           int8_t *y0, int8_t *y1) {
    for (int o = 0; o < O; o += 2) {
        int32_t a00 = b[o], a01 = b[o];
        int32_t a10 = b[o + 1], a11 = b[o + 1];
        rf_dot2x2_i8(W + (size_t)o * K, x, K, &a00, &a01, &a10, &a11);
        y0[o] = rf_sq_rq(a00, M[o], s[o]);
        y1[o] = rf_sq_rq(a01, M[o], s[o]);
        y0[o + 1] = rf_sq_rq(a10, M[o + 1], s[o + 1]);
        y1[o + 1] = rf_sq_rq(a11, M[o + 1], s[o + 1]);
    }
}

RF_HOT void rf_linear_i8_acc16(const int8_t *x, const int8_t *W, const int32_t *b,
                        const int32_t *M, const uint8_t *s, int K, int O,
                        int16_t *res) {
    for (int o = 0; o < O; o += 2) {
        int32_t a0 = b[o], a1 = b[o + 1];
        rf_dot2_i8(W + (size_t)o * K, W + (size_t)(o + 1) * K, x, K, &a0, &a1);
        res[o] = rf_sat16((int64_t)res[o] + rf_rq(a0, M[o], s[o]));
        res[o + 1] = rf_sat16((int64_t)res[o + 1] + rf_rq(a1, M[o + 1], s[o + 1]));
    }
}

/* token-pair variants: same per-token math, half the weight/activation
 * traffic (each loaded word feeds two dot products) */
RF_HOT void rf_linear2_i8(const int8_t *x, const int8_t *W, const int32_t *b,
                          const int32_t *M, const uint8_t *s, int K, int O,
                          int relu, int8_t *y0, int8_t *y1) {
    for (int o = 0; o < O; o += 2) {
        int32_t a00 = b[o], a01 = b[o];
        int32_t a10 = b[o + 1], a11 = b[o + 1];
        rf_dot2x2_i8(W + (size_t)o * K, x, K, &a00, &a01, &a10, &a11);
        int64_t v00 = rf_rq(a00, M[o], s[o]);
        int64_t v01 = rf_rq(a01, M[o], s[o]);
        int64_t v10 = rf_rq(a10, M[o + 1], s[o + 1]);
        int64_t v11 = rf_rq(a11, M[o + 1], s[o + 1]);
        if (relu) {
            if (v00 < 0) v00 = 0;
            if (v01 < 0) v01 = 0;
            if (v10 < 0) v10 = 0;
            if (v11 < 0) v11 = 0;
        }
        y0[o] = rf_sat8(v00);
        y1[o] = rf_sat8(v01);
        y0[o + 1] = rf_sat8(v10);
        y1[o + 1] = rf_sat8(v11);
    }
}

RF_HOT void rf_linear2_i8_acc16(const int8_t *x, const int8_t *W,
                                const int32_t *b, const int32_t *M,
                                const uint8_t *s, int K, int O, int16_t *res0,
                                int16_t *res1) {
    for (int o = 0; o < O; o += 2) {
        int32_t a00 = b[o], a01 = b[o];
        int32_t a10 = b[o + 1], a11 = b[o + 1];
        rf_dot2x2_i8(W + (size_t)o * K, x, K, &a00, &a01, &a10, &a11);
        res0[o] = rf_sat16((int64_t)res0[o] + rf_rq(a00, M[o], s[o]));
        res1[o] = rf_sat16((int64_t)res1[o] + rf_rq(a01, M[o], s[o]));
        res0[o + 1] = rf_sat16((int64_t)res0[o + 1] +
                               rf_rq(a10, M[o + 1], s[o + 1]));
        res1[o + 1] = rf_sat16((int64_t)res1[o + 1] +
                               rf_rq(a11, M[o + 1], s[o + 1]));
    }
}

RF_HOT void rf_conv3x3_i8_rows(const int8_t *x, int H, int W, int C, const int8_t *w,
                        const int32_t *b, const int32_t *M, const uint8_t *s,
                        int O, int relu, int stride, int upsample_in,
                        int out_u8, int8_t *y, int16_t *res16, int yo0,
                        int yo1) {
    int Wi = upsample_in ? 2 * W : W;
    int Wo = Wi / stride;
    for (int yo = yo0; yo < yo1; yo++) {
        for (int xo = 0; xo < Wo; xo++) {
            for (int o = 0; o < O; o++) {
                int32_t acc = b[o];
                const int8_t *wo = w + (size_t)o * 9 * C;
                for (int dy = 0; dy < 3; dy++) {
                    for (int dx = 0; dx < 3; dx++) {
                        int yi = yo * stride + dy - 1;
                        int xi = xo * stride + dx - 1;
                        if (upsample_in) {
                            yi >>= 1; /* -1 stays -1: pad check catches it */
                            xi >>= 1;
                        }
                        if (yi < 0 || yi >= H || xi < 0 || xi >= W) continue;
                        acc = rf_dot_i8(wo + (dy * 3 + dx) * C,
                                        x + ((size_t)yi * W + xi) * C, C, acc);
                    }
                }
                int64_t v = rf_rq(acc, M[o], s[o]);
                size_t oi = ((size_t)yo * Wo + xo) * O + o;
                if (res16) {
                    res16[oi] = rf_sat16((int64_t)res16[oi] + v);
                } else if (out_u8) {
                    v += 128;
                    ((uint8_t *)y)[oi] = (uint8_t)(v > 255 ? 255 : (v < 0 ? 0 : v));
                } else {
                    if (relu && v < 0) v = 0;
                    y[oi] = rf_sat8(v);
                }
            }
        }
    }
}

void rf_conv3x3_i8(const int8_t *x, int H, int W, int C, const int8_t *w,
                   const int32_t *b, const int32_t *M, const uint8_t *s, int O,
                   int relu, int stride, int upsample_in, int out_u8,
                   int8_t *y, int16_t *res16) {
    int Ho = (upsample_in ? 2 * H : H) / stride;
    rf_conv3x3_i8_rows(x, H, W, C, w, b, M, s, O, relu, stride, upsample_in,
                       out_u8, y, res16, 0, Ho);
}

RF_HOT void rf_requant_i16_to_i8(const int16_t *x, int n, int32_t M, uint8_t s,
                          int8_t *y) {
    for (int i = 0; i < n; i++) y[i] = rf_sat8(rf_rq(x[i], M, s));
}

RF_HOT void rf_rmsnorm_i16_to_i8(const int16_t *x, const int16_t *G, const int16_t *B,
                          int K, int8_t *y) {
    int64_t ss = 0;
    for (int k = 0; k < K; k++) ss += (int64_t)x[k] * x[k];
    /* fractional-precision rsqrt (Q8): a8 = 256*sqrt(ss/K). Legacy integer
     * isqrt quantized the norm scale to ~1/rms granularity (1.5-3% at the
     * small magnitudes of early-block residuals / qk-norm int8 vectors) --
     * a top int8 error source. Q8 shrinks it 256x; (ss<<16)/K avoids the
     * mean-square floor. No overflow: ss<=K*32767^2 -> ss<<16 < 2^53, and
     * x*R*G <= sqrt(K)*2^30*32767 < 2^49 (|x| <= rms*sqrt(K)). */
    uint32_t a8 = rf_isqrt64(((uint64_t)ss << 16) / (uint64_t)K);
    int64_t R = ((int64_t)1 << (RMS_R_SHIFT + 8)) / (a8 ? a8 : 1);
    for (int k = 0; k < K; k++) {
        int64_t p = (int64_t)x[k] * R * G[k] +
                    ((int64_t)B[k] << (RMS_OUT_SHIFT - RMS_B_Q));
        y[k] = rf_sat8((p + ((int64_t)1 << (RMS_OUT_SHIFT - 1))) >> RMS_OUT_SHIFT);
    }
}

RF_HOT void rf_softmax_i32_to_i8(const int32_t *scores, int N, int32_t M_sm,
                          uint8_t s_sm, const uint16_t *explut, int lut_max,
                          int8_t *p) {
    int32_t m = scores[0];
    for (int i = 1; i < N; i++)
        if (scores[i] > m) m = scores[i];
    uint32_t sum = 0;
    uint16_t e[512];
    for (int i = 0; i < N; i++) {
        int64_t idx = rf_rq((int64_t)m - scores[i], M_sm, s_sm);
        if (idx < 0) idx = 0;
        if (idx > lut_max) idx = lut_max;
        e[i] = explut[idx];
        sum += e[i];
    }
    /* round-to-nearest: floor lost up to 1 count/entry, downscaling diffuse
     * attention rows to ~95/127 systematically (top int8 error source). */
    for (int i = 0; i < N; i++)
        p[i] = (int8_t)((((uint32_t)e[i] * 127u) + sum / 2u) / sum);
}

RF_HOT void rf_lut_i8(const int8_t *x, int n, const int8_t *lut, int8_t *y) {
    for (int i = 0; i < n; i++) y[i] = lut[(uint8_t)(x[i] + 128)];
}

__attribute__((weak)) int rf_core_id(void) { return 0; }

RF_HOT int rf_compact_i8(const int8_t *x, int n, uint16_t *idx, int8_t *val) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (x[i]) {
            idx[m] = (uint16_t)i;
            val[m] = x[i];
            m++;
        }
    }
    return m;
}

RF_HOT void rf_axpy_acc16_sp(const uint16_t *idx, const int8_t *val, int m,
                             const int8_t *Wt, int O, const int32_t *b,
                             const int32_t *M, const uint8_t *s, int16_t *res) {
    for (int ob = 0; ob < O; ob += 8) {
        int32_t a[8];
        for (int j = 0; j < 8; j++) a[j] = b[ob + j];
        for (int i = 0; i < m; i++)
            rf_axpy8_i8(Wt + (size_t)idx[i] * O + ob, val[i], a);
        for (int j = 0; j < 8; j++)
            res[ob + j] = rf_sat16((int64_t)res[ob + j] +
                                   rf_rq(a[j], M[ob + j], s[ob + j]));
    }
}

/* incremental form for streamed output: seed with RF_CRC32_INIT, finalize
 * with ^RF_CRC32_INIT; rf_crc32(p, n) == that chain over one buffer */
uint32_t rf_crc32_acc(uint32_t crc, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++)
            crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320u : 0u);
    }
    return crc;
}

uint32_t rf_crc32(const uint8_t *data, size_t len) {
    return rf_crc32_acc(0xFFFFFFFFu, data, len) ^ 0xFFFFFFFFu;
}
