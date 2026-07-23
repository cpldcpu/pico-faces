"""THE specification of every integer operation in the deployed pipeline.

engine/src/kernels_ref.c and kernels_m33.c must match these functions
bit-for-bit. All intermediates fit in int64; shifts on negatives are
arithmetic (numpy int64 >> is arithmetic; gcc on x86/ARM likewise).

Conventions:
- int8 activations are symmetric, saturated to [-127, 127] (never -128)
- int16 residual values saturate to [-32767, 32767]
- requantization: p = acc * M (int64); y = (p + (1 << (s-1))) >> s; saturate.
  Rounding is half-up (toward +inf) everywhere.
- RMSNorm constants: R is Q30, gains are Q14, biases are Q7 in output units.
"""
import numpy as np

I64 = np.int64

RMS_R_SHIFT = 30   # reciprocal-rms fixed point
RMS_G_Q = 8        # gain fixed point: folded (1+scale)/s_out reaches ~64, so
                   # Q8 covers +-128 in int16 with 1/256 relative precision
RMS_B_Q = 7        # bias fixed point (in output-scale units)
RMS_OUT_SHIFT = RMS_R_SHIFT + RMS_G_Q  # 38


def sat8(x):
    return np.clip(x, -127, 127).astype(np.int8)


def sat16(x):
    return np.clip(x, -32767, 32767).astype(np.int16)


def requant(acc, M, s):
    """acc int32/int64 array, M int32 (scalar or per-channel), s uint8 shift
    (scalar or per-channel). Returns int64 (caller saturates)."""
    p = acc.astype(I64) * I64(M) if np.isscalar(M) else acc.astype(I64) * M.astype(I64)
    return (p + (I64(1) << (s - I64(1)))) >> s


def isqrt64(v):
    """floor(sqrt(v)) for uint64 v, restoring shift-subtract (matches C loop)."""
    v = int(v)
    r, bit = 0, 1 << 62
    while bit > v:
        bit >>= 2
    while bit:
        if v >= r + bit:
            v -= r + bit
            r = (r >> 1) + bit
        else:
            r >>= 1
        bit >>= 2
    return r


def linear_i8(x, W, b, M, s, relu=False):
    """x: i8[K], W: i8[O,K], b: i32[O], M: i32[O], s: u8[O] -> i8[O]."""
    acc = W.astype(I64) @ x.astype(I64) + b.astype(I64)
    y = requant(acc, M.astype(I64), s.astype(I64))
    return sat8(np.maximum(y, 0)) if relu else sat8(y)


def linear_i8_acc16(x, W, b, M, s, res):
    """Same matmul, but requantized into residual units and added to res (i16[O]).
    Used by branch-final projections (adaLN gate folded into M)."""
    acc = W.astype(I64) @ x.astype(I64) + b.astype(I64)
    y = requant(acc, M.astype(I64), s.astype(I64))
    return sat16(res.astype(I64) + y)


def conv3x3_i8(x, W, b, M, s, relu=True, stride=1, upsample_in=False,
               acc16_res=None, out_u8=False):
    """x: i8[H,W,C] (HWC), W: i8[O,3,3,C], b: i32[O], per-channel M/s.
    Zero padding (symmetric quant -> 0 is exact). upsample_in fuses nearest-
    neighbor 2x: output pixel (y,x) reads input pixel ((y+dy-1)>>1, (x+dx-1)>>1).
    acc16_res: if given (i16[Ho,Wo,O]), requantized output is added into it
    (residual mode); relu ignored in that mode.
    out_u8: final image layer - out = clip(rq + 128, 0, 255) as uint8."""
    H, Wd, C = x.shape
    O = W.shape[0]
    Hi, Wi = (2 * H, 2 * Wd) if upsample_in else (H, Wd)
    Ho, Wo = Hi // stride, Wi // stride
    out = np.zeros((Ho, Wo, O), dtype=np.uint8 if out_u8 else np.int8)
    xw = x.astype(I64)
    for yo in range(Ho):
        for xo in range(Wo):
            acc = b.astype(I64).copy()
            for dy in range(3):
                for dx in range(3):
                    yi, xi = yo * stride + dy - 1, xo * stride + dx - 1
                    if upsample_in:
                        # map upsampled-grid tap back to source pixel; -1 >> 1
                        # stays -1 (arithmetic shift) and fails the pad check
                        yi, xi = yi >> 1, xi >> 1
                    if 0 <= yi < H and 0 <= xi < Wd:
                        acc += W[:, dy, dx, :].astype(I64) @ xw[yi, xi]
            y = requant(acc, M.astype(I64), s.astype(I64))
            if acc16_res is not None:
                acc16_res[yo, xo] = sat16(acc16_res[yo, xo].astype(I64) + y)
            elif out_u8:
                out[yo, xo] = np.clip(y + 128, 0, 255).astype(np.uint8)
            else:
                out[yo, xo] = sat8(np.maximum(y, 0) if relu else y)
    return acc16_res if acc16_res is not None else out


def requant_i16_to_i8(x, M, s):
    """x: i16[...] -> i8 at a new scale (per-tensor M, s)."""
    return sat8(requant(x.astype(I64), I64(M), I64(s)))


def rmsnorm_i16_to_i8(x, G, B, k_div=None):
    """x: i16[K] (any fixed-point scale - RMS cancels it), G: i16[K] Q14 folded
    (1+adaLN_scale) gain in output units, B: i16[K] Q7 folded adaLN shift in
    output units -> i8[K].

    ss = sum(x^2); ms = ss // K; a = isqrt(ms); R = 2^30 // max(a,1)
    y  = sat8( (x*R*G + B<<(44-7) + 2^43) >> 44 )
    """
    K = k_div or len(x)
    xw = x.astype(I64)
    ss = int((xw * xw).sum())
    # fractional-precision rsqrt (Q8): a8 = 256*sqrt(ss/K). The legacy integer
    # isqrt quantized the norm scale to ~1/rms granularity -- at small residual
    # magnitudes (early blocks: rms ~30 int16 units; qk-norm: int8 head vectors)
    # that is a 1.5-3% multiplicative error per token per norm, one of the two
    # largest int8 error sources (error budget 2026-07-15). Q8 shrinks the
    # granularity 256x; (ss<<16)//K also avoids the mean-square floor.
    a8 = isqrt64((ss << 16) // K)
    R = (1 << (RMS_R_SHIFT + 8)) // max(a8, 1)
    n = xw * I64(R)
    p = n * G.astype(I64) + (B.astype(I64) << (RMS_OUT_SHIFT - RMS_B_Q))
    y = (p + (I64(1) << (RMS_OUT_SHIFT - 1))) >> RMS_OUT_SHIFT
    return sat8(y)


def softmax_i32_to_i8(scores, M_sm, s_sm, explut):
    """scores: i32[N] raw q.k accumulators; explut: u16[256] Q15 of exp(-x).
    idx = min(255, requant(max-score - s_i, M_sm, s_sm)); p = e*127 // sum."""
    d = (scores.max().astype(I64) - scores.astype(I64))
    idx = np.minimum(requant(d, I64(M_sm), I64(s_sm)),
                     len(explut) - 1).astype(np.int64)
    idx = np.maximum(idx, 0)
    e = explut[idx].astype(I64)
    S = int(e.sum())
    # round-to-nearest division: plain floor loses up to 1 count per entry, so
    # diffuse attention rows summed to ~95/127 instead of 127 -- a systematic
    # attention-output downscale (the largest single int8 error source).
    return ((e * 127 + S // 2) // S).astype(np.int8)  # in [0,127], scale 1/127


def lut_i8(x, lut):
    """x: i8[...], lut: i8[256] indexed by (x + 128)."""
    return lut[(x.astype(np.int16) + 128).astype(np.uint8)]


# ------------------------------------------------------------------ PRNG/noise

PCG_MULT = 6364136223846793005
PCG_INC = 1442695040888963407  # default stream (odd)
MASK64 = (1 << 64) - 1


def pcg32_init(seed):
    state = 0
    state = (state * PCG_MULT + PCG_INC) & MASK64
    state = (state + seed) & MASK64
    state = (state * PCG_MULT + PCG_INC) & MASK64
    return state


def pcg32_next(state):
    """Returns (new_state, u32). PCG-XSH-RR."""
    x = state
    count = x >> 59
    x ^= x >> 18
    out = (x >> 27) & 0xFFFFFFFF
    out = ((out >> count) | (out << ((32 - count) & 31))) & 0xFFFFFFFF
    return (state * PCG_MULT + PCG_INC) & MASK64, out


def gaussian_clt12(state, n):
    """n samples of ~N(0,1) in Q12 int (sigma = 4096): sum of 12 12-bit
    uniforms minus 24576. Returns (state, i32[n])."""
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        acc = 0
        for _ in range(12):
            state, u = pcg32_next(state)
            acc += u >> 20
        out[i] = acc - 24576
    return state, out


def crc32(data, crc=0xFFFFFFFF):
    """Standard CRC-32 (IEEE, reflected, poly 0xEDB88320), no table."""
    for byte in bytes(data):
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF
