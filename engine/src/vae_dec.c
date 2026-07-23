/* VAE decoder: unpatchify -> requant -> conv chain (ping-pong in rf_arena).
 * Mirrors quant/int_sim.py IntSim.decode; values are identical, only the
 * accumulation strategy differs (int32 order-independence).
 *
 * Mid layers (flag bit 3, weights stored [3][3][C][O]) use a sparse path:
 * post-ReLU inputs are ~50% exact zeros, so each needed input row is
 * compacted (per-pixel nonzero channel lists) and the conv gathers only
 * nonzeros, AXPY-ing across the contiguous output dim. First layer (dense
 * latent input) and final u8 layer (O=1) stay on the dense kernel. */
#include "rf_model.h"
#include "rf_ops.h"

#define MAX_ROW_NZ RF_DEC_NZ_MAX /* max W_in*C over the sparse layers */
#ifdef RF_DEC_O_MAX
#define MAX_SP_O RF_DEC_O_MAX     /* max out-channels over the sparse layers */
#else
#define MAX_SP_O 64               /* pre-RF_DEC_O_MAX cfgs: legacy decoder <=64 */
#endif
#define MAX_SP_W RF_DEC_W_MAX

typedef struct {
    int yi;                     /* which input row is cached (-2 = none) */
    uint16_t start[MAX_SP_W + 1];
    uint8_t ch[MAX_ROW_NZ];
    int8_t val[MAX_ROW_NZ];
} rowcomp_t;

/* rowbuf = [2 cores][3 rows] of row-compaction scratch. It is phase-exclusive
 * with the DiT's qkv/attn buffers (those are idle during rf_decode), so we
 * overlay it there (rf_dec_scratch) rather than spend fresh .bss -- the fat
 * decoders (128ch mid layers -> MAX_ROW_NZ 2048) would otherwise blow the 512KB
 * SRAM by ~2KB. Same phase-exclusive aliasing the hires head uses on
 * rf_dit_noise. The _Static_assert fires (clear compile error, not silent
 * corruption) if a future decoder's rowbuf ever exceeds that block. */
extern int8_t *const rf_dec_scratch;
#define rf_dec_scratch_bytes (RF_TOKENS * 4 * RF_DIM)  /* == sizeof dit_qa */
#define rowbuf ((rowcomp_t(*)[3])rf_dec_scratch)
_Static_assert(rf_dec_scratch_bytes >= (int)(2 * 3 * sizeof(rowcomp_t)),
               "rowbuf exceeds DiT qkv/attn scratch -- widen the alias");

typedef struct {
    const int8_t *x;
    const rf_declayer_t *L;
    int H, W;
    int relu, up, u8, wt;
    int8_t *dst;
} dctx_t;

static void compact_row(const dctx_t *c, rowcomp_t *rc, int yi) {
    const int C = (int)c->L->C;
    const int8_t *row = c->x + (size_t)yi * c->W * C;
    int m = 0;
    for (int xi = 0; xi < c->W; xi++) {
        rc->start[xi] = (uint16_t)m;
        for (int ch = 0; ch < C; ch++) {
            int8_t v = row[xi * C + ch];
            if (v) {
                rc->ch[m] = (uint8_t)ch;
                rc->val[m] = v;
                m++;
            }
        }
    }
    rc->start[c->W] = (uint16_t)m;
    rc->yi = yi;
}

RF_HOT static void conv_rows_sparse(int y0, int y1, void *p) {
    dctx_t *c = p;
    const rf_declayer_t *L = c->L;
    const int C = (int)L->C, O = (int)L->O;
    const int Wo = c->up ? 2 * c->W : c->W;
    rowcomp_t *slots = rowbuf[rf_core_id()];
    for (int i = 0; i < 3; i++) slots[i].yi = -2;

    for (int yo = y0; yo < y1; yo++) {
        for (int xo = 0; xo < Wo; xo++) {
            int32_t acc[MAX_SP_O];
            for (int o = 0; o < O; o++) acc[o] = L->b[o];
            for (int dy = 0; dy < 3; dy++) {
                int yi = yo + dy - 1;
                if (c->up) yi >>= 1;
                if (yi < 0 || yi >= c->H) continue;
                rowcomp_t *rc = &slots[yi % 3];
                if (rc->yi != yi) compact_row(c, rc, yi);
                for (int dx = 0; dx < 3; dx++) {
                    int xi = xo + dx - 1;
                    if (c->up) xi >>= 1;
                    if (xi < 0 || xi >= c->W) continue;
                    const int8_t *wtap = L->W + ((size_t)(dy * 3 + dx) * C) * O;
                    for (int i = rc->start[xi]; i < rc->start[xi + 1]; i++) {
                        const int8_t *w = wtap + (size_t)rc->ch[i] * O;
                        int32_t v = rc->val[i];
                        for (int o = 0; o < O; o += 8)
                            rf_axpy8_i8(w + o, v, acc + o);
                    }
                }
            }
            int8_t *out = c->dst + ((size_t)yo * Wo + xo) * O;
            for (int o = 0; o < O; o++) {
                int64_t v = rf_rq(acc[o], L->M[o], L->s[o]);
                if (v < 0) v = 0; /* sparse layers are always conv+ReLU */
                out[o] = rf_sat8(v);
            }
        }
    }
}

static void conv_rows_dense(int y0, int y1, void *p) {
    dctx_t *c = p;
    rf_conv3x3_i8_rows(c->x, c->H, c->W, (int)c->L->C, c->L->W, c->L->b,
                       c->L->M, c->L->s, (int)c->L->O, c->relu, 1, c->up,
                       c->u8, c->dst, NULL, y0, y1);
}

/* the final (u8) layer's input = the last 128^2 feature map; the hires head
 * (engine/src/hires.c) taps it from the arena right after rf_generate */
const int8_t *rf_dec_feat;

/* unpatchify staging (4 KB, decode entry only). Non-static: the hires head
 * reuses it as conv scratch -- phase-exclusive, like the fb/arena alias */
int16_t rf_dec_zhwc[RF_ZHW * RF_ZHW * RF_ZCH] RF_ALIGN4;

void rf_decode(const rf_model_t *m, const int16_t z_tok[RF_TOKENS][RF_PD],
               uint8_t *img) {
    /* unpatchify: row-major patch grid, token feature order (c, py, px) */
    int16_t *zhwc = rf_dec_zhwc;
    const int P = RF_PATCH, G = RF_ZHW / RF_PATCH;
    for (int y = 0; y < RF_ZHW; y++)
        for (int x = 0; x < RF_ZHW; x++)
            for (int c = 0; c < RF_ZCH; c++)
                zhwc[(y * RF_ZHW + x) * RF_ZCH + c] =
                    z_tok[(y / P) * G + (x / P)]
                         [c * P * P + (y % P) * P + (x % P)];

    rf_requant_i16_to_i8(zhwc, RF_ZHW * RF_ZHW * RF_ZCH, m->M_zdec, m->s_zdec,
                         rf_arena[0]);

    int H = RF_ZHW, W = RF_ZHW;
    const int8_t *in = rf_arena[0];
    int8_t *out = rf_arena[1];
    for (uint32_t i = 0; i < m->n_dec; i++) {
        const rf_declayer_t *L = &m->dec[i];
        dctx_t c = {in, L, H, W, (int)((L->flags >> 1) & 1),
                    (int)(L->flags & 1), (int)((L->flags >> 2) & 1),
                    (int)((L->flags >> 3) & 1), NULL};
        c.dst = c.u8 ? (int8_t *)img : out;
        if (c.u8) rf_dec_feat = in;
        int Ho = c.up ? 2 * H : H;
        rf_par_for(Ho, c.wt ? conv_rows_sparse : conv_rows_dense, &c);
        if (c.up) {
            H *= 2;
            W *= 2;
        }
        in = c.dst;
        out = (in == (const int8_t *)rf_arena[0]) ? rf_arena[1] : rf_arena[0];
    }
}
