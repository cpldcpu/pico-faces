/* 256 hires head (model.bin v5): ESPCN pixel-shuffle on the frozen decoder.
 * Mirrors quant/int_sim.py IntSim.decode_hires byte-for-byte.
 *
 * h0 = conv3x3+ReLU (8 -> c_mid) over the decoder's last feature map, still
 * resident in the arena rf_decode ping-ponged it into; h1 = linear conv
 * (c_mid -> 12) whose int8 output is a delta in pixel LSB units. Depth-to-
 * space 2x2 + a saturating add onto the NN-x2 upsampled u8 image yield the
 * 256 output, streamed through a per-row sink: the full 256 image never
 * exists in memory.
 *
 * Work runs in stripes of R source rows so both convs go through the dense
 * conv3x3 row kernel under rf_par_for (dual-core + SMLAD, same as the
 * decoder). The h0 window carries its last two rows into the next stripe,
 * so every h0 row is computed exactly once. Scratch (~34 KB) is static and
 * deliberately NOT in the arena: on the VGA build the framebuffer aliases
 * BOTH arenas, and the sink dithers display rows while this pass still
 * reads the feature map — the dither's write cursor trails the feature-map
 * read tail by >110 KB at every row, but only because nothing else of ours
 * lives there. */
#include <string.h>

#include "rf_model.h"
#include "rf_ops.h"

#if RF_HIRES

#define R 2                /* stripe height in source rows; RF_IMG_HW % R == 0.
                            * R sets the static scratch (the firmware has ~12 KB
                            * of slack) and the rows-per-par_for granularity. */
#define HW RF_IMG_HW       /* 128 */
#define CM RF_HIRES_CMID   /* h0 channels */
#define DC (4 * RF_IMG_CH) /* delta channels: (c, dy, dx) 2x2 blocks */

extern const int8_t *rf_dec_feat; /* set by rf_decode */

/* Scratch aliases two phase-exclusive staging buffers instead of new .bss
 * (the firmware has none to spare): the DiT's noise buffer (dead once the
 * first step runs) and the decoder's unpatchify buffer (dead once the conv
 * chain starts). Accessed through int8_t* only -- char aliasing is defined. */
extern int32_t rf_dit_noise[RF_TOKENS * RF_PD];
extern int16_t rf_dec_zhwc[RF_ZHW * RF_ZHW * RF_ZCH];
#define h0win ((int8_t *)rf_dit_noise) /* stripe window incl. halo rows */
#define delta ((int8_t *)rf_dec_zhwc)
_Static_assert(sizeof rf_dit_noise >= (R + 2) * HW * CM, "h0win scratch");
_Static_assert(sizeof rf_dec_zhwc >= R * HW * DC, "delta scratch");

typedef struct {
    const rf_declayer_t *L;
    const int8_t *x;
    int H;  /* input plane height the kernel sees */
    int y0; /* kernel yo of the first row this call computes */
    int8_t *dst; /* where that first row lands */
    int relu;
} hctx_t;

static void conv_rows(int i0, int i1, void *p) {
    hctx_t *c = p;
    const rf_declayer_t *L = c->L;
    /* the kernel indexes dst by its absolute yo; rebase so yo == c->y0
     * lands at c->dst (flat address space, in-bounds for every row written) */
    int8_t *dst = (int8_t *)((uintptr_t)c->dst -
                             (uintptr_t)((size_t)c->y0 * HW * L->O));
    rf_conv3x3_i8_rows(c->x, c->H, HW, (int)L->C, L->W, L->b, L->M, L->s,
                       (int)L->O, c->relu, 1, 0, 0, dst, NULL, c->y0 + i0,
                       c->y0 + i1);
}

void rf_hires(const rf_model_t *m, const uint8_t *img128,
              void (*sink)(int y, const uint8_t *row, void *user),
              void *user) {
    static uint8_t row[RF_OUT_HW * RF_IMG_CH];
    if (m->n_hires != 2) return;

    for (int r0 = 0; r0 < HW; r0 += R) {
        int r1 = r0 + R;              /* stripe covers source rows [r0, r1) */
        int a = r0 ? r0 - 1 : 0;      /* first h0 row in the window */
        int b = r1 + 1 > HW ? HW : r1 + 1; /* one past the last h0 row */

        hctx_t hc = {&m->hires[0], rf_dec_feat, HW, 0, h0win, 1};
        if (r0) { /* carry rows r0-1, r0 from the previous window's tail
                   * (memmove: for r0 == R src and dst overlap by one row) */
            int prev = (r0 == R) ? R + 1 : R + 2; /* rows in prev window */
            memmove(h0win, h0win + (size_t)(prev - 2) * HW * CM,
                    (size_t)2 * HW * CM);
            hc.y0 = r0 + 1; /* rows <= r0 already present */
            hc.dst = h0win + (size_t)2 * HW * CM;
        }
        rf_par_for(b - hc.y0, conv_rows, &hc);

        hctx_t hd = {&m->hires[1], h0win, b - a, r0 - a, delta, 0};
        rf_par_for(r1 - r0, conv_rows, &hd);

        for (int y = r0; y < r1; y++) {
            const int8_t *d = delta + (size_t)(y - r0) * HW * DC;
            const uint8_t *base = img128 + (size_t)y * HW * RF_IMG_CH;
            for (int sub = 0; sub < 2; sub++) { /* output rows 2y, 2y+1 */
                for (int x = 0; x < HW; x++) {
                    const int8_t *dp = d + x * DC + 2 * sub;
                    const uint8_t *bp = base + x * RF_IMG_CH;
                    for (int ch = 0; ch < RF_IMG_CH; ch++) {
                        int v = bp[ch] + dp[4 * ch];
                        row[(2 * x) * RF_IMG_CH + ch] =
                            (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                        v = bp[ch] + dp[4 * ch + 1];
                        row[(2 * x + 1) * RF_IMG_CH + ch] =
                            (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                    }
                }
                sink(2 * y + sub, row, user);
            }
        }
    }
}

#endif /* RF_HIRES */
