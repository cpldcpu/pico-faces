/* Parsed model: pointers into the (flash/malloc'd) model.bin blob.
 * Layout must match quant/export.py exactly (format v3). */
#ifndef RF_MODEL_H
#define RF_MODEL_H

#include <stddef.h>
#include <stdint.h>

/* Angle include: searched via -I order only, so a per-model generated
 * rf_cfg.h placed FIRST on the include path overrides the engine default
 * (a quoted include would always find the sibling default first). */
#include <rf_cfg.h>

/* rf_cfg.h files generated before the 256 head existed lack these */
#ifndef RF_HIRES
#define RF_HIRES 0
#define RF_HIRES_CMID 0
#define RF_OUT_HW RF_IMG_HW
#endif

#define RF_MAGIC 0x35325246u /* 'RF25' */
#define RF_W_MAX 4 /* max classifier-free-guidance table sets (format v4) */
#define RF_MAX_HIRES 2 /* hires head layers (format v5): conv+ReLU, delta conv */

typedef struct {
    const int16_t *G1, *B1, *G2, *B2;
    const int32_t *Mproj;
    const uint8_t *sproj; /* v<8: u8[dim]; v8: NULL (scalar in sproj_s) */
    const int32_t *Mfc2;
    const uint8_t *sfc2;  /* v<8: u8[dim]; v8: NULL (scalar in sfc2_s) */
    /* v8: fold.renorm_Ms aligned the mantissas to one shift per entry */
    uint8_t sproj_s, sfc2_s;
} rf_stepblk_t;

typedef struct {
    int has_attn;      /* v6: 0 = attention branch dropped (weights absent) */
    const int8_t *Wqkv;
    const int32_t *bqkv, *Mqkv;
    const uint8_t *sqkv;
    const int16_t *Gq, *Gk; /* [heads][head_dim] */
    int32_t M_sm;           /* legacy (<v7) scalar softmax scale */
    uint8_t s_sm;
    const int32_t *M_smh;   /* v7: per-head softmax scales (NULL on <v7) */
    const uint8_t *s_smh;
    const int32_t *M_att;
    const uint8_t *s_att;
    const int8_t *Wproj;
    const int32_t *bproj;
    const int8_t *Wfc1;
    const int32_t *bfc1, *Mfc1;
    const uint8_t *sfc1;
    const int8_t *act_lut;   /* legacy (<v7) LUT activation; NULL on v7 */
    /* v7 relu2 square-requant: per-channel h1 requant of relu(fc1_acc)^2>>12;
     * Mfc1/sfc1/act_lut are absent from the blob (NULL) when these are set */
    const int32_t *M_actq;
    const uint8_t *s_actq;
    const int8_t *Wfc2;
    const int32_t *bfc2;
} rf_blk_t;

typedef struct {
    uint32_t C, O, flags; /* flags: up | relu<<1 | u8<<2 | wt<<3 */
    const int8_t *W;
    const int32_t *b, *M;
    const uint8_t *s;
} rf_declayer_t;

typedef struct {
    uint32_t K, dim, depth, heads, tokens, zch, zhw, patch, pd;
    uint32_t n_cond, img_ch;
    const uint16_t *explut;
    int lut_max; /* last exp-LUT index: 255 (<v7) or 511 (v7, 1/64 grid) */
    const int16_t *pos;
    const int32_t *M_zin;
    const uint8_t *s_zin;
    /* adaLN tables per conditioning set (class or null); weights are shared */
    rf_stepblk_t step[RF_COND][RF_K_MAX][RF_DEPTH];
    const int16_t *Gf[RF_COND][RF_K_MAX], *Bf[RF_COND][RF_K_MAX];
    const int32_t *M_v[RF_K_MAX]; /* folded -dt: class-independent */
    const uint8_t *s_v[RF_K_MAX];
    /* classifier-free guidance (v4): per w, per-pass rescaled Euler folds.
     * cond pass: w*-dt; null pass: (1-w)*-dt. n_w = 0 on v3 models. */
    uint32_t n_w;
    uint32_t w_q8[RF_W_MAX]; /* w * 256, protocol/display only */
    const int32_t *M_v_c[RF_W_MAX][RF_K_MAX];
    const uint8_t *s_v_c[RF_W_MAX][RF_K_MAX];
    const int32_t *M_v_n[RF_W_MAX][RF_K_MAX];
    const uint8_t *s_v_n[RF_W_MAX][RF_K_MAX];
    const int8_t *W_emb;
    const int32_t *b_emb, *M_emb;
    const uint8_t *s_emb;
    rf_blk_t blk[RF_DEPTH];
    const int8_t *W_final;
    const int32_t *b_final;
    int32_t M_zdec;
    uint8_t s_zdec;
    uint32_t n_dec;
    rf_declayer_t dec[RF_MAX_DEC];
    /* v5: ESPCN hires head (0 or 2 layers). hires[0] = conv+ReLU tap of the
     * last decoder feature map, hires[1] = linear conv whose i8 output is a
     * pixel-unit delta, depth-to-spaced 2x2 onto the NN-x2 upsampled image. */
    uint32_t n_hires;
    rf_declayer_t hires[RF_MAX_HIRES];
} rf_model_t;

/* returns 0 on success */
int rf_model_load(const uint8_t *blob, size_t len, rf_model_t *m);

/* full generation: seed -> RF_IMG_HW^2 x RF_IMG_CH u8 image (HWC).
 * k_steps: Euler step count; must be K divided by a power of 2 (0 = full K).
 * The step tables are strided and the folded Euler dt rescaled by an exact
 * shift, so every k_steps choice is still bit-exact vs int_sim.
 * cond: table-set index (class, or n_classes = null); clamped to 0 if out
 * of range. Golden convention everywhere: cond = seed % n_cond.
 * w_idx: guidance table index (< 0 or >= n_w, or cond = null: plain pass).
 * Guided steps run the body twice (cond then null tables) from the same
 * pre-step z; both passes accumulate into z via their rescaled M_v.
 * CFG golden convention: w_idx = seed % (n_w+1) - 1.
 * taps: optional (k_steps+1) x tokens x pd int16 z-state buffer. */
void rf_generate(const rf_model_t *m, uint64_t seed, int k_steps, int cond,
                 int w_idx, uint8_t *img, int16_t *taps);

#if RF_HIRES
/* hires models (n_hires == 2): stream the 256 output. Must run immediately
 * after rf_generate — the head taps the decoder's last feature map, which is
 * still resident in the arena rf_decode ping-ponged it into. The sink
 * receives every output row y (0..RF_OUT_HW-1) in order, RF_OUT_HW *
 * RF_IMG_CH bytes each. The full 256 image never exists in memory. */
void rf_hires(const rf_model_t *m, const uint8_t *img128,
              void (*sink)(int y, const uint8_t *row, void *user),
              void *user);
#endif

#endif
