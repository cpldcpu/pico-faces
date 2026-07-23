/* DiT backbone execution. Mirrors quant/int_sim.py IntSim.dit_step exactly:
 * same ops, same widths. Loop order across tokens/heads is free because ops
 * are pure and int32 addition is order-independent.
 *
 * Performance structure (v2):
 * - each layer's weights are copied from flash to the SRAM arena once per
 *   layer (the 16KB XIP cache cannot hold a 48-64KB matrix across a 64-token
 *   loop, so reading weights in place re-fetches them from flash per token)
 * - token loops run via rf_par_for (dual-core on device, serial on desktop);
 *   all callback writes are disjoint per token
 */
#include <string.h>

#include "rf_model.h"
#include "rf_ops.h"

void rf_decode(const rf_model_t *m, const int16_t z_tok[RF_TOKENS][RF_PD],
               uint8_t *img);

static int16_t z_tok[RF_TOKENS][RF_PD] RF_ALIGN4;
/* CFG: pre-step z snapshot -- both guided passes read it while their Euler
 * folds accumulate into z_tok (4 KB, the whole guidance SRAM tax) */
static int16_t z_prev[RF_TOKENS][RF_PD] RF_ALIGN4;
/* CLT-12 noise staging (8 KB, generation init only). Non-static: the hires
 * head reuses it as conv scratch -- phase-exclusive, like the fb/arena alias */
int32_t rf_dit_noise[RF_TOKENS * RF_PD] RF_ALIGN4;
static int16_t res[RF_TOKENS][RF_DIM] RF_ALIGN4;
static int8_t xa[RF_TOKENS][RF_DIM] RF_ALIGN4;
/* qkv+attn are grouped so the VAE decoder can borrow their combined storage as
 * row-compaction scratch (rf_dec_scratch): both are DiT-step-only and fully
 * idle during rf_decode -- the same phase-exclusive aliasing the hires head
 * uses on rf_dit_noise. Non-static + a stable base symbol let vae_dec.c overlay
 * rowbuf here instead of allocating its own 26 KB of .bss. */
static struct rf_dit_qa {
    int8_t qkv[RF_TOKENS][3 * RF_DIM];
    int8_t attn[RF_TOKENS][RF_DIM];
} dit_qa RF_ALIGN4;
#define qkv (dit_qa.qkv)
#define attn (dit_qa.attn)
int8_t *const rf_dec_scratch = (int8_t *)&dit_qa;
/* vae_dec.c computes RF_DEC_SCRATCH_BYTES = RF_TOKENS*4*RF_DIM at compile time
 * to decide the rowbuf overlay; keep this block exactly that big. */
_Static_assert(sizeof dit_qa == RF_TOKENS * 4 * RF_DIM, "rf_dec_scratch size");
/* v transposed [head][dim][token]: makes the p.v sum a contiguous dot */
static int8_t vT[RF_HEADS][RF_DIM / RF_HEADS][RF_TOKENS] RF_ALIGN4;

/* CFG precision fix: raw int32 final-layer velocity accumulator for the cond
 * pass, so the guided blend forms (v_cond - v_null) in int32 BEFORE the *w
 * scaling -- avoids catastrophic cancellation of two separately-requantized
 * int16 velocities at high w. 4 KB (64 tok x 16 x 4B), guided-only. */
static int32_t v_acc_c[RF_TOKENS][RF_PD] RF_ALIGN4;

static const int16_t zeros16[RF_DIM] = {0};

/* v8 scalar step-table shifts, expanded per block so the per-channel requant
 * kernels stay untouched. Written by dit_step before the par_for that reads
 * them (the par_for entry barrier publishes the stores to core1). */
static uint8_t sp_buf[RF_DIM], sf_buf[RF_DIM];

/* display hooks: 0 = idle/done, 1..k_steps = Euler steps, k_steps+1 = decode */
volatile uint8_t rf_progress = 0;
volatile uint8_t rf_progress_total = 5;
const int16_t *rf_z_state(void) { return &z_tok[0][0]; }
/* called after each progress change; IRQ-free display builds override it
 * to repaint the latent preview / progress bar (plain stores, no IRQs) */
__attribute__((weak)) void rf_step_hook(void) {}

/* Weight staging: fixed arena slots (rf_ops.h), prefetched one op ahead so
 * the copy runs while the previous op computes (rf_stage_start is async on
 * device). Consecutive ops never share a slot, and a slot is only restarted
 * after the par_for that reads it has returned. */

typedef struct {
    const rf_model_t *m;
    const rf_blk_t *blk;
    const rf_stepblk_t *st;
    /* per-channel requant shifts for the branch tails. v<8 blobs point into
     * the step table; v8 blobs store one scalar per entry (renorm_Ms), which
     * dit_step expands into sp_buf/sf_buf so the kernels stay unchanged. */
    const uint8_t *sproj, *sfc2;
    const int8_t *w;
    const int16_t (*zsrc)[RF_PD]; /* where the pass reads z (z_tok or z_prev) */
    const int32_t *Mv;            /* Euler fold for this pass */
    const uint8_t *sv;
    int k, cond;
    /* guided-blend (int32 CFG fix): mode 0 = normal accumulate; 1 = store raw
     * int32 acc to v_acc_c (cond pass); 2 = blend with v_acc_c + requant base
     * M_v into z_tok (null pass). wq = w in Q8. */
    int gmode;
    uint32_t wq;
} ctx_t;

/* All dense ranges walk token PAIRS (rf_linear2_*: each loaded weight and
 * activation word feeds two dot products - the layers are SRAM-traffic
 * bound). par_for halves of 64 tokens are always even; the single-token
 * tails are kept for generality. */
static void embed_range(int t0, int t1, void *p) {
    ctx_t *c = p;
    const rf_model_t *m = c->m;
    const int16_t(*z)[RF_PD] = c->zsrc;
    int t = t0;
    for (; t + 1 < t1; t += 2) {
        int8_t x_in[2][RF_PD] RF_ALIGN4;
        rf_requant_i16_to_i8(z[t], RF_PD, m->M_zin[c->k], m->s_zin[c->k],
                             x_in[0]);
        rf_requant_i16_to_i8(z[t + 1], RF_PD, m->M_zin[c->k],
                             m->s_zin[c->k], x_in[1]);
        rf_linear2_i8_acc16(x_in[0], c->w, m->b_emb, m->M_emb, m->s_emb,
                            RF_PD, RF_DIM, res[t], res[t + 1]);
    }
    for (; t < t1; t++) {
        int8_t x_in[RF_PD] RF_ALIGN4;
        rf_requant_i16_to_i8(z[t], RF_PD, m->M_zin[c->k], m->s_zin[c->k],
                             x_in);
        rf_linear_i8_acc16(x_in, c->w, m->b_emb, m->M_emb, m->s_emb, RF_PD,
                           RF_DIM, res[t]);
    }
}

static void qk_heads_one(const ctx_t *c, int t) {
    const int hd = RF_DIM / RF_HEADS;
    int16_t tmp[RF_DIM / RF_HEADS];
    for (int h = 0; h < RF_HEADS; h++) {
        for (int i = 0; i < hd; i++) tmp[i] = qkv[t][h * hd + i];
        rf_rmsnorm_i16_to_i8(tmp, c->blk->Gq + h * hd, zeros16, hd,
                             &qkv[t][h * hd]);
        for (int i = 0; i < hd; i++) tmp[i] = qkv[t][RF_DIM + h * hd + i];
        rf_rmsnorm_i16_to_i8(tmp, c->blk->Gk + h * hd, zeros16, hd,
                             &qkv[t][RF_DIM + h * hd]);
        for (int i = 0; i < hd; i++)
            vT[h][i][t] = qkv[t][2 * RF_DIM + h * hd + i];
    }
}

static void norm_qkv_range(int t0, int t1, void *p) {
    ctx_t *c = p;
    int t = t0;
    for (; t + 1 < t1; t += 2) {
        rf_rmsnorm_i16_to_i8(res[t], c->st->G1, c->st->B1, RF_DIM, xa[t]);
        rf_rmsnorm_i16_to_i8(res[t + 1], c->st->G1, c->st->B1, RF_DIM,
                             xa[t + 1]);
        rf_linear2_i8(xa[t], c->w, c->blk->bqkv, c->blk->Mqkv, c->blk->sqkv,
                      RF_DIM, 3 * RF_DIM, 0, qkv[t], qkv[t + 1]);
        qk_heads_one(c, t);
        qk_heads_one(c, t + 1);
    }
    for (; t < t1; t++) {
        rf_rmsnorm_i16_to_i8(res[t], c->st->G1, c->st->B1, RF_DIM, xa[t]);
        rf_linear_i8(xa[t], c->w, c->blk->bqkv, c->blk->Mqkv, c->blk->sqkv,
                     RF_DIM, 3 * RF_DIM, 0, qkv[t]);
        qk_heads_one(c, t);
    }
}

static void attn_range(int t0, int t1, void *p) {
    ctx_t *c = p;
    const rf_blk_t *blk = c->blk;
    const int hd = RF_DIM / RF_HEADS;
    for (int t = t0; t < t1; t++) {
        for (int h = 0; h < RF_HEADS; h++) {
            int32_t scores[RF_TOKENS];
            int8_t prow[RF_TOKENS];
            const int8_t *q = &qkv[t][h * hd];
            for (int j = 0; j < RF_TOKENS; j++)
                scores[j] = rf_dot_i8(&qkv[j][RF_DIM + h * hd], q, hd, 0);
            rf_softmax_i32_to_i8(scores, RF_TOKENS,
                                 blk->M_smh ? blk->M_smh[h] : blk->M_sm,
                                 blk->s_smh ? blk->s_smh[h] : blk->s_sm,
                                 c->m->explut, c->m->lut_max, prow);
            for (int i = 0; i < hd; i++) {
                int32_t acc = rf_dot_i8(vT[h][i], prow, RF_TOKENS, 0);
                attn[t][h * hd + i] = rf_sat8(rf_rq(
                    acc, blk->M_att[h * hd + i], blk->s_att[h * hd + i]));
            }
        }
    }
}

static void proj_range(int t0, int t1, void *p) {
    ctx_t *c = p;
    int t = t0;
    for (; t + 1 < t1; t += 2)
        rf_linear2_i8_acc16(attn[t], c->w, c->blk->bproj, c->st->Mproj,
                            c->sproj, RF_DIM, RF_DIM, res[t], res[t + 1]);
    for (; t < t1; t++)
        rf_linear_i8_acc16(attn[t], c->w, c->blk->bproj, c->st->Mproj,
                           c->sproj, RF_DIM, RF_DIM, res[t]);
}

static void mlp_tail_one(const ctx_t *c, const int8_t *wfc2t, int8_t *h1,
                         int t) {
    uint16_t nzi[4 * RF_DIM];
    int8_t nzv[4 * RF_DIM];
    if (c->blk->act_lut) /* legacy LUT; v7 h1 is already the final int8 */
        rf_lut_i8(h1, 4 * RF_DIM, c->blk->act_lut, h1);
    int m = rf_compact_i8(h1, 4 * RF_DIM, nzi, nzv);
    rf_axpy_acc16_sp(nzi, nzv, m, wfc2t, RF_DIM, c->blk->bfc2, c->st->Mfc2,
                     c->sfc2, res[t]);
}

static void mlp_range(int t0, int t1, void *p) {
    ctx_t *c = p;
    const int8_t *wfc1 = c->w;
    const int8_t *wfc2t = c->w + (size_t)4 * RF_DIM * RF_DIM; /* [K][O] */
    int t = t0;
    for (; t + 1 < t1; t += 2) {
        int8_t xm[2][RF_DIM] RF_ALIGN4, h1[2][4 * RF_DIM] RF_ALIGN4;
        rf_rmsnorm_i16_to_i8(res[t], c->st->G2, c->st->B2, RF_DIM, xm[0]);
        rf_rmsnorm_i16_to_i8(res[t + 1], c->st->G2, c->st->B2, RF_DIM, xm[1]);
        if (c->blk->M_actq) /* v7: square the raw acc, per-channel requant */
            rf_relu2sq2_i8(xm[0], wfc1, c->blk->bfc1, c->blk->M_actq,
                           c->blk->s_actq, RF_DIM, 4 * RF_DIM, h1[0], h1[1]);
        else
            rf_linear2_i8(xm[0], wfc1, c->blk->bfc1, c->blk->Mfc1,
                          c->blk->sfc1, RF_DIM, 4 * RF_DIM, 0, h1[0], h1[1]);
        mlp_tail_one(c, wfc2t, h1[0], t);
        mlp_tail_one(c, wfc2t, h1[1], t + 1);
    }
    for (; t < t1; t++) {
        int8_t xm[RF_DIM] RF_ALIGN4, h1[4 * RF_DIM] RF_ALIGN4;
        rf_rmsnorm_i16_to_i8(res[t], c->st->G2, c->st->B2, RF_DIM, xm);
        if (c->blk->M_actq)
            rf_relu2sq_i8(xm, wfc1, c->blk->bfc1, c->blk->M_actq,
                          c->blk->s_actq, RF_DIM, 4 * RF_DIM, h1);
        else
            rf_linear_i8(xm, wfc1, c->blk->bfc1, c->blk->Mfc1, c->blk->sfc1,
                         RF_DIM, 4 * RF_DIM, 0, h1);
        mlp_tail_one(c, wfc2t, h1, t);
    }
}

static uint8_t s_v_adj[RF_PD]; /* s_v with the stride dt-shift applied */

static void final_range(int t0, int t1, void *p) {
    ctx_t *c = p;
    const rf_model_t *m = c->m;
    /* gmode 1/2 = the int32 CFG blend. gmode 1 (cond pass): store raw acc.
     * gmode 2 (null pass): blend v_acc_c + w*(v_acc_c - acc_null) then requant
     * with the base M_v/s_v (c->Mv is m->M_v[k] here) into z_tok. */
    if (c->gmode) {
        for (int t = t0; t < t1; t++) {
            int8_t xf[RF_DIM] RF_ALIGN4;
            rf_rmsnorm_i16_to_i8(res[t], m->Gf[c->cond][c->k],
                                 m->Bf[c->cond][c->k], RF_DIM, xf);
            if (c->gmode == 1) {
                for (int o = 0; o < RF_PD; o++)
                    v_acc_c[t][o] = rf_dot_i8(c->w + (size_t)o * RF_DIM, xf,
                                              RF_DIM, m->b_final[o]);
            } else { /* gmode 2: null pass -> blend + requant base M_v */
                for (int o = 0; o < RF_PD; o++) {
                    int32_t an = rf_dot_i8(c->w + (size_t)o * RF_DIM, xf,
                                           RF_DIM, m->b_final[o]);
                    int32_t diff = v_acc_c[t][o] - an;
                    /* acc_g = acc_null + w*(acc_cond - acc_null), w in Q8 */
                    int64_t accg = (int64_t)an +
                                   (((int64_t)c->wq * diff) >> 8);
                    z_tok[t][o] = rf_sat16((int64_t)z_tok[t][o] +
                                           rf_rq(accg, c->Mv[o], s_v_adj[o]));
                }
            }
        }
        return;
    }
    int t = t0;
    for (; t + 1 < t1; t += 2) {
        int8_t xf[2][RF_DIM] RF_ALIGN4;
        rf_rmsnorm_i16_to_i8(res[t], m->Gf[c->cond][c->k],
                             m->Bf[c->cond][c->k], RF_DIM, xf[0]);
        rf_rmsnorm_i16_to_i8(res[t + 1], m->Gf[c->cond][c->k],
                             m->Bf[c->cond][c->k], RF_DIM, xf[1]);
        rf_linear2_i8_acc16(xf[0], c->w, m->b_final, c->Mv, s_v_adj,
                            RF_DIM, RF_PD, z_tok[t], z_tok[t + 1]);
    }
    for (; t < t1; t++) {
        int8_t xf[RF_DIM] RF_ALIGN4;
        rf_rmsnorm_i16_to_i8(res[t], m->Gf[c->cond][c->k], m->Bf[c->cond][c->k],
                             RF_DIM, xf);
        rf_linear_i8_acc16(xf, c->w, m->b_final, c->Mv, s_v_adj,
                           RF_DIM, RF_PD, z_tok[t]);
    }
}

/* One DiT pass. zsrc: where z is read (plain: z_tok, the pass updates it in
 * place; CFG passes: z_prev while both accumulate into z_tok). Mv/sv: the
 * Euler fold for this pass (plain: m->M_v[k]; CFG: per-pass rescale). */
static void dit_step(const rf_model_t *m, int k, int s_shift, int cond,
                     int prefetch_next, const int16_t (*zsrc)[RF_PD],
                     const int32_t *Mv, const uint8_t *sv, int gmode,
                     uint32_t wq) {
    ctx_t c = {.m = m, .zsrc = zsrc, .Mv = Mv, .sv = sv,
               .k = k, .cond = cond, .gmode = gmode, .wq = wq};
    for (int i = 0; i < RF_PD; i++)
        s_v_adj[i] = (uint8_t)(sv[i] - s_shift);

    memcpy(res, m->pos, sizeof(res));
    /* qkv[0] was prefetched by the previous step (or by rf_generate) */
    c.w = rf_stage_wait(RF_SLOT_EMB);
    rf_par_for(RF_TOKENS, embed_range, &c);

    for (int b = 0; b < RF_DEPTH; b++) {
        c.blk = &m->blk[b];
        c.st = &m->step[cond][k][b];
        if (c.st->sproj) { /* v<8: per-channel shift arrays in the blob */
            c.sproj = c.st->sproj;
            c.sfc2 = c.st->sfc2;
        } else {           /* v8: expand the per-entry scalars */
            memset(sp_buf, c.st->sproj_s, RF_DIM);
            memset(sf_buf, c.st->sfc2_s, RF_DIM);
            c.sproj = sp_buf;
            c.sfc2 = sf_buf;
        }
        rf_stage_start(RF_SLOT_FC1, c.blk->Wfc1, (size_t)4 * RF_DIM * RF_DIM);
        rf_stage_start(RF_SLOT_FC2, c.blk->Wfc2, (size_t)4 * RF_DIM * RF_DIM);
        if (c.blk->has_attn) {
            c.w = rf_stage_wait(RF_SLOT_QKV);
            rf_stage_start(RF_SLOT_PROJ, c.blk->Wproj, (size_t)RF_DIM * RF_DIM);
            rf_par_for(RF_TOKENS, norm_qkv_range, &c);
        }
        /* qkv weights (if consumed) freed; prefetch the NEXT attention block's
         * qkv so the paced copy spans this block's remaining ops. Skip over
         * dropped-attention blocks -- their qkv is absent (NULL). */
        {
            const rf_blk_t *nxt = NULL;
            for (int j = b + 1; j < RF_DEPTH; j++)
                if (m->blk[j].has_attn) { nxt = &m->blk[j]; break; }
            if (!nxt && prefetch_next)
                for (int j = 0; j < RF_DEPTH; j++)
                    if (m->blk[j].has_attn) { nxt = &m->blk[j]; break; }
            if (nxt)
                rf_stage_start(RF_SLOT_QKV, nxt->Wqkv,
                               (size_t)3 * RF_DIM * RF_DIM);
        }
        if (c.blk->has_attn) {
            rf_par_for(RF_TOKENS, attn_range, &c);
            c.w = rf_stage_wait(RF_SLOT_PROJ);
            rf_par_for(RF_TOKENS, proj_range, &c);
        }
        /* fc1 at arena[0], fc2 right behind it: one contiguous block */
        c.w = rf_stage_wait(RF_SLOT_FC1);
        (void)rf_stage_wait(RF_SLOT_FC2);
        rf_par_for(RF_TOKENS, mlp_range, &c);
    }

    /* v-output with folded -dt: the Euler update is this acc16 */
    c.w = rf_stage_wait(RF_SLOT_FINAL);
    rf_par_for(RF_TOKENS, final_range, &c);
}

void rf_generate(const rf_model_t *m, uint64_t seed, int k_steps, int cond,
                 int w_idx, uint8_t *img, int16_t *taps) {
    if (k_steps <= 0 || k_steps > (int)m->K || m->K % (uint32_t)k_steps)
        k_steps = (int)m->K;
    if (cond < 0 || cond >= (int)m->n_cond) cond = 0;
    /* guidance is a no-op for the null class (v_cond == v_null) */
    int guided = w_idx >= 0 && w_idx < (int)m->n_w &&
                 cond != (int)m->n_cond - 1;
    int stride = (int)m->K / k_steps;
    int s_shift = 0;
    while ((2 << s_shift) <= stride) s_shift++;

    rf_pcg32_t g;
    rf_pcg32_seed(&g, seed);
    rf_gaussian_clt12(&g, RF_TOKENS * RF_PD, rf_dit_noise);
    for (int i = 0; i < RF_TOKENS * RF_PD; i++)
        ((int16_t *)z_tok)[i] = rf_sat16(rf_dit_noise[i]);
    if (taps) memcpy(taps, z_tok, sizeof(z_tok));
    /* emb/final are step-invariant: staged once for the whole generation */
    rf_stage_start(RF_SLOT_EMB, m->W_emb, (size_t)RF_DIM * RF_PD);
    rf_stage_start(RF_SLOT_FINAL, m->W_final, (size_t)RF_PD * RF_DIM);
    rf_stage_start(RF_SLOT_QKV, m->blk[0].Wqkv, (size_t)3 * RF_DIM * RF_DIM);
    rf_progress_total = (uint8_t)(k_steps + 1);
    rf_step_hook();
    for (int i = 0; i < k_steps; i++) {
        rf_progress = (uint8_t)(i + 1);
        int k = i * stride, last = i + 1 == k_steps;
        if (guided) {
            memcpy(z_prev, z_tok, sizeof(z_prev));
            /* int32 CFG blend: cond pass stores raw acc (gmode 1), null pass
             * blends v_null + w*(v_cond - v_null) and requants with the BASE
             * M_v (gmode 2). Both use m->M_v[k]/s_v[k]; wq = w in Q8. */
            dit_step(m, k, s_shift, cond, 1,
                     (const int16_t(*)[RF_PD])z_prev,
                     m->M_v[k], m->s_v[k], 1, 0);
            dit_step(m, k, s_shift, (int)m->n_cond - 1, !last,
                     (const int16_t(*)[RF_PD])z_prev,
                     m->M_v[k], m->s_v[k], 2, m->w_q8[w_idx]);
        } else {
            dit_step(m, k, s_shift, cond, !last,
                     (const int16_t(*)[RF_PD])z_tok, m->M_v[k], m->s_v[k], 0, 0);
        }
        rf_step_hook();
        if (taps) memcpy(taps + (i + 1) * RF_TOKENS * RF_PD, z_tok, sizeof(z_tok));
    }
    rf_progress = (uint8_t)(k_steps + 1);
    rf_step_hook();
    rf_stage_drain(); /* no in-flight DMA may touch the arena the decoder owns */
    rf_decode(m, (const int16_t(*)[RF_PD])z_tok, img);
    rf_progress = 0;
}
