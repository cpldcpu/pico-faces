/* model.bin loader: a sequential pointer walk mirroring quant/export.py. */
#include <string.h>

#include "rf_model.h"

typedef struct {
    const uint8_t *p, *end;
    int err;
} cur_t;

static const void *take(cur_t *c, size_t n) {
    const uint8_t *p = c->p;
    n = (n + 3) & ~(size_t)3; /* every array is padded to 4 in the file */
    if (p + n > c->end) {
        c->err = 1;
        return c->end;
    }
    c->p += n;
    return p;
}

static uint32_t u32(cur_t *c) {
    uint32_t v;
    memcpy(&v, take(c, 4), 4);
    return v;
}

static void declayer(cur_t *c, rf_declayer_t *L) {
    L->C = u32(c);
    L->O = u32(c);
    L->flags = u32(c);
    L->W = take(c, (size_t)L->O * 9 * L->C);
    L->b = take(c, 4 * L->O);
    L->M = take(c, 4 * L->O);
    L->s = take(c, L->O);
}

int rf_model_load(const uint8_t *blob, size_t len, rf_model_t *m) {
    cur_t c = {blob, blob + len, 0};
    if (u32(&c) != RF_MAGIC) return -1;
    uint32_t ver = u32(&c);
    if (ver < 3 || ver > 8 || u32(&c) != 1) return -1;
    m->K = u32(&c);
    m->dim = u32(&c);
    m->depth = u32(&c);
    m->heads = u32(&c);
    m->tokens = u32(&c);
    m->zch = u32(&c);
    m->zhw = u32(&c);
    m->patch = u32(&c);
    m->n_cond = u32(&c);
    m->img_ch = u32(&c);
    m->pd = m->zch * m->patch * m->patch;
    if (m->K < 1 || m->K > RF_K_MAX || m->dim != RF_DIM ||
        m->depth != RF_DEPTH || m->heads != RF_HEADS ||
        m->tokens != RF_TOKENS || m->pd != RF_PD || m->zhw != RF_ZHW ||
        m->patch != RF_PATCH || m->n_cond < 1 || m->n_cond > RF_COND ||
        m->img_ch != RF_IMG_CH)
        return -2;

    /* v6: per-block attention mask (bit b set = block b has attention). Older
     * versions have attention in every block. */
    uint32_t attn_mask = (ver >= 6) ? u32(&c) : 0xffffffffu;

    const uint32_t d = m->dim, pd = m->pd, T = m->tokens;
    m->lut_max = (ver >= 7) ? 511 : 255; /* v7: 512-entry, 1/64 exp grid */
    m->explut = take(&c, 2 * (size_t)(m->lut_max + 1));
    m->pos = take(&c, 2 * (size_t)T * d);
    m->M_zin = take(&c, 4 * m->K);
    m->s_zin = take(&c, m->K);
    for (uint32_t y = 0; y < m->n_cond; y++) {
        for (uint32_t k = 0; k < m->K; k++) {
            for (uint32_t b = 0; b < m->depth; b++) {
                rf_stepblk_t *st = &m->step[y][k][b];
                st->G1 = take(&c, 2 * d);
                st->B1 = take(&c, 2 * d);
                st->G2 = take(&c, 2 * d);
                st->B2 = take(&c, 2 * d);
                st->Mproj = take(&c, 4 * d);
                if (ver >= 8) { /* one shift per entry (renorm_Ms) */
                    st->sproj = NULL;
                    st->sproj_s = *(const uint8_t *)take(&c, 1);
                    st->Mfc2 = take(&c, 4 * d);
                    st->sfc2 = NULL;
                    st->sfc2_s = *(const uint8_t *)take(&c, 1);
                } else {
                    st->sproj = take(&c, d);
                    st->Mfc2 = take(&c, 4 * d);
                    st->sfc2 = take(&c, d);
                }
            }
            m->Gf[y][k] = take(&c, 2 * d);
            m->Bf[y][k] = take(&c, 2 * d);
        }
    }
    for (uint32_t k = 0; k < m->K; k++) {
        m->M_v[k] = take(&c, 4 * pd);
        m->s_v[k] = take(&c, pd);
    }
    m->n_w = 0;
    if (ver >= 4) { /* classifier-free guidance table sets */
        m->n_w = u32(&c);
        if (m->n_w > RF_W_MAX) return -5;
        for (uint32_t j = 0; j < m->n_w; j++) {
            m->w_q8[j] = u32(&c);
            if (ver < 8) { /* dead since the int32-diff blend; gone in v8 */
                for (uint32_t k = 0; k < m->K; k++) {
                    m->M_v_c[j][k] = take(&c, 4 * pd);
                    m->s_v_c[j][k] = take(&c, pd);
                }
                for (uint32_t k = 0; k < m->K; k++) {
                    m->M_v_n[j][k] = take(&c, 4 * pd);
                    m->s_v_n[j][k] = take(&c, pd);
                }
            } else {
                for (uint32_t k = 0; k < m->K; k++) {
                    m->M_v_c[j][k] = m->M_v_n[j][k] = NULL;
                    m->s_v_c[j][k] = m->s_v_n[j][k] = NULL;
                }
            }
        }
    }
    m->W_emb = take(&c, (size_t)d * pd);
    m->b_emb = take(&c, 4 * d);
    m->M_emb = take(&c, 4 * d);
    m->s_emb = take(&c, d);
    for (uint32_t b = 0; b < m->depth; b++) {
        rf_blk_t *blk = &m->blk[b];
        blk->has_attn = (attn_mask >> b) & 1u;
        if (blk->has_attn) {
            blk->Wqkv = take(&c, (size_t)3 * d * d);
            blk->bqkv = take(&c, 4 * 3 * d);
            blk->Mqkv = take(&c, 4 * 3 * d);
            blk->sqkv = take(&c, 3 * d);
            blk->Gq = take(&c, 2 * d);
            blk->Gk = take(&c, 2 * d);
            if (ver >= 7) { /* per-head softmax scales */
                blk->M_smh = take(&c, 4 * m->heads);
                blk->s_smh = take(&c, m->heads);
                blk->M_sm = 0;
                blk->s_sm = 0;
            } else {
                blk->M_sm = (int32_t)u32(&c);
                blk->s_sm = *(const uint8_t *)take(&c, 1);
                blk->M_smh = NULL;
                blk->s_smh = NULL;
            }
            blk->M_att = take(&c, 4 * d);
            blk->s_att = take(&c, d);
            blk->Wproj = take(&c, (size_t)d * d);
            blk->bproj = take(&c, 4 * d);
        } else {
            /* attention weights absent from the blob; leave pointers NULL so a
             * stray read faults loudly rather than using garbage */
            blk->Wqkv = NULL;
            blk->bqkv = blk->Mqkv = NULL;
            blk->sqkv = NULL;
            blk->Gq = blk->Gk = NULL;
            blk->M_att = NULL;
            blk->s_att = NULL;
            blk->Wproj = NULL;
            blk->bproj = NULL;
            blk->M_sm = 0;
            blk->s_sm = 0;
            blk->M_smh = NULL;
            blk->s_smh = NULL;
        }
        blk->Wfc1 = take(&c, (size_t)4 * d * d);
        blk->bfc1 = take(&c, 4 * 4 * d);
        if (ver >= 7) { /* relu2 square-requant: M_actq/s_actq replace
                         * Mfc1/sfc1/ACT_LUT (h1 = rq(relu(acc)^2>>12, M)) */
            blk->M_actq = take(&c, 4 * 4 * d);
            blk->s_actq = take(&c, 4 * d);
            blk->Mfc1 = NULL;
            blk->sfc1 = NULL;
            blk->act_lut = NULL;
        } else {
            blk->Mfc1 = take(&c, 4 * 4 * d);
            blk->sfc1 = take(&c, 4 * d);
            blk->act_lut = take(&c, 256);
            blk->M_actq = NULL;
            blk->s_actq = NULL;
        }
        blk->Wfc2 = take(&c, (size_t)d * 4 * d);
        blk->bfc2 = take(&c, 4 * d);
    }
    m->W_final = take(&c, (size_t)pd * d);
    m->b_final = take(&c, 4 * pd);
    m->M_zdec = (int32_t)u32(&c);
    m->s_zdec = *(const uint8_t *)take(&c, 1);
    m->n_dec = u32(&c);
    if (m->n_dec > RF_MAX_DEC) return -3;
    for (uint32_t i = 0; i < m->n_dec; i++) declayer(&c, &m->dec[i]);
    m->n_hires = 0;
    if (ver >= 5) { /* ESPCN hires head */
        m->n_hires = u32(&c);
        if (m->n_hires > RF_MAX_HIRES) return -6;
        for (uint32_t i = 0; i < m->n_hires; i++) declayer(&c, &m->hires[i]);
    }
    return c.err ? -4 : 0;
}
