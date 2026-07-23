/* Default per-model engine configuration = m1_gray. Per-model builds
 * override this file by putting the generated artifacts/<model>/export/
 * rf_cfg.h (or build/m2_rand/rf_cfg.h) FIRST on the include path; keep
 * these values byte-identical to what quant/export.py emits for m1_gray. */
#ifndef RF_CFG_H
#define RF_CFG_H
#define RF_K_MAX 8
#define RF_DIM 128
#define RF_DEPTH 8
#define RF_HEADS 4
#define RF_TOKENS 64
#define RF_ZCH 4
#define RF_ZHW 16
#define RF_PATCH 2
#define RF_PD 16
#define RF_COND 1
#define RF_IMG_CH 1
#define RF_IMG_HW 128
#define RF_MAX_DEC 16
#define RF_DEC_NZ_MAX 1024
#define RF_DEC_W_MAX 128
#define RF_DEC_O_MAX 64
#endif
