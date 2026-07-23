/* PCG32 (XSH-RR, fixed default stream) + CLT-12 gaussian. Mirrors int_ops.py. */
#include "rf_ops.h"

#define PCG_MULT 6364136223846793005ULL
#define PCG_INC 1442695040888963407ULL

void rf_pcg32_seed(rf_pcg32_t *g, uint64_t seed) {
    g->state = 0;
    g->state = g->state * PCG_MULT + PCG_INC;
    g->state += seed;
    g->state = g->state * PCG_MULT + PCG_INC;
}

uint32_t rf_pcg32_next(rf_pcg32_t *g) {
    uint64_t x = g->state;
    unsigned count = (unsigned)(x >> 59);
    g->state = x * PCG_MULT + PCG_INC;
    x ^= x >> 18;
    uint32_t out = (uint32_t)(x >> 27);
    return (out >> count) | (out << ((32 - count) & 31));
}

void rf_gaussian_clt12(rf_pcg32_t *g, int n, int32_t *out) {
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < 12; j++) acc += (int32_t)(rf_pcg32_next(g) >> 20);
        out[i] = acc - 24576;
    }
}
