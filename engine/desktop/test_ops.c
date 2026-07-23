/* Replays tests/vectors/ops.bin through the C kernels and byte-compares. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rf_ops.h"

static unsigned char *buf;
static size_t pos, len;

static void *take(size_t n) {
    if (pos + n > len) {
        fprintf(stderr, "vector file underrun\n");
        exit(2);
    }
    void *p = buf + pos;
    pos += n;
    return p;
}

static int fails;

static void check(int op, int idx, const void *got, const void *want, size_t n) {
    if (memcmp(got, want, n) != 0) {
        int first = -1;
        for (size_t i = 0; i < n; i++)
            if (((const unsigned char *)got)[i] != ((const unsigned char *)want)[i]) {
                first = (int)i;
                break;
            }
        printf("FAIL op %d case %d (first diff at byte %d)\n", op, idx, first);
        fails++;
    } else {
        printf("ok   op %d case %d\n", op, idx);
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "tests/vectors/ops.bin";
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc(len);
    if (fread(buf, 1, len, f) != len) return 2;
    fclose(f);

    if (memcmp(take(4), "OPTV", 4) != 0) {
        fprintf(stderr, "bad magic\n");
        return 2;
    }
    uint32_t n_tests = *(uint32_t *)take(4);
    int counts[16] = {0};

    for (uint32_t t = 0; t < n_tests; t++) {
        uint32_t *hdr = take(20);
        uint32_t op = hdr[0], d0 = hdr[1], d1 = hdr[2], d2 = hdr[3], d3 = hdr[4];
        int idx = counts[op]++;
        switch (op) {
        case 1: { /* linear */
            int K = d0, O = d1, relu = d2;
            int8_t *x = take(K), *W = take((size_t)O * K);
            int32_t *b = take(4 * O), *M = take(4 * O);
            uint8_t *s = take(O);
            int8_t *want = take(O);
            int8_t *y = malloc(O);
            rf_linear_i8(x, W, b, M, s, K, O, relu, y);
            check(op, idx, y, want, O);
            free(y);
            break;
        }
        case 2: { /* linear_acc16 */
            int K = d0, O = d1;
            int8_t *x = take(K), *W = take((size_t)O * K);
            int32_t *b = take(4 * O), *M = take(4 * O);
            uint8_t *s = take(O);
            int16_t *res = take(2 * O), *want = take(2 * O);
            int16_t *r = malloc(2 * O);
            memcpy(r, res, 2 * O);
            rf_linear_i8_acc16(x, W, b, M, s, K, O, r);
            check(op, idx, r, want, 2 * O);
            free(r);
            break;
        }
        case 3: { /* conv3x3 */
            int H = d0 >> 16, W_ = d0 & 0xFFFF, C = d1, O = d2;
            int relu = d3 & 1, stride = (d3 & 2) ? 2 : 1, up = (d3 >> 2) & 1,
                acc16 = (d3 >> 3) & 1, u8 = (d3 >> 4) & 1;
            int Ho = (up ? 2 * H : H) / stride, Wo = (up ? 2 * W_ : W_) / stride;
            int8_t *x = take((size_t)H * W_ * C);
            int8_t *w = take((size_t)O * 9 * C);
            int32_t *b = take(4 * O), *M = take(4 * O);
            uint8_t *s = take(O);
            size_t on = (size_t)Ho * Wo * O;
            if (acc16) {
                int16_t *res = take(2 * on), *want = take(2 * on);
                int16_t *r = malloc(2 * on);
                memcpy(r, res, 2 * on);
                rf_conv3x3_i8(x, H, W_, C, w, b, M, s, O, relu, stride, up, 0,
                              NULL, r);
                check(op, idx, r, want, 2 * on);
                free(r);
            } else {
                int8_t *want = take(on);
                int8_t *y = malloc(on);
                rf_conv3x3_i8(x, H, W_, C, w, b, M, s, O, relu, stride, up, u8,
                              y, NULL);
                check(op, idx, y, want, on);
                free(y);
            }
            break;
        }
        case 4: { /* requant 16->8 */
            int n = d0;
            int16_t *x = take(2 * n);
            int8_t *want = take(n);
            int8_t *y = malloc(n);
            rf_requant_i16_to_i8(x, n, (int32_t)d1, (uint8_t)d2, y);
            check(op, idx, y, want, n);
            free(y);
            break;
        }
        case 5: { /* rmsnorm */
            int K = d0;
            int16_t *x = take(2 * K), *G = take(2 * K), *B = take(2 * K);
            int8_t *want = take(K);
            int8_t *y = malloc(K);
            rf_rmsnorm_i16_to_i8(x, G, B, K, y);
            check(op, idx, y, want, K);
            free(y);
            break;
        }
        case 6: { /* softmax */
            int N = d0;
            int32_t *scores = take(4 * N);
            uint16_t *lut = take(2 * 256);
            int8_t *want = take(N);
            int8_t *y = malloc(N);
            rf_softmax_i32_to_i8(scores, N, (int32_t)d1, (uint8_t)d2, lut, y);
            check(op, idx, y, want, N);
            free(y);
            break;
        }
        case 7: { /* lut */
            int n = d0;
            int8_t *x = take(n), *lut = take(256), *want = take(n);
            int8_t *y = malloc(n);
            rf_lut_i8(x, n, lut, y);
            check(op, idx, y, want, n);
            free(y);
            break;
        }
        case 8: { /* gaussian */
            uint64_t seed = d0 | ((uint64_t)d1 << 32);
            int n = d2;
            int32_t *want = take(4 * n);
            int32_t *g = malloc(4 * n);
            rf_pcg32_t st;
            rf_pcg32_seed(&st, seed);
            rf_gaussian_clt12(&st, n, g);
            check(op, idx, g, want, 4 * n);
            free(g);
            break;
        }
        case 9: { /* crc32 */
            int n = d0;
            uint8_t *data = take(n);
            uint32_t got = rf_crc32(data, n);
            check(op, idx, &got, &d1, 4);
            break;
        }
        case 10: { /* isqrt64 */
            int n = d0;
            uint64_t *v = take(8 * n);
            uint32_t *want = take(4 * n);
            uint32_t *r = malloc(4 * n);
            for (int i = 0; i < n; i++) r[i] = rf_isqrt64(v[i]);
            check(op, idx, r, want, 4 * n);
            free(r);
            break;
        }
        default:
            fprintf(stderr, "unknown op %u\n", op);
            return 2;
        }
    }
    printf(fails ? "\n%d FAILURES\n" : "\nALL PASS (%u tests)\n",
           fails ? fails : n_tests);
    return fails ? 1 : 0;
}
