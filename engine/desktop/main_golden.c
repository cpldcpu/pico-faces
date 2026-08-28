/* Desktop golden runner: model.bin + seeds -> .gray images + crc32 prints.
 * Usage: main_golden model.bin out_dir k_steps seed [seed...] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rf_model.h"
#include "rf_ops.h"

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s model.bin out_dir k_steps seed...\n",
                argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 2;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc((size_t)len);
    if (fread(blob, 1, (size_t)len, f) != (size_t)len) return 2;
    fclose(f);

    rf_model_t m;
    int rc = rf_model_load(blob, (size_t)len, &m);
    if (rc) {
        fprintf(stderr, "model load failed: %d\n", rc);
        return 2;
    }

    static uint8_t img[RF_IMG_HW * RF_IMG_HW * RF_IMG_CH];
    static int16_t taps[(RF_K_MAX + 1) * RF_TOKENS * RF_PD];
    int k_steps = atoi(argv[3]);
    for (int a = 4; a < argc; a++) {
        uint64_t seed = strtoull(argv[a], NULL, 0);
        int cond = (int)(seed % m.n_cond); /* golden convention */
        /* CFG golden convention: cycle plain + every guidance table set */
        int w_idx = m.n_w ? (int)(seed % (m.n_w + 1)) - 1 : -1;
        rf_generate(&m, seed, k_steps, cond, w_idx, img, taps);
        const uint8_t *out = img;
        size_t osz = sizeof img;
        char path[512];
        snprintf(path, sizeof path, "%s/eng_%llu.%s", argv[2],
                 (unsigned long long)seed, RF_IMG_CH == 1 ? "gray" : "rgb");
        FILE *o = fopen(path, "wb");
        fwrite(out, 1, osz, o);
        fclose(o);
        snprintf(path, sizeof path, "%s/eng_%llu_taps.bin", argv[2],
                 (unsigned long long)seed);
        o = fopen(path, "wb");
        fwrite(taps, sizeof(int16_t), sizeof(taps) / sizeof(int16_t), o);
        fclose(o);
        printf("seed %llu: crc32 %08x\n", (unsigned long long)seed,
               rf_crc32(out, osz));
    }
    return 0;
}
