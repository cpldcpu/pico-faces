/* Desktop probe: like main_golden but with EXPLICIT cond + w_idx, to reproduce
 * a device run that forces --cls/--w instead of the seed%%n golden convention.
 * Usage: main_probe model.bin out_dir k_steps cond w_idx seed [seed...]
 *   w_idx: -1 = plain; 0..n_w-1 = CFG w-table index (w=4 -> 0 when cfg_w=[4,6,8])
 * Prints the crc32 the device computes over the raw image, and writes eng_<seed>.rgb. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rf_model.h"
#include "rf_ops.h"

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s model.bin out_dir k_steps cond w_idx seed...\n",
                argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = malloc((size_t)len);
    if (fread(blob, 1, (size_t)len, f) != (size_t)len) return 2;
    fclose(f);

    rf_model_t m;
    if (rf_model_load(blob, (size_t)len, &m)) {
        fprintf(stderr, "model load failed\n");
        return 2;
    }

    static uint8_t img[RF_IMG_HW * RF_IMG_HW * RF_IMG_CH];
    static int16_t taps[(RF_K_MAX + 1) * RF_TOKENS * RF_PD];
    int k_steps = atoi(argv[3]);
    int cond = atoi(argv[4]);
    int w_idx = atoi(argv[5]);
    fprintf(stderr, "model n_cond=%d n_w=%d | k=%d cond=%d w_idx=%d\n",
            m.n_cond, m.n_w, k_steps, cond, w_idx);
    for (int a = 6; a < argc; a++) {
        uint64_t seed = strtoull(argv[a], NULL, 0);
        rf_generate(&m, seed, k_steps, cond, w_idx, img, taps);
        char path[512];
        snprintf(path, sizeof path, "%s/eng_%llu.%s", argv[2],
                 (unsigned long long)seed, RF_IMG_CH == 1 ? "gray" : "rgb");
        FILE *o = fopen(path, "wb");
        fwrite(img, 1, sizeof img, o);
        fclose(o);
        printf("seed %llu k=%d cond=%d w_idx=%d: crc32 %08x\n",
               (unsigned long long)seed, k_steps, cond, w_idx,
               rf_crc32(img, sizeof img));
    }
    return 0;
}
