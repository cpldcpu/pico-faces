/* pico-faces firmware.
 * USB-CDC protocol:
 *   host -> "G <seed> [k_steps] [class]\n"   (class default: seed % n_cond)
 *   dev  -> "RFI2" | u32 seed | u16 w | u16 h | u16 ch | u16 class
 *           | w*h*ch image bytes (HWC) | u32 crc32 | u32 gen_ms
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/structs/qmi.h"
#include "hardware/vreg.h"
#include "pico/stdio_usb.h"
#include "pico/stdlib.h"

#include "rf_model.h"
#include "rf_ops.h"

extern const uint8_t rf_model_blob[];
extern const uint8_t rf_model_blob_end[];

static rf_model_t model;
/* non-static: the VGA renderer reads it live */
uint8_t rf_img[RF_IMG_HW * RF_IMG_HW * RF_IMG_CH];

static void put_u32(uint32_t v) { fwrite(&v, 4, 1, stdout); }
static void put_u16(uint16_t v) { fwrite(&v, 2, 1, stdout); }

#if RF_HIRES
/* USB sink: stream each 256 row and accumulate its CRC. Does NOT touch the
 * arena, so the decoder feature map rf_hires reads stays intact for the whole
 * pass. The 256 image never exists in memory, so gen_ms includes the USB
 * transfer of the pixel payload.
 *
 * The VGA screen does NOT show the 256: the 384x384 framebuffer aliases BOTH
 * arenas, so dithering the 256 into it would overwrite the feature map that
 * rf_hires is still reading (in any pass -- read and write share arena[1]),
 * and there is no arena-free 131 KB to stage the 256 pixels. So on hires VGA
 * builds the screen shows the 128 decode upscaled 3x (rf_img, arena-free,
 * via the standard rf_vga_dither); the true 256 is the USB deliverable. */
static uint32_t tx_crc;
static void hires_usb_sink(int y, const uint8_t *row, void *user) {
    (void)y;
    (void)user;
    fwrite(row, 1, (size_t)RF_OUT_HW * RF_IMG_CH, stdout);
    tx_crc = rf_crc32_acc(tx_crc, row, (size_t)RF_OUT_HW * RF_IMG_CH);
}
#endif

/* raise the flash clock divider before overclocking; must run from SRAM
 * because it changes XIP timing underneath any flash-resident caller */
static void __no_inline_not_in_flash_func(qmi_set_clkdiv)(uint32_t div) {
    uint32_t t = qmi_hw->m[0].timing;
    qmi_hw->m[0].timing = (t & ~QMI_M0_TIMING_CLKDIV_BITS) |
                          (div << QMI_M0_TIMING_CLKDIV_LSB);
    __compiler_memory_barrier();
}

int main(void) {
#if RF_SYS_KHZ > 150000
    vreg_set_voltage(VREG_VOLTAGE_1_30);
    sleep_ms(10);
    /* QSPI stays at RF_SYS_KHZ/4 (75 MHz at 300) - within W25Q32 spec */
    qmi_set_clkdiv(4);
    set_sys_clock_khz(RF_SYS_KHZ, true);
#endif
    stdio_init_all();
    stdio_set_translate_crlf(&stdio_usb, false);

    extern void rf_par_init(void);
    rf_par_init();
#if RF_VGA
    /* scanvideo claims FIXED DMA channels (0..); init it before the
     * staging channels are allocated from the unused pool */
    extern void rf_vga_init(void);
    rf_vga_init();
#endif
#if RF_STAGE_DMA
    extern void rf_stage_init(void);
    rf_stage_init();
#endif

    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);

    int rc = rf_model_load(rf_model_blob,
                           (size_t)(rf_model_blob_end - rf_model_blob), &model);

    char line[64];
    int n = 0;
    for (;;) {
        int ch = getchar_timeout_us(100000);
        if (ch == PICO_ERROR_TIMEOUT) continue;
        if (ch != '\n' && ch != '\r') {
            if (n < (int)sizeof line - 1) line[n++] = (char)ch;
            continue;
        }
        line[n] = 0;
        n = 0;
        if (rc != 0) {
            printf("ERR model load %d\n", rc);
            continue;
        }
        if (line[0] == 'G') {
            char *e1, *e2, *e3, *e4;
            uint64_t seed = strtoull(line + 1, &e1, 0);
            int k_steps = (int)strtol(e1, &e2, 0);
            if (!k_steps) k_steps = 4;
            long cv = strtol(e2, &e3, 0);
            /* golden convention when the class token is absent */
            int cond = (e3 != e2) ? (int)cv : (int)(seed % model.n_cond);
            /* optional guidance strength w (e.g. 4/6/8): matched against
             * the baked w_q8 sets; absent w on a CFG build follows the
             * golden convention so goldens reproduce over USB */
            long wv = strtol(e3, &e4, 0);
            int w_idx = -1;
            if (e4 != e3) {
                for (uint32_t j = 0; j < model.n_w; j++)
                    if (model.w_q8[j] == (uint32_t)(wv * 256)) w_idx = (int)j;
            } else if (e3 == e2 && model.n_w) {
                w_idx = (int)(seed % (model.n_w + 1)) - 1;
            }
            gpio_put(PICO_DEFAULT_LED_PIN, 1);
#if RF_VGA
            /* framebuffer aliases rf_arena; engine is about to reuse it */
            extern void rf_vga_invalidate(void), rf_vga_dither(void);
            rf_vga_invalidate();
#endif
            absolute_time_t t0 = get_absolute_time();
            rf_generate(&model, seed, k_steps, cond, w_idx, rf_img, NULL);
#if RF_HIRES
            /* header first, then the head pass streams 256 rows straight into
             * the reply. The USB sink never writes the arena, so the decoder
             * feature map rf_hires reads survives the whole pass. */
            fwrite("RFI2", 1, 4, stdout);
            put_u32((uint32_t)seed);
            put_u16(RF_OUT_HW);
            put_u16(RF_OUT_HW);
            put_u16(RF_IMG_CH);
            put_u16((uint16_t)cond);
            tx_crc = RF_CRC32_INIT;
            rf_hires(&model, rf_img, hires_usb_sink, NULL);
            uint32_t ms = (uint32_t)(absolute_time_diff_us(t0, get_absolute_time()) / 1000);
#if RF_VGA
            /* screen shows the 128 decode at 384 (rf_img is arena-free and
             * still valid); the 256 cannot be dithered into the arena-aliased
             * framebuffer without clobbering the feature map. */
            rf_vga_dither();
#endif
            gpio_put(PICO_DEFAULT_LED_PIN, 0);
            put_u32(tx_crc ^ RF_CRC32_INIT);
            put_u32(ms);
            fflush(stdout);
#else
            uint32_t ms = (uint32_t)(absolute_time_diff_us(t0, get_absolute_time()) / 1000);
#if RF_VGA
            rf_vga_dither();
#endif
            gpio_put(PICO_DEFAULT_LED_PIN, 0);
            fwrite("RFI2", 1, 4, stdout);
            put_u32((uint32_t)seed);
            put_u16(RF_IMG_HW);
            put_u16(RF_IMG_HW);
            put_u16(RF_IMG_CH);
            put_u16((uint16_t)cond);
            fwrite(rf_img, 1, sizeof rf_img, stdout);
            put_u32(rf_crc32(rf_img, sizeof rf_img));
            put_u32(ms);
            fflush(stdout);
#endif
        } else if (line[0] == 'I') { /* info */
            printf("pico-faces K=%u dim=%u depth=%u cond=%u ch=%u blob=%u sys=%dkHz\n",
                   model.K, model.dim, model.depth, model.n_cond, model.img_ch,
                   (unsigned)(rf_model_blob_end - rf_model_blob), RF_SYS_KHZ);
        }
    }
}
