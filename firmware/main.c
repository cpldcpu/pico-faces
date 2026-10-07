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

#ifndef RF_LED_INVERTED
#if defined(ADAFRUIT_FRUIT_JAM)
#define RF_LED_INVERTED 1
#else
#define RF_LED_INVERTED PICO_DEFAULT_LED_PIN_INVERTED
#endif
#endif
#if RF_LED_INVERTED
#define RF_LED_ON 0
#define RF_LED_OFF 1
#else
#define RF_LED_ON 1
#define RF_LED_OFF 0
#endif

/* Either scanout provides the same four-function contract (rf_vga_init,
 * rf_vga_invalidate, rf_vga_dither, rf_step_hook), so the call sites below
 * only care whether one is linked at all. */
#define RF_VIDEO (RF_VGA + RF_DVI)

#define RF_HSTX_KHZ 150000
#if RF_DVI
#if (RF_SYS_KHZ) % (RF_HSTX_KHZ) != 0 || \
    (RF_SYS_KHZ) / (RF_HSTX_KHZ) < 1 || (RF_SYS_KHZ) / (RF_HSTX_KHZ) > 3
#error "RF_DVI needs clk_hstx=150MHz: RF_SYS_KHZ must be 1..3 x 150000"
#endif
#if RF_VGA
#error "RF_DVI and RF_VGA are two scanouts for one framebuffer; pick one"
#endif
#endif

static rf_model_t model;
/* non-static: the display backend reads it live */
uint8_t rf_img[RF_IMG_HW * RF_IMG_HW * RF_IMG_CH];

static void put_u32(uint32_t v) { fwrite(&v, 4, 1, stdout); }
static void put_u16(uint16_t v) { fwrite(&v, 2, 1, stdout); }

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
#if RF_DVI
    /* Must come after set_sys_clock_khz(): the SDK's runtime init leaves
     * clk_hstx glued undivided to clk_sys, and retuning clk_sys neither
     * re-divides it nor updates its recorded frequency. Take the actual
     * clk_sys rather than RF_SYS_KHZ. */
    clock_configure(clk_hstx, 0, CLOCKS_CLK_HSTX_CTRL_AUXSRC_VALUE_CLK_SYS,
                    clock_get_hz(clk_sys), RF_HSTX_KHZ * 1000u);
#endif
    stdio_init_all();
    stdio_set_translate_crlf(&stdio_usb, false);

    extern void rf_par_init(void);
    rf_par_init();
#if RF_VIDEO
    /* Historically this had to come first: scanvideo claims FIXED DMA
     * channels (0..), so the display had to be up before the staging
     * channels were taken from the unused pool. dvi_hstx.c has no such
     * constraint - it claims whatever is free - but the ordering is kept so
     * the scanout is alive (and showing the boot pattern) before anything
     * slower runs. */
    extern void rf_vga_init(void);
    rf_vga_init();
#endif
#if RF_STAGE_DMA
    extern void rf_stage_init(void);
    rf_stage_init();
#endif

    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    /* an output defaults to 0, which on an inverted LED is "lit" - park it */
    gpio_put(PICO_DEFAULT_LED_PIN, RF_LED_OFF);

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
            gpio_put(PICO_DEFAULT_LED_PIN, RF_LED_ON);
#if RF_VIDEO
            /* framebuffer aliases rf_arena; engine is about to reuse it */
            extern void rf_vga_invalidate(void), rf_vga_dither(void);
            rf_vga_invalidate();
#endif
            absolute_time_t t0 = get_absolute_time();
            rf_generate(&model, seed, k_steps, cond, w_idx, rf_img, NULL);
            uint32_t ms = (uint32_t)(absolute_time_diff_us(t0, get_absolute_time()) / 1000);
#if RF_VIDEO
            rf_vga_dither();
#endif
            gpio_put(PICO_DEFAULT_LED_PIN, RF_LED_OFF);
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
#if RF_DVI && RF_VGA_TEST
        } else if (line[0] == 'P') { /* boot test pattern 0..2 */
            extern void rf_dvi_test_pattern(int pat);
            int pat = (int)strtol(line + 1, NULL, 0);
            rf_dvi_test_pattern(pat);
            printf("OK pattern %d\n", pat);
#endif
#if RF_DVI
        } else if (line[0] == 'V') { /* scanout health / restart */
            extern int rf_dvi_bringup_tries, rf_dvi_restart(void);
            extern uint32_t rf_dvi_frame_rate_mhz(uint32_t ms);
            if (line[1] == 'R') rf_dvi_restart();
            printf("dvi bringup=%d refresh=%umHz\n", rf_dvi_bringup_tries,
                   (unsigned)rf_dvi_frame_rate_mhz(500));
#endif
        } else if (line[0] == 'I') { /* info */
            printf("pico-faces K=%u dim=%u depth=%u cond=%u ch=%u blob=%u "
                   "sys=%ukHz hstx=%ukHz meas=%ukHz\n",
                   model.K, model.dim, model.depth, model.n_cond, model.img_ch,
                   (unsigned)(rf_model_blob_end - rf_model_blob),
                   (unsigned)(clock_get_hz(clk_sys) / 1000u),
                   (unsigned)(clock_get_hz(clk_hstx) / 1000u),
                   (unsigned)frequency_count_khz(CLOCKS_FC0_SRC_VALUE_CLK_HSTX));
        }
    }
}
