/* VGA output on the Pimoroni Pico VGA Demo Base via pico_scanvideo (DPI).
 *
 * Mode: 640x480@60, yscale=1 -> 480 real lines/frame. The face is shown at
 * 384x384 (3x3 pixels), centered. While the engine runs (rf_progress != 0)
 * the display shows the evolving 16x16 latent (channel 0) as chunky blocks
 * plus a progress bar; when idle it shows the last image.
 *
 * Banding: RGB555 drops 3 bits per channel, visible as bands on smooth face
 * gradients. The displayed image is Floyd-Steinberg dithered at full DISPLAY
 * resolution (384x384, serpentine) into a precomputed RGB555 framebuffer --
 * error diffusion needs a whole-image sequential pass, so it runs once per
 * generated image (rf_vga_dither, ~ms), not per scanout. The framebuffer
 * cannot be a plain static (no 288KB spare), so its rows alias rf_arena
 * (256KB, idle whenever an image is on display) plus a small static spill.
 * Contract with main.c: rf_vga_invalidate() BEFORE rf_generate() (the engine
 * reuses the arena), rf_vga_dither() after it returns. While the framebuffer
 * is invalid (decode phase: the image "scans in" as the decoder writes it)
 * lines fall back to on-the-fly ordered dithering, also at display res.
 *
 * Scanline assembly happens in a 100us repeating-timer IRQ on core0; with
 * the framebuffer path a face line is a single memcpy.
 */
#include <string.h>

#include "pico/scanvideo.h"
#include "pico/scanvideo/composable_scanline.h"
#include "pico/stdlib.h"

#include "rf_model.h"
#include "rf_ops.h"

extern uint8_t rf_img[RF_IMG_HW * RF_IMG_HW * RF_IMG_CH];
extern volatile uint8_t rf_progress;
extern volatile uint8_t rf_progress_total;
const int16_t *rf_z_state(void);

#define IMG_X0 128 /* left margin in pixels (640 - 384) / 2 */
#define IMG_W 384
#define IMG_H 384
#define LINES 480
#define IMG_Y0 48          /* image band: lines 48..431 */
#define BAR_Y0 450         /* progress bar: lines 450..461 */

static const scanvideo_mode_t rf_vga_mode = {
    .default_timing = &vga_timing_640x480_60_default,
    .pio_program = &video_24mhz_composable,
    .width = 640,
    .height = LINES,
    .xscale = 1,
    .yscale = 1,
};

static uint16_t gray_lut[256];

/* ---- dithering --------------------------------------------------------- */

/* 8x8 Bayer matrix, pre-shifted to thresholds 0..7 (the 3 dropped bits);
 * used by the fallback path while the framebuffer is invalid. */
static const uint8_t bayer3[8][8] = {
    {0, 4, 1, 5, 0, 4, 1, 5}, {6, 2, 7, 3, 6, 2, 7, 3},
    {1, 5, 0, 4, 1, 5, 0, 4}, {7, 3, 6, 2, 7, 3, 6, 2},
    {0, 4, 1, 5, 0, 4, 1, 5}, {6, 2, 7, 3, 6, 2, 7, 3},
    {1, 5, 0, 4, 1, 5, 0, 4}, {7, 3, 6, 2, 7, 3, 6, 2},
};
/* d5[t][v] = min(255, v + t) >> 3: dithered 8->5 bit, one load per channel */
static uint8_t d5[8][256];

static inline uint16_t dpix_rgb(const uint8_t *px, uint8_t t) {
    return PICO_SCANVIDEO_PIXEL_FROM_RGB5(d5[t][px[0]], d5[t][px[1]],
                                          d5[t][px[2]]);
}
static inline uint16_t dpix_gray(uint8_t v, uint8_t t) {
    return (uint16_t)(d5[t][v] * 0x0421u); /* replicate 5 bits to R,G,B */
}

/* Framebuffer rows: first FB_ARENA_ROWS alias rf_arena, the rest spill into
 * a static. Indirection keeps the split invisible to producer and scanout. */
#define FB_ARENA_ROWS ((int)(sizeof rf_arena / (IMG_W * 2)))
static uint16_t fb_spill[IMG_H - FB_ARENA_ROWS][IMG_W];
static uint16_t *fb_row[IMG_H];
static volatile bool fb_valid;

void rf_vga_invalidate(void) { fb_valid = false; }

/* Serpentine Floyd-Steinberg at display res: each source pixel spans 3
 * columns and 3 rows, so quantization error diffuses in display pixels and
 * the grain stays 1 pixel fine. Error is carried in 16ths (weights 7/3/5/1);
 * worst-case accumulation < 16*8 keeps int16 safe. Runs with the engine
 * idle only: it writes rf_arena through fb_row. */
void rf_vga_dither(void) {
    static int16_t err[2][IMG_W + 2][RF_IMG_CH]; /* too big for the stack */
    memset(err, 0, sizeof err);
    for (int y = 0; y < IMG_H; y++) {
        const uint8_t *row = rf_img + (size_t)(y / 3) * 128 * RF_IMG_CH;
        int16_t (*cur)[RF_IMG_CH] = err[y & 1] + 1;
        int16_t (*nxt)[RF_IMG_CH] = err[(y & 1) ^ 1] + 1;
        memset(err[(y & 1) ^ 1], 0, sizeof err[0]);
        int dir = (y & 1) ? -1 : 1;
        int x = (y & 1) ? IMG_W - 1 : 0;
        uint16_t *out = fb_row[y];
        for (int i = 0; i < IMG_W; i++, x += dir) {
            const uint8_t *s = row + (size_t)(x / 3) * RF_IMG_CH;
            uint8_t q[RF_IMG_CH];
            for (int c = 0; c < RF_IMG_CH; c++) {
                int v = s[c] + ((cur[x][c] + 8) >> 4);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                int q5 = v >> 3;
                int e = v - ((q5 << 3) | (q5 >> 2)); /* exact 5->8 recon */
                cur[x + dir][c] = (int16_t)(cur[x + dir][c] + 7 * e);
                nxt[x - dir][c] = (int16_t)(nxt[x - dir][c] + 3 * e);
                nxt[x][c] = (int16_t)(nxt[x][c] + 5 * e);
                nxt[x + dir][c] = (int16_t)(nxt[x + dir][c] + e);
                q[c] = (uint8_t)q5;
            }
#if RF_IMG_CH == 3
            out[x] = PICO_SCANVIDEO_PIXEL_FROM_RGB5(q[0], q[1], q[2]);
#else
            out[x] = (uint16_t)(q[0] * 0x0421u);
#endif
        }
    }
    fb_valid = true;
}

/* ---- scanline assembly ------------------------------------------------- */

static uint16_t *end_line(scanvideo_scanline_buffer_t *b, uint16_t *p) {
    *p++ = COMPOSABLE_RAW_1P;
    *p++ = 0;
    if (2 & (uintptr_t)p) {
        *p++ = COMPOSABLE_EOL_ALIGN;
    } else {
        *p++ = COMPOSABLE_EOL_SKIP_ALIGN;
        *p++ = 0xffff;
    }
    b->data_used = (uint16_t)(((uint32_t *)p) - b->data);
    return p;
}

static uint16_t *color_run(uint16_t *p, uint16_t color, int w) {
    *p++ = COMPOSABLE_COLOR_RUN;
    *p++ = color;
    *p++ = (uint16_t)(w - 3);
    return p;
}

static void render_line(scanvideo_scanline_buffer_t *b) {
    int y = scanvideo_scanline_number(b->scanline_id);
    uint16_t *p = (uint16_t *)b->data;
    uint8_t prog = rf_progress;
    const uint16_t bg = PICO_SCANVIDEO_PIXEL_FROM_RGB8(24, 24, 40);

    if (y >= IMG_Y0 && y < IMG_Y0 + IMG_H) {
        int iy = y - IMG_Y0;
        p = color_run(p, bg, IMG_X0);
        if (prog) {
            /* latent preview: zhw x zhw blocks, channel 0 of z (unpatchify) */
            const int16_t *z = rf_z_state();
            const int P = RF_PATCH, G = RF_ZHW / RF_PATCH;
            int py = iy * RF_ZHW / IMG_H;
            for (int px = 0; px < RF_ZHW; px++) {
                int16_t v = z[((py / P) * G + (px / P)) * RF_PD +
                              (py % P) * P + (px % P)];
                int g = 128 + (v >> 6);
                if (g < 0) g = 0;
                if (g > 255) g = 255;
                p = color_run(p, gray_lut[g], IMG_W / RF_ZHW);
            }
        } else if (fb_valid) {
            const uint16_t *src = fb_row[iy];
            *p++ = COMPOSABLE_RAW_RUN;
            *p++ = src[0];
            *p++ = IMG_W - 3;
            memcpy(p, src + 1, (IMG_W - 1) * sizeof(uint16_t));
            p += IMG_W - 1;
        } else {
            const uint8_t *row = rf_img + (size_t)(iy / 3) * 128 * RF_IMG_CH;
            const uint8_t *dl = bayer3[iy & 7];
#if RF_IMG_CH == 3
#define ROWPIX(i, j) dpix_rgb(row + 3 * (i), dl[(3 * (i) + (j)) & 7])
#else
#define ROWPIX(i, j) dpix_gray(row[i], dl[(3 * (i) + (j)) & 7])
#endif
            *p++ = COMPOSABLE_RAW_RUN;
            *p++ = ROWPIX(0, 0);
            *p++ = IMG_W - 3;
            *p++ = ROWPIX(0, 1);
            *p++ = ROWPIX(0, 2);
            for (int sx = 1; sx < 128; sx++) {
                *p++ = ROWPIX(sx, 0);
                *p++ = ROWPIX(sx, 1);
                *p++ = ROWPIX(sx, 2);
            }
#undef ROWPIX
        }
        p = color_run(p, bg, 640 - IMG_X0 - IMG_W - 1);
    } else if (prog && y >= BAR_Y0 && y < BAR_Y0 + 12) {
        int done = (int)prog * IMG_W / (int)rf_progress_total;
        p = color_run(p, bg, IMG_X0);
        if (done > 3)
            p = color_run(p, PICO_SCANVIDEO_PIXEL_FROM_RGB8(80, 220, 120), done);
        if (IMG_W - done > 3)
            p = color_run(p, PICO_SCANVIDEO_PIXEL_FROM_RGB8(50, 50, 60),
                          IMG_W - done);
        p = color_run(p, bg, 640 - IMG_X0 - IMG_W - 1);
    } else {
        p = color_run(p, bg, 639);
    }
    end_line(b, p);
}

static bool vga_timer_cb(repeating_timer_t *t) {
    (void)t;
    scanvideo_scanline_buffer_t *b;
    for (int i = 0; i < 16; i++) {
        b = scanvideo_begin_scanline_generation(false);
        if (!b) break;
        render_line(b);
        scanvideo_end_scanline_generation(b);
    }
    return true;
}

static repeating_timer_t vga_timer;

void rf_vga_init(void) {
    for (int i = 0; i < 256; i++)
        gray_lut[i] = PICO_SCANVIDEO_PIXEL_FROM_RGB8(i, i, i);
    for (int t = 0; t < 8; t++)
        for (int v = 0; v < 256; v++)
            d5[t][v] = (uint8_t)((v + t > 255 ? 255 : v + t) >> 3);
    for (int y = 0; y < IMG_H; y++)
        fb_row[y] = (y < FB_ARENA_ROWS)
                        ? (uint16_t *)&rf_arena[0][0] + (size_t)y * IMG_W
                        : fb_spill[y - FB_ARENA_ROWS];
    scanvideo_setup(&rf_vga_mode);
    scanvideo_timing_enable(true);
    add_repeating_timer_us(-100, vga_timer_cb, NULL, &vga_timer);
}
