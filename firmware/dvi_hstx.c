/* Interrupt-free DVI scanout over HSTX (640x480@60, Adafruit Fruit Jam).
 *
 * The DVI twin of vga_noirq.c, same four-function contract, same core idea:
 * a static DMA command list that loops in hardware. Steady state is ZERO
 * interrupts and zero CPU, which is what keeps generation fast.
 *
 *   ctl channel   walks a static list of 16-byte blocks in the DMA channel's
 *     native register order (READ/WRITE/COUNT/CTRL_TRIG, write-ring 16B onto
 *     the dat channel's registers). Each block it writes triggers dat.
 *   dat channel   streams that block into the HSTX FIFO (DREQ_HSTX) and
 *     chains back to ctl. The list's final block instead makes dat write
 *     &cblist[0] into ctl's al3 READ_ADDR trigger alias - the frame rewinds
 *     entirely in hardware, and nothing is ever rebuilt.
 *
 * Live content without IRQs: every image-band line slot has the SAME transfer
 * count whether it points at a framebuffer row or a preview row, so switching
 * display<->preview is one atomic 32-bit READ_ADDR store per slot.
 */
#include <assert.h>
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/gpio.h"
#include "hardware/structs/bus_ctrl.h"
#include "hardware/structs/hstx_ctrl.h"
#include "hardware/structs/hstx_fifo.h"
#include "hardware/sync.h"
#include "pico/stdlib.h"

#include "rf_model.h"
#include "rf_ops.h"

extern uint8_t rf_img[RF_IMG_HW * RF_IMG_HW * RF_IMG_CH];
extern volatile uint8_t rf_progress;
extern volatile uint8_t rf_progress_total;
const int16_t *rf_z_state(void);

/* ---- screen layout (matches vga_noirq.c) -------------------------------- */
#define IMG_X0 128
#define IMG_W 384
#define IMG_H 384
#define IMG_Y0 48
#define BAR_Y0 450
#define BAR_H 12
#define H_ACTIVE 640
#define V_ACTIVE 480
#define IMG_X1 (H_ACTIVE - IMG_X0 - IMG_W) /* right margin, 128 */

/* ---- DVI timing: (30 MHz pixel clock, 952x525, 60.0 Hz) ------
 * clk_hstx = 150 MHz is set in main.c under RF_DVI; HSTX is DDR and spends
 * 10 TMDS bits on a pixel, so the pixel clock is clk_hstx / 5. The porch
 * split is VGA's scaled by 30/25.175. Both syncs are active low. */
#define H_FRONT 19
#define H_SYNC 114
#define H_BACK 179
#define H_TOTAL (H_FRONT + H_SYNC + H_BACK + H_ACTIVE) /* 952 */
#define V_FRONT 10
#define V_SYNC 2
#define V_BACK 33
#define V_TOTAL (V_FRONT + V_SYNC + V_BACK + V_ACTIVE) /* 525 */
#define V_BLANK (V_TOTAL - V_ACTIVE)                   /* 45 */

/* TMDS control symbols; lanes 1 and 2 idle at CTRL_00 through blanking. */
#define TMDS_CTRL_00 0x354u
#define TMDS_CTRL_01 0x0abu
#define TMDS_CTRL_10 0x154u
#define TMDS_CTRL_11 0x2abu
#define SYNC_V0_H0 (TMDS_CTRL_00 | (TMDS_CTRL_00 << 10) | (TMDS_CTRL_00 << 20))
#define SYNC_V0_H1 (TMDS_CTRL_01 | (TMDS_CTRL_00 << 10) | (TMDS_CTRL_00 << 20))
#define SYNC_V1_H0 (TMDS_CTRL_10 | (TMDS_CTRL_00 << 10) | (TMDS_CTRL_00 << 20))
#define SYNC_V1_H1 (TMDS_CTRL_11 | (TMDS_CTRL_00 << 10) | (TMDS_CTRL_00 << 20))

/* command expander opcodes (count in the low 12 bits, in *pixel slots*) */
#define CMD_RAW_REPEAT (0x1u << 12)  /* pop 1 word, emit it raw N times   */
#define CMD_TMDS (0x2u << 12)        /* TMDS-encode N pixels from the FIFO */
#define CMD_TMDS_REPEAT (0x3u << 12) /* pop 1 word, encode it N times      */
#define CMD_NOP (0xfu << 12)

/* RGB332 in a byte: [7:5]=R [4:2]=G [1:0]=B, 4 px per 32-bit FIFO word.
 * NBITS is (width - 1); ROT brings each channel's field up to bit 31. */
#define EXPAND_TMDS_RGB332                                                 \
    (2u << HSTX_CTRL_EXPAND_TMDS_L2_NBITS_LSB |                            \
     0u << HSTX_CTRL_EXPAND_TMDS_L2_ROT_LSB |                              \
     2u << HSTX_CTRL_EXPAND_TMDS_L1_NBITS_LSB |                            \
     29u << HSTX_CTRL_EXPAND_TMDS_L1_ROT_LSB |                             \
     1u << HSTX_CTRL_EXPAND_TMDS_L0_NBITS_LSB |                            \
     26u << HSTX_CTRL_EXPAND_TMDS_L0_ROT_LSB)
#define EXPAND_SHIFT_RGB332                                                \
    (4u << HSTX_CTRL_EXPAND_SHIFT_ENC_N_SHIFTS_LSB |                       \
     8u << HSTX_CTRL_EXPAND_SHIFT_ENC_SHIFT_LSB |                          \
     1u << HSTX_CTRL_EXPAND_SHIFT_RAW_N_SHIFTS_LSB |                       \
     0u << HSTX_CTRL_EXPAND_SHIFT_RAW_SHIFT_LSB)

#define RGB332(r, g, b) (((r) << 5) | ((g) << 2) | (b)) /* r,g,b = 0..7,0..7,0..3 */
#define RGB332_W(c) ((uint32_t)(c) * 0x01010101u)

#define BG RGB332(0, 0, 1)     /* (0,0,85)     */
#define BAR_OFF RGB332(1, 1, 1) /* (36,36,85)  */
#define BAR_ON RGB332(2, 6, 1)  /* (73,219,85) */

/* ---- scanline buffers ---------------------------------------------------
 * One DMA block is one whole scanline, so each image row carries its own
 * command prefix and suffix:
 *
 *   [0..7]    front porch / hsync / back porch  (RAW_REPEAT + sync word)
 *   [8..9]    TMDS_REPEAT | IMG_X0, BG          left margin
 *   [10]      TMDS | IMG_W                      the band header
 *   [11..106] 96 words of RGB332 pixels, 4 px each, leftmost in byte 0
 *   [107..108] TMDS_REPEAT | IMG_X1, BG         right margin
 */
#define ROW_PX_OFF 11
#define ROW_PX_WORDS (IMG_W / 4) /* 96 */
#define ROW_WORDS (ROW_PX_OFF + ROW_PX_WORDS + 2) /* 109 */

/* fb rows alias rf_arena, which is idle whenever an image is displayed */
static_assert(IMG_H * ROW_WORDS * 4 <= (int)sizeof rf_arena,
              "framebuffer does not fit in rf_arena");
static inline uint32_t *fb_row(int y) {
    return (uint32_t *)(void *)&rf_arena[0][0] + (size_t)y * ROW_WORDS;
}

/* Preview rows are static: they are live exactly when the arena is not.
 * 16 latent rows, each feeding PREV_SCALE scanline slots. */
#define PREV_ROWS RF_ZHW
#define PREV_SCALE (IMG_H / RF_ZHW) /* 24 */
static uint32_t prev_row[PREV_ROWS][ROW_WORDS];

/* full-width background line, and the progress-bar line (live) */
static uint32_t bg_line[10];
static uint32_t bar_line[16];
static uint32_t vblank_off[7], vblank_on[7];

static void row_head(uint32_t *r) {
    r[0] = CMD_RAW_REPEAT | H_FRONT;
    r[1] = SYNC_V1_H1;
    r[2] = CMD_NOP;
    r[3] = CMD_RAW_REPEAT | H_SYNC;
    r[4] = SYNC_V1_H0;
    r[5] = CMD_NOP;
    r[6] = CMD_RAW_REPEAT | H_BACK;
    r[7] = SYNC_V1_H1;
}

/* rewrite a row's commands; leaves the 96 pixel words alone */
static void row_frame(uint32_t *r) {
    row_head(r);
    r[8] = CMD_TMDS_REPEAT | IMG_X0;
    r[9] = RGB332_W(BG);
    r[10] = CMD_TMDS | IMG_W;
    r[ROW_PX_OFF + ROW_PX_WORDS] = CMD_TMDS_REPEAT | IMG_X1;
    r[ROW_PX_OFF + ROW_PX_WORDS + 1] = RGB332_W(BG);
}

/* pixel x of a row, 4 px per word, leftmost pixel in the low byte */
static inline void row_px(uint32_t *r, int x, uint8_t c) {
    ((uint8_t *)(r + ROW_PX_OFF))[x] = c;
}

static void build_lines(void) {
    /* blanking lines merge back porch and active into one raw run */
    vblank_off[0] = CMD_RAW_REPEAT | H_FRONT;
    vblank_off[1] = SYNC_V1_H1;
    vblank_off[2] = CMD_RAW_REPEAT | H_SYNC;
    vblank_off[3] = SYNC_V1_H0;
    vblank_off[4] = CMD_RAW_REPEAT | (H_BACK + H_ACTIVE);
    vblank_off[5] = SYNC_V1_H1;
    vblank_off[6] = CMD_NOP;
    memcpy(vblank_on, vblank_off, sizeof vblank_on);
    vblank_on[1] = SYNC_V0_H1;
    vblank_on[3] = SYNC_V0_H0;
    vblank_on[5] = SYNC_V0_H1;

    row_head(bg_line);
    bg_line[8] = CMD_TMDS_REPEAT | H_ACTIVE;
    bg_line[9] = RGB332_W(BG);

    row_head(bar_line);
    bar_line[8] = CMD_TMDS_REPEAT | IMG_X0;
    bar_line[9] = RGB332_W(BG);
    bar_line[10] = CMD_TMDS_REPEAT | 1; /* live: done   */
    bar_line[11] = RGB332_W(BAR_ON);
    bar_line[12] = CMD_TMDS_REPEAT | (IMG_W - 1); /* live: remaining */
    bar_line[13] = RGB332_W(BAR_OFF);
    bar_line[14] = CMD_TMDS_REPEAT | IMG_X1;
    bar_line[15] = RGB332_W(BG);

    for (int y = 0; y < IMG_H; y++) row_frame(fb_row(y));
    for (int p = 0; p < PREV_ROWS; p++) row_frame(prev_row[p]);
}

static void bar_set(int done) {
    uint32_t on = RGB332_W(BAR_ON), off = RGB332_W(BAR_OFF);
    if (done < 0) { /* hidden */
        on = off = RGB332_W(BG);
        done = 1;
    }
    if (done < 1) done = 1;
    if (done > IMG_W - 1) done = IMG_W - 1;
    bar_line[10] = CMD_TMDS_REPEAT | (uint32_t)done;
    bar_line[11] = on;
    bar_line[12] = CMD_TMDS_REPEAT | (uint32_t)(IMG_W - done);
    bar_line[13] = off;
}

/* ---- DMA command list ---------------------------------------------------
 * 16-byte blocks in the DMA channel's NATIVE register order. */
typedef struct {
    const volatile void *read;
    volatile void *write;
    uint32_t count;
    uint32_t ctrl;
} cb_t;

#define N_CB (V_BLANK + V_ACTIVE + 1) /* 526 */
static cb_t cblist[N_CB] __attribute__((aligned(16)));
static uint16_t rowblk[IMG_H]; /* cblist index of each image-band line slot */
static const void *volatile rewind_src; /* holds &cblist[0] */

static int ch_ctl, ch_dat;

static void build_cblist(void) {
    uint32_t feed = (dma_channel_get_default_config(ch_dat).ctrl |
                     DMA_CH0_CTRL_TRIG_INCR_READ_BITS) &
                    ~DMA_CH0_CTRL_TRIG_INCR_WRITE_BITS;
    feed &= ~DMA_CH0_CTRL_TRIG_TREQ_SEL_BITS;
    feed |= (uint32_t)DREQ_HSTX << DMA_CH0_CTRL_TRIG_TREQ_SEL_LSB;
    feed &= ~DMA_CH0_CTRL_TRIG_CHAIN_TO_BITS;
    feed |= (uint32_t)ch_ctl << DMA_CH0_CTRL_TRIG_CHAIN_TO_LSB;
    feed |= DMA_CH0_CTRL_TRIG_IRQ_QUIET_BITS; /* 32-bit size is the default */

    volatile void *fifo = &hstx_fifo_hw->fifo;
    int n = 0;
    for (int y = 0; y < V_TOTAL; y++) {
        if (y < V_BLANK) {
            int vs = (y >= V_FRONT) && (y < V_FRONT + V_SYNC);
            cblist[n++] = (cb_t){vs ? vblank_on : vblank_off, fifo, 7, feed};
            continue;
        }
        int ay = y - V_BLANK;
        if (ay >= IMG_Y0 && ay < IMG_Y0 + IMG_H) {
            int iy = ay - IMG_Y0;
            rowblk[iy] = (uint16_t)n;
            cblist[n++] = (cb_t){fb_row(iy), fifo, ROW_WORDS, feed};
        } else if (ay >= BAR_Y0 && ay < BAR_Y0 + BAR_H) {
            cblist[n++] = (cb_t){bar_line, fifo, count_of(bar_line), feed};
        } else {
            cblist[n++] = (cb_t){bg_line, fifo, count_of(bg_line), feed};
        }
    }
    /* rewind: dat copies &cblist[0] into ctl's READ_ADDR trigger alias, with
     * chaining disabled (chain-to-self) and no DREQ to wait on */
    uint32_t rw = feed & ~(DMA_CH0_CTRL_TRIG_TREQ_SEL_BITS |
                           DMA_CH0_CTRL_TRIG_INCR_READ_BITS |
                           DMA_CH0_CTRL_TRIG_CHAIN_TO_BITS);
    rw |= (uint32_t)DREQ_FORCE << DMA_CH0_CTRL_TRIG_TREQ_SEL_LSB;
    rw |= (uint32_t)ch_dat << DMA_CH0_CTRL_TRIG_CHAIN_TO_LSB;
    cblist[n++] = (cb_t){&rewind_src, &dma_hw->ch[ch_ctl].al3_read_addr_trig,
                         1, rw};
}

/* ---- bring-up ----------------------------------------------------------- */

static void video_start(void) {
    hstx_ctrl_hw->expand_shift = EXPAND_SHIFT_RGB332;
    hstx_ctrl_hw->expand_tmds = EXPAND_TMDS_RGB332;
    hstx_ctrl_hw->csr = 0;
    hstx_ctrl_hw->csr = HSTX_CTRL_CSR_EXPAND_EN_BITS |
                        5u << HSTX_CTRL_CSR_CLKDIV_LSB |
                        5u << HSTX_CTRL_CSR_N_SHIFTS_LSB |
                        2u << HSTX_CTRL_CSR_SHIFT_LSB | HSTX_CTRL_CSR_EN_BITS;

    /* Fruit Jam runs CK, D0, D1, D2 in pin order from GP12 with the
     * negative leg on the even pin. */
    hstx_ctrl_hw->bit[0] = HSTX_CTRL_BIT0_CLK_BITS | HSTX_CTRL_BIT0_INV_BITS;
    hstx_ctrl_hw->bit[1] = HSTX_CTRL_BIT0_CLK_BITS;
    for (uint lane = 0; lane < 3; ++lane) {
        int bit = 2 + 2 * (int)lane;
        uint32_t sel = (lane * 10u) << HSTX_CTRL_BIT0_SEL_P_LSB |
                       (lane * 10u + 1) << HSTX_CTRL_BIT0_SEL_N_LSB;
        hstx_ctrl_hw->bit[bit] = sel | HSTX_CTRL_BIT0_INV_BITS;
        hstx_ctrl_hw->bit[bit + 1] = sel;
    }
    for (int p = ADAFRUIT_FRUIT_JAM_DVI_CKN_PIN;
         p <= ADAFRUIT_FRUIT_JAM_DVI_D2P_PIN; ++p)
        gpio_set_function((uint)p, GPIO_FUNC_HSTX);

    /* ctl: 4-word blocks onto dat's native registers, write ring 16B.
     * Retriggered by dat's chain, and by the rewind block at frame end. */
    dma_channel_config cc = dma_channel_get_default_config(ch_ctl);
    channel_config_set_read_increment(&cc, true);
    channel_config_set_write_increment(&cc, true);
    channel_config_set_ring(&cc, true, 4);
    dma_channel_configure(ch_ctl, &cc, &dma_hw->ch[ch_dat].read_addr, cblist, 4,
                          false);

    dma_channel_start((uint)ch_ctl);
}

static void video_stop(void) {
    /* Break the chain before aborting: either channel would otherwise
     * retrigger the one that just stopped. */
    dma_hw->ch[ch_ctl].al1_ctrl = 0;
    dma_hw->ch[ch_dat].al1_ctrl = 0;
    uint32_t mask = (1u << ch_ctl) | (1u << ch_dat);
    dma_hw->abort = mask;
    while (dma_hw->abort & mask) tight_loop_contents();
    hstx_ctrl_hw->csr = 0;
}

/* Frames per second in mHz */
uint32_t rf_dvi_frame_rate_mhz(uint32_t ms) {
    absolute_time_t t0 = get_absolute_time();
    absolute_time_t end = delayed_by_ms(t0, ms);
    uint32_t prev = dma_hw->ch[ch_ctl].read_addr, n = 0;
    do {
        uint32_t r = dma_hw->ch[ch_ctl].read_addr;
        if (r < prev) ++n;
        prev = r;
    } while (!time_reached(end));
    int64_t dt = absolute_time_diff_us(t0, get_absolute_time());
    return (uint32_t)((int64_t)n * 1000000000ll / dt);
}

/* The expander can come up desynced, and when it does it never
 * recovers. The only cure is to tear the pipeline down and try again.
 * Returns the attempt that worked, or -1. */
int rf_dvi_bringup_tries;
static int video_start_checked(void) {
    for (int try = 1; try <= 20; ++try) {
        video_start();
        uint32_t r = rf_dvi_frame_rate_mhz(200);
        if (r > 40000 && r < 90000) return try;
        video_stop();
        sleep_ms(5);
    }
    return -1;
}

/* Tear the scanout down and bring it back up; returns the attempt that
 * worked, or -1. Exposed so a host can recover a display without power-
 * cycling the board (main.c's "VR"). */
int rf_dvi_restart(void) {
    video_stop();
    sleep_ms(5);
    rf_dvi_bringup_tries = video_start_checked();
    return rf_dvi_bringup_tries;
}

/* ---- live content -------------------------------------------------------- */
static uint8_t gray_lut[256];

void rf_step_hook(void) { /* overrides the weak engine stub */
    if (!rf_progress) return;
    const int16_t *z = rf_z_state();
    const int P = RF_PATCH, G = RF_ZHW / RF_PATCH;
    for (int py = 0; py < PREV_ROWS; py++) {
        uint32_t *r = prev_row[py];
        for (int gx = 0; gx < RF_ZHW; gx++) {
            int16_t v = z[((py / P) * G + (gx / P)) * RF_PD + (py % P) * P +
                          (gx % P)];
            int g = 128 + (v >> 6);
            if (g < 0) g = 0;
            if (g > 255) g = 255;
            uint8_t c = gray_lut[g];
            memset((uint8_t *)(r + ROW_PX_OFF) + gx * PREV_SCALE, c,
                   PREV_SCALE);
        }
    }
    bar_set((int)rf_progress * IMG_W / (int)rf_progress_total);
}

/* Point the image band at the preview rows and arm the bar. One atomic
 * 32-bit READ_ADDR store per slot; every slot keeps its transfer count, so
 * the list stays valid at every instant and is never rebuilt. */
void rf_vga_invalidate(void) {
    for (int p = 0; p < PREV_ROWS; p++)
        memset((uint8_t *)(prev_row[p] + ROW_PX_OFF), 0, IMG_W);
    bar_set(1); /* armed, empty */
    __dmb();
    for (int iy = 0; iy < IMG_H; iy++)
        cblist[rowblk[iy]].read = prev_row[iy / PREV_SCALE];
    __dmb();
}

/* Floyd-Steinberg the 128x128 decode (rf_img) into the fb, upscaled 3x to
 * 384, then switch the band back.*/
void rf_vga_dither(void) {
    static int16_t err[2][IMG_W + 2][RF_IMG_CH];
    memset(err, 0, sizeof err);
    for (int y = 0; y < IMG_H; y++) {
        const uint8_t *src = rf_img + (size_t)(y / 3) * RF_IMG_HW * RF_IMG_CH;
        int16_t(*cur)[RF_IMG_CH] = err[y & 1] + 1;
        int16_t(*nxt)[RF_IMG_CH] = err[(y & 1) ^ 1] + 1;
        memset(err[(y & 1) ^ 1], 0, sizeof err[0]);
        int dir = (y & 1) ? -1 : 1;
        int x = (y & 1) ? IMG_W - 1 : 0;
        uint32_t *r = fb_row(y);
        row_frame(r);
        for (int i = 0; i < IMG_W; i++, x += dir) {
            const uint8_t *s = src + (size_t)(x / 3) * RF_IMG_CH;
            uint8_t q[RF_IMG_CH];
            for (int c = 0; c < RF_IMG_CH; c++) {
                /* blue gets 2 bits, red and green 3 (RGB332) */
                int nb = (RF_IMG_CH == 3 && c == 2) ? 2 : 3;
                int v = s[c] + ((cur[x][c] + 8) >> 4);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                int qn = v >> (8 - nb);
                /* reconstruct by bit replication, so 0->0 and max->255 */
                int rec = (nb == 3) ? ((qn << 5) | (qn << 2) | (qn >> 1))
                                    : ((qn << 6) | (qn << 4) | (qn << 2) | qn);
                int e = v - rec;
                cur[x + dir][c] = (int16_t)(cur[x + dir][c] + 7 * e);
                nxt[x - dir][c] = (int16_t)(nxt[x - dir][c] + 3 * e);
                nxt[x][c] = (int16_t)(nxt[x][c] + 5 * e);
                nxt[x + dir][c] = (int16_t)(nxt[x + dir][c] + e);
                q[c] = (uint8_t)qn;
            }
#if RF_IMG_CH == 3
            row_px(r, x, (uint8_t)RGB332(q[0], q[1], q[2]));
#else
            row_px(r, x, gray_lut[(q[0] << 5) | (q[0] << 2) | (q[0] >> 1)]);
#endif
        }
    }
    bar_set(-1); /* an image is up; the bar has nothing to say */
    __dmb();
    for (int iy = 0; iy < IMG_H; iy++) cblist[rowblk[iy]].read = fb_row(iy);
    __dmb();
}

#if RF_VGA_TEST
/* Boot pattern through the real fb path (arena rows, real command list) -
 * validates the scanout before any generation runs. Pattern 0 is the
 * gradient bars vga_noirq.c draws; */
static inline uint32_t rf_dvi_hash(uint32_t x, uint32_t y) {
    uint32_t h = x * 0x9E3779B1u + y * 0x85EBCA77u;
    h ^= h >> 16; h *= 0x7FEB352Du;
    h ^= h >> 15; h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}
void rf_dvi_test_pattern(int pat) {
    for (int y = 0; y < IMG_H; y++) {
        uint32_t *r = fb_row(y);
        row_frame(r);
        for (int x = 0; x < IMG_W; x++) {
            uint8_t c;
            if (pat == 1) {
                c = (rf_dvi_hash((uint32_t)x, (uint32_t)y) & 1u) ? 0xffu : 0u;
            } else if (pat == 2) {
                c = (uint8_t)RGB332(x * 8 / IMG_W, y * 8 / IMG_H,
                                    ((x + y) / 48) & 3);
            } else {
                int v = x * 8 / IMG_W;
                c = (uint8_t)((y < IMG_H / 3)       ? RGB332(v, 0, 0)
                              : (y < 2 * IMG_H / 3) ? RGB332(0, v, 0)
                                                    : RGB332(0, 0, v >> 1));
            }
            row_px(r, x, c);
        }
    }
    __dmb();
    for (int iy = 0; iy < IMG_H; iy++) cblist[rowblk[iy]].read = fb_row(iy);
}
#endif

/* ---- init ---------------------------------------------------------------- */
void rf_vga_init(void) {
    for (int i = 0; i < 256; i++)
        gray_lut[i] = (uint8_t)RGB332(i >> 5, i >> 5, i >> 6);

    ch_ctl = dma_claim_unused_channel(true);
    ch_dat = dma_claim_unused_channel(true);

    build_lines();
    build_cblist();
    rewind_src = &cblist[0];
    bar_set(-1);

    /* The scanout must not lose an arbitration race against the engine's
     * weight traffic: an underrun would break a TMDS command mid-flight. */
    bus_ctrl_hw->priority = BUSCTRL_BUS_PRIORITY_DMA_R_BITS;

#if RF_VGA_TEST
    rf_dvi_test_pattern(0);
#endif
    rf_dvi_bringup_tries = video_start_checked();
}
