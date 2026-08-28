/* Interrupt-free VGA scanout (640x480@60, Pimoroni VGA Demo Base).
 *
 * Reuses pico_scanvideo's PIO programs, pin map and timing values verbatim
 * (proven on this board) but replaces every piece of its CPU/IRQ machinery
 * with self-running DMA. Steady state: ZERO interrupts, zero CPU.
 *
 *   timing SM (pio0 SM3)  consumes 4 tokens per scanline. The whole frame
 *     (523 lines x 4 words, vsync bits baked per line) is precomputed; a
 *     data channel streams it (DREQ TX3) and chains to a 1-word reload
 *     channel that rewrites its read address and retriggers it. The C
 *     token of each ACTIVE line executes `irq 4`, releasing the pixel SM.
 *
 *   pixel SM (pio0 SM0)   parks at `wait irq 4`. A control channel walks
 *     a static list of 16-byte blocks (native READ/WRITE/COUNT/CTRL_TRIG
 *     order, write-ring of 16B onto the data channel's registers); each
 *     data block feeds composable tokens into the FIFO (DREQ TX0) and
 *     chains back to the control channel. The final block makes the data
 *     channel rewrite the control channel's READ_ADDR back to the list
 *     start (al3 trigger alias) - the frame loops entirely in hardware.
 *
 * Live content without IRQs: every image-line row block has the SAME
 * transfer count whether it points at a framebuffer row or a preview row,
 * so display<->preview switching is one atomic 32-bit READ_ADDR store per
 * line slot; the whole list is never rebuilt. During generation the
 * engine's rf_step_hook repaints 16 preview rows (the 16x16 latent, each
 * row scanned 24x by block repetition) and a progress-bar line - plain
 * stores from core 0 between par_for calls.
 *
 * Row body layout: [p0][count][p1..p383] halfwords - the RAW_RUN command
 * half-word lives in a shared 2-word prefix block; its first-pixel operand
 * and length field belong to the row buffer so one DMA block covers them.
 * fb rows alias rf_arena (idle while an image is displayed) + a static
 * spill; the preview rows and bar live in the spill region, which is dead
 * during generation (phase-exclusive, like the arena itself). */
#include <string.h>

#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/pio.h"
#include "hardware/sync.h"
#include "pico/scanvideo.h" /* pin defines + PIXEL_FROM_RGB macros */
#include "pico/stdlib.h"

#include "scanvideo.pio.h"
#include "timing.pio.h"

#include "rf_model.h"
#include "rf_ops.h"

extern uint8_t rf_img[RF_IMG_HW * RF_IMG_HW * RF_IMG_CH];
extern volatile uint8_t rf_progress;
extern volatile uint8_t rf_progress_total;
const int16_t *rf_z_state(void);

#define IMG_X0 128
#define IMG_W 384
#define IMG_H 384
#define LINES 480
#define IMG_Y0 48
#define BAR_Y0 450
#define BAR_H 12

/* vga_timing_640x480_60_default (25 MHz pixel clock, sync active-low) */
#define H_FRONT 16
#define H_PULSE 64
#define H_TOTAL 800
#define H_BACK (H_TOTAL - H_FRONT - H_PULSE - 640)
#define V_ACTIVE 480
#define V_FRONT 1
#define V_PULSE 2
#define V_TOTAL 523
#define PIX_HZ 25000000u

#define VID_PIO pio0
#define SM_PIX 0u
#define SM_TIM 3u

/* composable commands = absolute PIO addresses (program loaded at 0) */
#define C_COLOR_RUN video_24mhz_composable_default_offset_color_run
#define C_RAW_RUN video_24mhz_composable_default_offset_raw_run
#define C_RAW_1P video_24mhz_composable_default_offset_raw_1p
#define C_EOL_ALIGN video_24mhz_composable_default_offset_end_of_scanline_ALIGN
#define C_EOL_SKIP \
    video_24mhz_composable_default_offset_end_of_scanline_skip_ALIGN

#define BG PICO_SCANVIDEO_PIXEL_FROM_RGB8(24, 24, 40)
#define BAR_ON PICO_SCANVIDEO_PIXEL_FROM_RGB8(80, 220, 120)
#define BAR_OFF PICO_SCANVIDEO_PIXEL_FROM_RGB8(50, 50, 60)

/* ---- row buffers -------------------------------------------------------- */
#define ROW_HW (IMG_W + 2)      /* [p0][count][p1..p383][next-cmd tail] */
#define ROW_WORDS (ROW_HW / 2)  /* 193 */
#define ROW_TAIL (IMG_W + 1)    /* index of the trailing command hw */
#define FB_ARENA_ROWS ((int)(sizeof rf_arena / (ROW_HW * 2)))
static uint16_t fb_spill[IMG_H - FB_ARENA_ROWS][ROW_HW];
static uint16_t *fb_row[IMG_H];

/* preview rows + bar alias the spill: fb is invalid whenever they are used */
#define PREV_ROWS RF_ZHW
#define PREV_SCALE (IMG_W / RF_ZHW) /* 24 */
static uint16_t (*const prev_row)[ROW_HW] = fb_spill;

/* set pixel x of a row-body buffer (layout [p0][count][p1..]) */
static inline void row_px(uint16_t *body, int x, uint16_t v) {
    body[x ? x + 1 : 0] = v;
}

/* ---- static token buffers ----------------------------------------------- */
static uint16_t px_prefix[4];  /* |COLOR_RUN|bg|125|RAW_RUN|            2w */
static uint16_t px_suffix[6];  /* |COLOR_RUN|bg|124|RAW_1P|blk|EOL|     3w */
static uint16_t px_bg8[16];    /* full 640px background line            8w */
static uint16_t px_bar8[16];   /* margin|done|rest|margin|blk|EOL       8w */

static void build_tokens(void) {
    px_prefix[0] = C_COLOR_RUN;
    px_prefix[1] = BG;
    px_prefix[2] = IMG_X0 - 3;
    px_prefix[3] = C_RAW_RUN;

    /* the suffix's COLOR_RUN command half-word lives at the END of each
     * row buffer (row tail): RAW_RUN consumes exactly [p0][cnt][W-1 px]
     * [next cmd] = 386 halfwords, keeping the row block whole-word with
     * no dead pad. The suffix block starts at the run's color operand. */
    px_suffix[0] = BG;
    px_suffix[1] = (640 - IMG_X0 - IMG_W - 1) - 3; /* 127 px */
    px_suffix[2] = C_RAW_1P;
    px_suffix[3] = 0; /* terminating black (sticky-OUT contract) */
    px_suffix[4] = C_EOL_SKIP;
    px_suffix[5] = 0xffff; /* discarded by EOL_SKIP's out null,32 */

    const uint16_t bgw[4] = {213, 213, 106, 107}; /* + black = 640 */
    for (int i = 0; i < 4; i++) {
        px_bg8[3 * i] = C_COLOR_RUN;
        px_bg8[3 * i + 1] = BG;
        px_bg8[3 * i + 2] = (uint16_t)(bgw[i] - 3);
    }
    px_bg8[12] = C_RAW_1P;
    px_bg8[13] = 0;
    px_bg8[14] = C_EOL_SKIP;
    px_bg8[15] = 0xffff;

    px_bar8[0] = C_COLOR_RUN;
    px_bar8[1] = BG;
    px_bar8[2] = IMG_X0 - 3;
    px_bar8[3] = C_COLOR_RUN;
    px_bar8[4] = BAR_ON;
    px_bar8[5] = 0; /* done-3: live */
    px_bar8[6] = C_COLOR_RUN;
    px_bar8[7] = BAR_OFF;
    px_bar8[8] = IMG_W - 6 - 3; /* rest-3: live */
    px_bar8[9] = C_COLOR_RUN;
    px_bar8[10] = BG;
    px_bar8[11] = (640 - IMG_X0 - IMG_W - 1) - 3;
    px_bar8[12] = C_RAW_1P;
    px_bar8[13] = 0;
    px_bar8[14] = C_EOL_SKIP;
    px_bar8[15] = 0xffff;
}

static void bar_set(int done) {
    if (done < 3) done = 3;
    if (done > IMG_W - 3) done = IMG_W - 3;
    px_bar8[5] = (uint16_t)(done - 3);
    px_bar8[8] = (uint16_t)((IMG_W - done) - 3);
}

/* ---- pixel control list -------------------------------------------------
 * 16-byte blocks in the DMA channel's NATIVE register order
 * (READ_ADDR, WRITE_ADDR, TRANS_COUNT, CTRL_TRIG). */
typedef struct {
    const volatile void *read;
    volatile void *write;
    uint32_t count;
    uint32_t ctrl;
} cb_t;

#define N_CB (96 + IMG_H * 3 + 1)
static cb_t cblist[N_CB] __attribute__((aligned(16)));
static uint16_t rowblk[IMG_H];      /* cblist index of each row-body block */
static uint16_t barblk[BAR_H];      /* cblist index of each bar-line block */
static const void *volatile rewind_src; /* holds &cblist[0] */

static int ch_px_ctl, ch_px_dat, ch_tim_dat, ch_tim_rld;

static void build_cblist(void) {
    uint32_t feed = (dma_channel_get_default_config(ch_px_dat).ctrl |
                     DMA_CH0_CTRL_TRIG_INCR_READ_BITS) &
                    ~DMA_CH0_CTRL_TRIG_INCR_WRITE_BITS;
    feed &= ~DMA_CH0_CTRL_TRIG_TREQ_SEL_BITS;
    feed |= (DREQ_PIO0_TX0 + SM_PIX) << DMA_CH0_CTRL_TRIG_TREQ_SEL_LSB;
    feed &= ~DMA_CH0_CTRL_TRIG_CHAIN_TO_BITS;
    feed |= (uint32_t)ch_px_ctl << DMA_CH0_CTRL_TRIG_CHAIN_TO_LSB;
    feed |= DMA_CH0_CTRL_TRIG_IRQ_QUIET_BITS; /* 32-bit size is default */

    int n = 0;
    for (int y = 0; y < LINES; y++) {
        if (y >= IMG_Y0 && y < IMG_Y0 + IMG_H) {
            int iy = y - IMG_Y0;
            cblist[n++] = (cb_t){px_prefix, &VID_PIO->txf[SM_PIX], 2, feed};
            rowblk[iy] = (uint16_t)n;
            cblist[n++] =
                (cb_t){fb_row[iy], &VID_PIO->txf[SM_PIX], ROW_WORDS, feed};
            cblist[n++] = (cb_t){px_suffix, &VID_PIO->txf[SM_PIX], 3, feed};
        } else {
            if (y >= BAR_Y0 && y < BAR_Y0 + BAR_H)
                barblk[y - BAR_Y0] = (uint16_t)n;
            cblist[n++] = (cb_t){px_bg8, &VID_PIO->txf[SM_PIX], 8, feed};
        }
    }
    /* rewind: data channel copies &cblist[0] into the control channel's
     * READ_ADDR trigger alias; chain disabled (self), no dreq */
    uint32_t rw = feed & ~(DMA_CH0_CTRL_TRIG_TREQ_SEL_BITS |
                           DMA_CH0_CTRL_TRIG_INCR_READ_BITS |
                           DMA_CH0_CTRL_TRIG_CHAIN_TO_BITS);
    rw |= DREQ_FORCE << DMA_CH0_CTRL_TRIG_TREQ_SEL_LSB;
    rw |= (uint32_t)ch_px_dat << DMA_CH0_CTRL_TRIG_CHAIN_TO_LSB;
    cblist[n++] = (cb_t){&rewind_src,
                         &dma_hw->ch[ch_px_ctl].al3_read_addr_trig, 1, rw};
}

/* ---- timing token ring --------------------------------------------------- */
static uint32_t timing_ring[V_TOTAL * 4];
static const void *volatile timing_src; /* holds &timing_ring[0] */

#define TIMING_CYCLE 3u
enum { ST_IRQ0 = 0, ST_IRQ1 = 1, ST_IRQ4 = 2, ST_CLR4 = 3 };
static uint32_t tenc(int st, uint32_t len, uint32_t pins) {
    return (uint32_t)video_htiming_states_program.instructions[st] |
           ((len - TIMING_CYCLE) << 16) | (pins << 29);
}

static void build_timing_ring(void) {
    /* h/v sync active-low: pulse level 0. VSYNC = OUT-pins bit1 = word
     * bit 30; DEN (bit2, unused pin here) set during active video only. */
    const uint32_t vs_idle = 1u << 30;
    uint32_t a = tenc(ST_CLR4, 4, 0);       /* sync pulse head (CPU poke */
    uint32_t b1 = tenc(ST_CLR4, H_PULSE - 4, 0); /* token neutered)      */
    uint32_t b2 = tenc(ST_CLR4, H_BACK, 1);
    uint32_t c_act = tenc(ST_IRQ4, H_TOTAL - H_BACK - H_PULSE, 4u | 1u);
    uint32_t c_bln = tenc(ST_CLR4, H_TOTAL - H_BACK - H_PULSE, 1u);
    for (int y = 0; y < V_TOTAL; y++) {
        int pulse =
            (y >= V_ACTIVE + V_FRONT) && (y < V_ACTIVE + V_FRONT + V_PULSE);
        uint32_t vs = pulse ? 0 : vs_idle;
        uint32_t *t = &timing_ring[y * 4];
        t[0] = a | vs;
        t[1] = b1 | vs;
        t[2] = b2 | vs;
        t[3] = (y < V_ACTIVE ? c_act : c_bln) | vs;
    }
}

/* ---- live content -------------------------------------------------------- */
static uint16_t gray_lut[256];

void rf_step_hook(void) { /* overrides the weak engine stub */
    if (!rf_progress) return;
    const int16_t *z = rf_z_state();
    const int P = RF_PATCH, G = RF_ZHW / RF_PATCH;
    for (int py = 0; py < PREV_ROWS; py++) {
        uint16_t *body = prev_row[py];
        for (int gx = 0; gx < RF_ZHW; gx++) {
            int16_t v = z[((py / P) * G + (gx / P)) * RF_PD +
                          (py % P) * P + (gx % P)];
            int g = 128 + (v >> 6);
            if (g < 0) g = 0;
            if (g > 255) g = 255;
            uint16_t c = gray_lut[g];
            for (int r = 0; r < PREV_SCALE; r++)
                row_px(body, gx * PREV_SCALE + r, c);
        }
    }
    bar_set((int)rf_progress * IMG_W / (int)rf_progress_total);
}

/* switch the image band to the preview rows + arm the bar (atomic
 * READ_ADDR stores; every slot keeps its transfer count) */
void rf_vga_invalidate(void) {
    for (int py = 0; py < PREV_ROWS; py++) {
        prev_row[py][0] = 0;
        prev_row[py][1] = IMG_W - 3;
        for (int x = 1; x < IMG_W; x++) prev_row[py][x + 1] = 0;
        prev_row[py][ROW_TAIL] = C_COLOR_RUN;
    }
    bar_set(3);
    for (int iy = 0; iy < IMG_H; iy++)
        cblist[rowblk[iy]].read = prev_row[iy / PREV_SCALE];
    for (int i = 0; i < BAR_H; i++) cblist[barblk[i]].read = px_bar8;
    __dmb();
}

/* Floyd-Steinberg dither of the 128 decode (rf_img) into the fb, upscaled
 * 3x to 384, then switch the image band back to the fb rows. rf_img is
 * arena-free and still valid here. */
void rf_vga_dither(void) {
    static int16_t err[2][IMG_W + 2][RF_IMG_CH];
    memset(err, 0, sizeof err);
    for (int y = 0; y < IMG_H; y++) {
        const uint8_t *row = rf_img + (size_t)(y / 3) * 128 * RF_IMG_CH;
        int16_t (*cur)[RF_IMG_CH] = err[y & 1] + 1;
        int16_t (*nxt)[RF_IMG_CH] = err[(y & 1) ^ 1] + 1;
        memset(err[(y & 1) ^ 1], 0, sizeof err[0]);
        int dir = (y & 1) ? -1 : 1;
        int x = (y & 1) ? IMG_W - 1 : 0;
        uint16_t *body = fb_row[y];
        body[1] = IMG_W - 3;
        body[ROW_TAIL] = C_COLOR_RUN;
        for (int i = 0; i < IMG_W; i++, x += dir) {
            const uint8_t *s = row + (size_t)(x / 3) * RF_IMG_CH;
            uint8_t q[RF_IMG_CH];
            for (int c = 0; c < RF_IMG_CH; c++) {
                int v = s[c] + ((cur[x][c] + 8) >> 4);
                if (v < 0) v = 0;
                if (v > 255) v = 255;
                int q5 = v >> 3;
                int e = v - ((q5 << 3) | (q5 >> 2));
                cur[x + dir][c] = (int16_t)(cur[x + dir][c] + 7 * e);
                nxt[x - dir][c] = (int16_t)(nxt[x - dir][c] + 3 * e);
                nxt[x][c] = (int16_t)(nxt[x][c] + 5 * e);
                nxt[x + dir][c] = (int16_t)(nxt[x + dir][c] + e);
                q[c] = (uint8_t)q5;
            }
#if RF_IMG_CH == 3
            row_px(body, x, PICO_SCANVIDEO_PIXEL_FROM_RGB5(q[0], q[1], q[2]));
#else
            row_px(body, x, (uint16_t)(q[0] * 0x0421u));
#endif
        }
    }
    __dmb();
    for (int iy = 0; iy < IMG_H; iy++) cblist[rowblk[iy]].read = fb_row[iy];
    for (int i = 0; i < BAR_H; i++) cblist[barblk[i]].read = px_bg8;
}

/* ---- init ---------------------------------------------------------------- */
void rf_vga_init(void) {
    for (int i = 0; i < 256; i++)
        gray_lut[i] = PICO_SCANVIDEO_PIXEL_FROM_RGB8(i, i, i);
    for (int y = 0; y < IMG_H; y++) {
        fb_row[y] = (y < FB_ARENA_ROWS)
                        ? (uint16_t *)&rf_arena[0][0] + (size_t)y * ROW_HW
                        : fb_spill[y - FB_ARENA_ROWS];
        fb_row[y][0] = 0;
        fb_row[y][1] = IMG_W - 3;
        fb_row[y][ROW_TAIL] = C_COLOR_RUN;
    }

    /* pins: RGB555 on GPIO0..15, HSYNC/VSYNC on 16/17 (scanvideo map) */
    for (int p = 0; p < 16; p++)
        if (p != 5) gpio_set_function((uint)p, GPIO_FUNC_PIO0);
    gpio_set_function(PICO_SCANVIDEO_SYNC_PIN_BASE, GPIO_FUNC_PIO0);
    gpio_set_function(PICO_SCANVIDEO_SYNC_PIN_BASE + 1, GPIO_FUNC_PIO0);

    /* composable program must sit at PIO address 0 (tokens hold absolute
     * jump targets). xscale=1 delay patch: +1 cycle on the extra1 slots. */
    uint16_t instr[32];
    pio_program_t prog = video_24mhz_composable_default_program;
    memcpy(instr, prog.instructions, prog.length * sizeof(uint16_t));
    instr[video_24mhz_composable_default_offset_delay_a_1] |= 1u << 8;
    instr[video_24mhz_composable_default_offset_delay_b_1] |= 1u << 8;
    instr[video_24mhz_composable_default_offset_delay_f_1] |= 1u << 8;
    prog.instructions = instr;
    pio_add_program_at_offset(VID_PIO, &prog, 0);
    uint tim_off = pio_add_program(VID_PIO, &video_htiming_program);

    uint32_t div2 = clock_get_hz(clk_sys) / PIX_HZ; /* SM at 2x pixel clk */

    pio_sm_config pc = video_24mhz_composable_default_program_get_default_config(0);
    sm_config_set_out_pins(&pc, PICO_SCANVIDEO_COLOR_PIN_BASE, 16);
    sm_config_set_out_shift(&pc, true, true, 32);
    sm_config_set_fifo_join(&pc, PIO_FIFO_JOIN_TX);
    sm_config_set_out_special(&pc, true, false, 0); /* sticky OUT */
    sm_config_set_clkdiv_int_frac(&pc, div2 / 2, (uint8_t)((div2 & 1u) << 7));
    pio_sm_init(VID_PIO, SM_PIX, 0, &pc);
    pio_sm_set_consecutive_pindirs(VID_PIO, SM_PIX,
                                   PICO_SCANVIDEO_COLOR_PIN_BASE, 16, true);

    pio_sm_config tc = video_htiming_program_get_default_config(tim_off);
    sm_config_set_out_pins(&tc, PICO_SCANVIDEO_SYNC_PIN_BASE, 2);
    sm_config_set_sideset_pins(&tc, PICO_SCANVIDEO_SYNC_PIN_BASE + 2);
    sm_config_set_out_shift(&tc, true, true, 32);
    sm_config_set_clkdiv_int_frac(&tc, div2 / 2, (uint8_t)((div2 & 1u) << 7));
    pio_sm_init(VID_PIO, SM_TIM, tim_off, &tc);
    pio_sm_set_consecutive_pindirs(VID_PIO, SM_TIM,
                                   PICO_SCANVIDEO_SYNC_PIN_BASE, 2, true);

    /* DMA plumbing */
    ch_px_ctl = dma_claim_unused_channel(true);
    ch_px_dat = dma_claim_unused_channel(true);
    ch_tim_dat = dma_claim_unused_channel(true);
    ch_tim_rld = dma_claim_unused_channel(true);

    build_tokens();
    build_timing_ring();
    build_cblist();
    rewind_src = &cblist[0];
    timing_src = &timing_ring[0];
    bar_set(3);

    /* control channel: 4-word blocks onto the data channel's native regs,
     * write ring 16B; retriggered by the data channel's chain (and by the
     * rewind block's al3 write at frame end) */
    dma_channel_config cc = dma_channel_get_default_config(ch_px_ctl);
    channel_config_set_read_increment(&cc, true);
    channel_config_set_write_increment(&cc, true);
    channel_config_set_ring(&cc, true, 4); /* wrap write addr every 16B */
    dma_channel_configure(ch_px_ctl, &cc, &dma_hw->ch[ch_px_dat].read_addr,
                          cblist, 4, false);

    /* timing stream: whole frame per trigger, chain to the reloader */
    dma_channel_config tdc = dma_channel_get_default_config(ch_tim_dat);
    channel_config_set_dreq(&tdc, DREQ_PIO0_TX0 + SM_TIM);
    channel_config_set_read_increment(&tdc, true);
    channel_config_set_chain_to(&tdc, (uint)ch_tim_rld);
    dma_channel_configure(ch_tim_dat, &tdc, &VID_PIO->txf[SM_TIM],
                          timing_ring, V_TOTAL * 4, false);

    dma_channel_config trc = dma_channel_get_default_config(ch_tim_rld);
    channel_config_set_read_increment(&trc, false);
    channel_config_set_write_increment(&trc, false);
    dma_channel_configure(ch_tim_rld, &trc,
                          &dma_hw->ch[ch_tim_dat].al3_read_addr_trig,
                          &timing_src, 1, false);

#if RF_VGA_TEST
    /* boot test pattern: RGB gradient bars through the whole fb path
     * (arena + spill rows, RAW_RUN stream) - validates the static scanout
     * before any generation runs */
    for (int y = 0; y < IMG_H; y++) {
        uint16_t *body = fb_row[y];
        body[1] = IMG_W - 3;
        body[ROW_TAIL] = C_COLOR_RUN;
        for (int x = 0; x < IMG_W; x++) {
            int v = (x * 32) / IMG_W;
            uint16_t c = (y < 128)   ? PICO_SCANVIDEO_PIXEL_FROM_RGB5(v, 0, 0)
                         : (y < 256) ? PICO_SCANVIDEO_PIXEL_FROM_RGB5(0, v, 0)
                                     : PICO_SCANVIDEO_PIXEL_FROM_RGB5(0, 0, v);
            row_px(body, x, c);
        }
    }
#endif

    /* preload FIFOs, then start both SMs in the same cycle */
    pio_interrupt_clear(VID_PIO, 4);
    dma_channel_start((uint)ch_px_ctl);
    dma_channel_start((uint)ch_tim_dat);
    busy_wait_us(50); /* let the FIFOs fill */
    pio_sm_exec(VID_PIO, SM_PIX,
                pio_encode_jmp(video_24mhz_composable_default_offset_entry_point));
    pio_sm_exec(VID_PIO, SM_TIM,
                pio_encode_jmp(tim_off + video_htiming_offset_entry_point));
    pio_enable_sm_mask_in_sync(VID_PIO, (1u << SM_PIX) | (1u << SM_TIM));
}
