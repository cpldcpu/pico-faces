/* DMA weight staging: overrides the weak synchronous copies in
 * kernels_ref.c. One DMA channel per arena slot, all free-running and
 * polled (no IRQs): rf_stage_start triggers the copy and returns, the
 * compute continues on both cores, rf_stage_wait blocks only if the slot
 * has not landed yet.
 *
 * Source addresses are rewritten to the uncached XIP alias so the streams
 * do not thrash the 16KB XIP cache. CRITICAL: the channels are PACED by a
 * shared DMA timer (~RF_STAGE_HZ bytes/s). Unpaced, the staging saturates
 * the QMI and every flash-resident instruction fetch stalls behind it -
 * measured on device as scanvideo missing-scanline flashes (its per-line
 * ISRs run from flash 31.5k times/s) and a net SLOWDOWN of the whole
 * generation. The steady-state need is only ~2.5 MB/s (192KB per block
 * over an ~85 ms compute window), so a gentle cap hides the copies just
 * as well while leaving the QMI to the XIP cache. */
#include "hardware/clocks.h"
#include "hardware/dma.h"
#include "hardware/regs/addressmap.h"

#include "rf_ops.h"

/* total staging throughput cap, bytes/sec (shared pacing timer) */
#ifndef RF_STAGE_HZ
#define RF_STAGE_HZ 5000000u
#endif

static int ch[RF_SLOT_N];
static uint pace_dreq;

void rf_stage_init(void) {
    for (int i = 0; i < RF_SLOT_N; i++)
        ch[i] = dma_claim_unused_channel(true);
    int t = dma_claim_unused_timer(true);
    /* one DREQ pulse per 32-bit word: X/Y = words_per_sec / sys_hz */
    uint32_t sys = clock_get_hz(clk_sys);
    uint32_t y = sys / (RF_STAGE_HZ / 4u);
    if (y > 0xffffu) y = 0xffffu;
    dma_timer_set_fraction(t, 1, (uint16_t)y);
    pace_dreq = dma_get_timer_dreq(t);
}

static const void *xip_nocache(const void *p) {
    uintptr_t a = (uintptr_t)p;
    if (a >= XIP_BASE && a < XIP_BASE + (16u << 20))
        return (const void *)(a - XIP_BASE + XIP_NOCACHE_NOALLOC_BASE);
    return p; /* already SRAM (random-model tests) */
}

void rf_stage_start(int slot, const void *src, size_t n) {
    /* a slot is never restarted while a reader holds it, but guard anyway */
    dma_channel_wait_for_finish_blocking(ch[slot]);
    dma_channel_config c = dma_channel_get_default_config(ch[slot]);
    channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
    channel_config_set_read_increment(&c, true);
    channel_config_set_write_increment(&c, true);
    channel_config_set_dreq(&c, pace_dreq);
    dma_channel_configure(ch[slot], &c, rf_stage_slot(slot), xip_nocache(src),
                          n / 4, true);
}

const int8_t *rf_stage_wait(int slot) {
    dma_channel_wait_for_finish_blocking(ch[slot]);
    return rf_stage_slot(slot);
}

void rf_stage_drain(void) {
    for (int i = 0; i < RF_SLOT_N; i++)
        dma_channel_wait_for_finish_blocking(ch[i]);
}
