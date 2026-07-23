/* Dual-core rf_par_for: core0 takes rows [0, n/2), core1 [n/2, n).
 * Disjoint writes only, no locks; FIFO doorbell + done handshake.
 * Overrides the weak serial version in kernels_ref.c. */
#include "pico/multicore.h"

#include "rf_ops.h"

typedef void (*par_fn)(int, int, void *);
static par_fn volatile g_fn;
static void *volatile g_ctx;
static int volatile g_i0, g_i1;

static void core1_entry(void) {
    for (;;) {
        multicore_fifo_pop_blocking(); /* "go" */
        g_fn(g_i0, g_i1, g_ctx);
        __dmb();
        multicore_fifo_push_blocking(1); /* "done" */
    }
}

void rf_par_init(void) { multicore_launch_core1(core1_entry); }

int rf_core_id(void) { return (int)get_core_num(); }

void rf_par_for(int n, void (*fn)(int, int, void *), void *ctx) {
    int mid = n / 2;
    if (mid == 0) {
        fn(0, n, ctx);
        return;
    }
    g_fn = fn;
    g_ctx = ctx;
    g_i0 = mid;
    g_i1 = n;
    __dmb();
    multicore_fifo_push_blocking(0);
    fn(0, mid, ctx);
    multicore_fifo_pop_blocking();
    __dmb();
}
