/* model.bin baked into flash. Path injected via -DMODEL_BIN=<path>. */
#define XSTR(s) STR(s)
#define STR(s) #s

.section .rodata
.balign 4
.global rf_model_blob
.global rf_model_blob_end
rf_model_blob:
.incbin XSTR(MODEL_BIN)
rf_model_blob_end:
