"""Serialize a folded model dict (md) into model.bin for the C engine, and
emit end-to-end golden vectors via the exact-integer simulator.

Binary layout (little-endian, each array padded to 4-byte alignment, fixed
order - the C loader in engine/src/graph.c mirrors this exactly):

  header: 'RF25' ver=3 arch=1 K dim depth heads tokens zch zhw patch n_cond img_ch

Version 2 legacy note (still true in v3): weights consumed by the sparse
(zero-skipping) kernels are stored TRANSPOSED so the gather is contiguous:
Wfc2 as [4d][d] ([K][O]), and decoder layers with flag bit 3 ("wt") as
[3][3][C][O]. Values are unchanged - int32 accumulation order does not
affect results - so int_sim and goldens are identical.

Version 3 adds conditional table sets (n_cond = n_classes+1, or 1) and
multi-channel u8 output (img_ch). The adaLN step tables and final-norm
Gf/Bf repeat per cond set; weights and the folded Euler dt (M_v/s_v,
class-independent because the final-input scale is shared) do not.
  EXPLUT u16[256]
  POS i16[tokens*dim]
  M_zin i32[K], s_zin u8[K]
  step tables, cond-major then k: per y { per k { per b { G1 B1 G2 B2 i16[dim],
      Mproj i32[dim] sproj u8[dim], Mfc2 i32[dim] sfc2 u8[dim] },
      Gf Bf i16[dim] } }
  M_v i32[pd] s_v u8[pd] per k                        (pd = zch*patch^2)
  W_emb i8[dim*pd] b_emb/M_emb i32[dim] s_emb u8[dim]
  per block { Wqkv i8[3d*d] bqkv/Mqkv i32[3d] sqkv u8[3d], Gq Gk i16[d],
      M_sm i32 s_sm u8, M_att i32[d] s_att u8[d], Wproj i8[d*d] bproj i32[d],
      Wfc1 i8[4d*d] bfc1/Mfc1 i32[4d] sfc1 u8[4d], ACT_LUT i8[256],
      Wfc2 i8[d*4d] bfc2 i32[d] }
  W_final i8[pd*d] b_final i32[pd]
  M_zdec i32 s_zdec u8, n_dec u32
  per layer { C u32, O u32, flags u32 (up|relu<<1|u8<<2),
      W i8[O*9*C] b/M i32[O] s u8[O] }

Version 4 inserts the classifier-free-guidance section after M_v/s_v:
  n_w u32, per w { w_q8 u32, K x (M_v_c i32[pd] s_v_c u8[pd]),
      K x (M_v_n i32[pd] s_v_n u8[pd]) }
Version 5 (hires models) always writes the v4 n_w count (0 when no cfg)
and appends the hires head after the decoder layers:
  n_hires u32, per layer { C, O, flags, W i8[O*9*C] dense, b/M i32[O], s u8[O] }
Version 8 (act_sq folds; v7 is a legacy read-only format) shrinks storage:
  step tables carry ONE requant shift per entry (sproj/sfc2 = u8 scalar,
  mantissas pre-aligned by fold.renorm_Ms), and the guidance section is just
  n_w u32 + per w { w_q8 u32 } -- the per-pass M_v_c/M_v_n tables were dead
  since the int32-difference blend.
"""
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class _W:
    def __init__(self):
        self.buf = bytearray()

    def arr(self, a, dtype):
        b = np.ascontiguousarray(a, dtype=dtype).tobytes()
        self.buf.extend(b)
        while len(self.buf) % 4:
            self.buf.append(0)

    def u32(self, *vals):
        self.buf.extend(struct.pack(f"<{len(vals)}I", *vals))


def write_model_bin(md, path):
    m = md["meta"]
    K, d, depth = int(m["n_steps"]), int(m["dim"]), int(m["depth"])
    pd = int(m["zch"]) * int(m["patch"]) ** 2
    n_cond = int(m.get("n_cond", 1))
    cfg_w = list(m.get("cfg_w", []))
    hires = md.get("hires")
    # per-block attention mask (bit b set = block b keeps attention). Blocks in
    # md["drop_attn"] have their attention branch omitted (v6).
    drop_attn = set(md.get("drop_attn", ()))
    attn_mask = 0
    for b in range(depth):
        if b not in drop_attn:
            attn_mask |= (1 << b)
    # v4 only when CFG tables are present: cfg-free models stay byte-exact v3.
    # v5 = v4 + hires head section (and always writes the n_w count, 0 or not).
    # v6 = any build that drops an attention branch (adds the mask + omits those
    # blocks' attn weights). A full-attention build never becomes v6, so all
    # existing models stay byte-identical.
    # v7 = the precision pack: (a) relu2 square-requant MLP -- Mfc1/sfc1/ACT_LUT
    # replaced by M_actq i32[4d] + s_actq u8[4d] (h1 = rq(relu(acc)^2>>12, M));
    # (b) per-head softmax scales -- M_sm i32[heads] + s_sm u8[heads]; (c) the
    # exp LUT is 512 entries (1/64 exponent grid).
    act_sq = bool(m.get("act_sq"))
    # v8 = v7 + (a) ONE requant shift per step-table entry (sproj/sfc2 u8
    # scalar; fold.renorm_Ms aligned the mantissas -- the per-channel shift
    # array was ~information-free) and (b) the dead per-w M_v_c/M_v_n
    # guidance tables are gone (unused since the int32-difference blend).
    # act_sq folds always emit v8 now; shipped v7 blobs still parse.
    ver = 8 if act_sq else (
        6 if drop_attn else (5 if hires else (4 if cfg_w else 3)))
    w = _W()
    w.u32(0x35325246, ver, 1, K, d, depth, int(m["heads"]), int(m["tokens"]),
          int(m["zch"]), int(m["zhw"]), int(m["patch"]), n_cond,
          int(m.get("img_ch", 1)))
    if ver >= 6:
        w.u32(attn_mask)
    w.arr(md["EXPLUT"], np.uint16)
    w.arr(md["POS"], np.int16)
    w.arr(md["M_zin"], np.int32)
    w.arr(md["s_zin"], np.uint8)
    for y in range(n_cond):
        for k in range(K):
            for b in range(depth):
                st = md["step_tab"][y][k][b]
                for key in ("G1", "B1", "G2", "B2"):
                    w.arr(st[key], np.int16)
                w.arr(st["Mproj"], np.int32)
                if ver >= 8:  # scalar shift (renorm_Ms), padded to 4 in file
                    assert np.ndim(st["sproj"]) == 0 and np.ndim(st["sfc2"]) == 0
                    w.arr(np.array([st["sproj"]]), np.uint8)
                    w.arr(st["Mfc2"], np.int32)
                    w.arr(np.array([st["sfc2"]]), np.uint8)
                else:
                    w.arr(st["sproj"], np.uint8)
                    w.arr(st["Mfc2"], np.int32)
                    w.arr(st["sfc2"], np.uint8)
            w.arr(md["Gf"][y][k], np.int16)
            w.arr(md["Bf"][y][k], np.int16)
    for k in range(K):
        w.arr(md["M_v"][k], np.int32)
        w.arr(md["s_v"][k], np.uint8)
    if ver >= 4:  # guidance: w_q8 list; per-pass tables only pre-v8 (dead)
        w.u32(len(cfg_w))
        for j, wv in enumerate(cfg_w):
            w.u32(int(round(wv * 256)))
            if ver < 8:
                for k in range(K):
                    w.arr(md["M_v_c"][j][k], np.int32)
                    w.arr(md["s_v_c"][j][k], np.uint8)
                for k in range(K):
                    w.arr(md["M_v_n"][j][k], np.int32)
                    w.arr(md["s_v_n"][j][k], np.uint8)
    w.arr(md["W_emb"], np.int8)
    w.arr(md["b_emb"], np.int32)
    w.arr(md["M_emb"], np.int32)
    w.arr(md["s_emb"], np.uint8)
    for b in range(depth):
        blk = md["blocks"][b]
        if b not in drop_attn:  # attention weights omitted for dropped blocks
            w.arr(blk["Wqkv"], np.int8)
            w.arr(blk["bqkv"], np.int32)
            w.arr(blk["Mqkv"], np.int32)
            w.arr(blk["sqkv"], np.uint8)
            w.arr(blk["Gq"], np.int16)
            w.arr(blk["Gk"], np.int16)
            if ver >= 7:  # per-head softmax scales
                w.arr(blk["M_sm"], np.int32)
                w.arr(blk["s_sm"], np.uint8)
            else:
                w.u32(int(np.uint32(blk["M_sm"])))
                w.arr(np.array([blk["s_sm"]]), np.uint8)
            w.arr(blk["M_att"], np.int32)
            w.arr(blk["s_att"], np.uint8)
            w.arr(blk["Wproj"], np.int8)
            w.arr(blk["bproj"], np.int32)
        w.arr(blk["Wfc1"], np.int8)
        w.arr(blk["bfc1"], np.int32)
        if "M_actq" in blk:  # v7 relu2 square-requant (replaces Mfc1/sfc1/LUT)
            w.arr(blk["M_actq"], np.int32)
            w.arr(blk["s_actq"], np.uint8)
        else:
            w.arr(blk["Mfc1"], np.int32)
            w.arr(blk["sfc1"], np.uint8)
            w.arr(blk["ACT_LUT"], np.int8)
        w.arr(np.ascontiguousarray(blk["Wfc2"].T), np.int8)  # [K][O] for sparse
        w.arr(blk["bfc2"], np.int32)
    w.arr(md["W_final"], np.int8)
    w.arr(md["b_final"], np.int32)
    w.u32(int(np.uint32(md["M_zdec"])))
    w.arr(np.array([md["s_zdec"]]), np.uint8)
    w.u32(len(md["dec"]))
    for i, L in enumerate(md["dec"]):
        O, _, _, C = L["W"].shape
        # sparse path for the mid layers: input is post-ReLU (skippable zeros);
        # layer 0 sees the dense latent and the final u8 layer has O=1
        wt = 1 if (0 < i < len(md["dec"]) - 1) else 0
        flags = (int(L["up"]) | int(L.get("relu", 1)) << 1
                 | int(L.get("u8", 0)) << 2 | wt << 3)
        w.u32(C, O, flags)
        w.arr(np.transpose(L["W"], (1, 2, 3, 0)) if wt else L["W"], np.int8)
        w.arr(L["b"], np.int32)
        w.arr(L["M"], np.int32)
        w.arr(L["s"], np.uint8)
    if ver >= 5:  # hires head layers: dense [O][3][3][C], flags as above.
        # v6-without-hires (attn-dropped, no head) still writes n_hires=0 so the
        # section is present exactly where graph.c expects it.
        hlist = hires or []
        w.u32(len(hlist))
        for L in hlist:
            O, _, _, C = L["W"].shape
            w.u32(C, O, int(L["up"]) | int(L["relu"]) << 1 | int(L["u8"]) << 2)
            w.arr(L["W"], np.int8)
            w.arr(L["b"], np.int32)
            w.arr(L["M"], np.int32)
            w.arr(L["s"], np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(w.buf))
    print(f"wrote {path} ({len(w.buf)/1e6:.2f} MB)")


def write_goldens(md, seeds, out_dir, k_steps=4):
    """Run the exact-integer sim for each seed, save image + crc + z taps.
    k_steps=4 exercises the strided-schedule path (exported K is 8).
    Conditional models: cond = seed % n_cond -- the SAME rule is applied by
    the desktop runner and the firmware verify path, so no extra plumbing.
    CFG models additionally cycle guidance: w_idx = seed % (n_w+1) - 1
    (-1 = plain pass), so seeds 1..n_w cover every guided table set."""
    from quant.int_ops import crc32
    from quant.int_sim import IntSim

    sim = IntSim(md)
    n_w = len(md["meta"].get("cfg_w", []))
    ext = "gray" if int(md["meta"].get("img_ch", 1)) == 1 else "rgb"
    os.makedirs(out_dir, exist_ok=True)
    for seed in seeds:
        cond = seed % sim.n_cond
        w_idx = seed % (n_w + 1) - 1 if n_w else -1
        img, taps = sim.generate(seed, k_steps, cond, w_idx)
        np.savez(os.path.join(out_dir, f"golden_{seed}.npz"),
                 img=img, cond=np.uint32(cond), w_idx=np.int32(w_idx),
                 crc=np.uint32(crc32(img.tobytes())),
                 **{f"z{k}": t for k, t in enumerate(taps)})
        img.tofile(os.path.join(out_dir, f"golden_{seed}.{ext}"))
        print(f"seed {seed} cond {cond} w_idx {w_idx}: "
              f"crc32 {crc32(img.tobytes()):08x}")


def write_rf_cfg(md, path):
    """Emit the per-model compile-time header the C engine builds against."""
    m = md["meta"]
    pd = int(m["zch"]) * int(m["patch"]) ** 2
    img_hw = int(m["zhw"]) * (1 << sum(int(L["up"]) for L in md["dec"]))
    # sparse-decoder row-compaction bounds: max input-row nonzeros (W_in * C),
    # max input width, and max output channels (acc[] size) over the sparse
    # (mid) layers. W is stored [O,3,3,C], so shape[0]=O (out), shape[-1]=C (in).
    wi, nz_max, w_max, o_max = int(m["zhw"]), 1, 1, 1
    for i, L in enumerate(md["dec"]):
        if 0 < i < len(md["dec"]) - 1:
            nz_max = max(nz_max, wi * int(L["W"].shape[-1]))
            w_max = max(w_max, wi)
            o_max = max(o_max, int(L["W"].shape[0]))
        if int(L["up"]):
            wi *= 2
    lines = [
        "/* generated by quant/export.py - per-model engine configuration */",
        "#ifndef RF_CFG_H",
        "#define RF_CFG_H",
        f"#define RF_K_MAX {int(m['n_steps'])}",
        f"#define RF_DIM {int(m['dim'])}",
        f"#define RF_DEPTH {int(m['depth'])}",
        f"#define RF_HEADS {int(m['heads'])}",
        f"#define RF_TOKENS {int(m['tokens'])}",
        f"#define RF_ZCH {int(m['zch'])}",
        f"#define RF_ZHW {int(m['zhw'])}",
        f"#define RF_PATCH {int(m['patch'])}",
        f"#define RF_PD {pd}",
        f"#define RF_COND {int(m.get('n_cond', 1))}",
        f"#define RF_IMG_CH {int(m.get('img_ch', 1))}",
        f"#define RF_IMG_HW {img_hw}",
        "#define RF_MAX_DEC 16",
        f"#define RF_DEC_NZ_MAX {nz_max}",
        f"#define RF_DEC_W_MAX {w_max}",
        f"#define RF_DEC_O_MAX {o_max}",
        f"#define RF_HIRES {1 if md.get('hires') else 0}",
        f"#define RF_HIRES_CMID "
        f"{int(md['hires'][0]['W'].shape[0]) if md.get('hires') else 0}",
        f"#define RF_OUT_HW {img_hw * 2 if md.get('hires') else img_hw}",
        "#endif",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
