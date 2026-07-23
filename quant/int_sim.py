"""Exact-integer simulator of the deployed pipeline (DiT backbone + VAE
decoder). This is the golden reference the C engine must match byte-for-byte.

Everything from `seed` to the 128x128 uint8 image is integer arithmetic from
quant/int_ops.py. The model is a dict of numpy arrays ("md") produced by
quant/fold.py (or tests/make_random_model.py for engine bring-up).

Device dataflow per Euler step k (schedule folded at export):
  z tokens i16[64][16] (Q12 at init: CLT-12 noise, sigma=4096)
  -> requant to i8 (M_zin[k])                          [per-step: z range shrinks]
  -> embed linear acc16 into residual := pos_i16 copy  [res i16[64][128]]
  -> 8x block:
       rmsnorm(res, G1[k][b], B1[k][b]) -> i8          [adaLN scale/shift folded]
       qkv linear -> q,k (i8, pre-norm), v (i8)
       per-head qk-RMSNorm (K=32, gains Gq/Gk, B=0)
       scores i32 = q.k ; int softmax (M_sm) -> p i8 (1/127)
       attn = p @ v -> i32 -> requant (M_att) -> i8
       proj linear acc16 -> res                        [adaLN gate folded in M]
       rmsnorm(res, G2,B2) -> i8 ; fc1 -> i8 ; ACT_LUT ; fc2 acc16 -> res
  -> final rmsnorm (Gf,Bf) -> final linear acc16 into z tokens
     with M_v[k] < 0 folding (-dt[k] * scale): the Euler update itself.
After K steps: unpatchify z -> HWC i16, requant (M_zdec) -> i8,
10-layer conv decoder (NN-upsample fused), final conv -> uint8 (bias 128).

Token order: row-major 8x8 patch grid; token feature = c*4 + py*2 + px.
"""
import numpy as np

from quant.int_ops import (
    I64, conv3x3_i8, gaussian_clt12, linear_i8, linear_i8_acc16, lut_i8,
    pcg32_init, requant, requant_i16_to_i8, rmsnorm_i16_to_i8, sat16,
    softmax_i32_to_i8,
)


class IntSim:
    def __init__(self, md):
        self.md = md
        m = md["meta"]
        self.K = int(m["n_steps"])
        self.depth = int(m["depth"])
        self.dim = int(m["dim"])
        self.heads = int(m["heads"])
        self.tokens = int(m["tokens"])
        self.hd = self.dim // self.heads
        self.zch = int(m.get("zch", 4))
        self.zhw = int(m.get("zhw", 16))
        self.patch = int(m.get("patch", 2))
        self.pd = self.zch * self.patch ** 2
        self.g = self.zhw // self.patch
        self.n_cond = int(m.get("n_cond", 1))

    # ---- backbone ----------------------------------------------------------
    def _attn_branch(self, res, blk, st, md, T, D, H, hd):
        """int8 attention branch; updates res in place (res += proj(attn(...))).
        Extracted so dit_step can skip it for dropped-attention blocks."""
        xa = np.empty((T, D), np.int8)
        for t in range(T):
            xa[t] = rmsnorm_i16_to_i8(res[t], st["G1"], st["B1"])
        qkv = np.empty((T, 3 * D), np.int8)
        for t in range(T):
            qkv[t] = linear_i8(xa[t], blk["Wqkv"], blk["bqkv"],
                               blk["Mqkv"], blk["sqkv"])
        q = qkv[:, :D].reshape(T, H, hd)
        kk = qkv[:, D:2 * D].reshape(T, H, hd)
        v = qkv[:, 2 * D:].reshape(T, H, hd)
        qn = np.empty_like(q)
        kn = np.empty_like(kk)
        zb = np.zeros(hd, np.int16)
        for t in range(T):
            for h in range(H):
                qn[t, h] = rmsnorm_i16_to_i8(q[t, h].astype(np.int16),
                                             blk["Gq"][h], zb)
                kn[t, h] = rmsnorm_i16_to_i8(kk[t, h].astype(np.int16),
                                             blk["Gk"][h], zb)
        attn = np.empty((T, D), np.int8)
        for h in range(H):
            # v7: per-head softmax scales (scores carry s_qk[h]^2)
            M_sm = blk["M_sm"][h] if np.ndim(blk["M_sm"]) else blk["M_sm"]
            s_sm = blk["s_sm"][h] if np.ndim(blk["s_sm"]) else blk["s_sm"]
            scores = qn[:, h].astype(np.int32) @ kn[:, h].astype(np.int32).T
            p = np.empty((T, T), np.int8)
            for t in range(T):
                p[t] = softmax_i32_to_i8(scores[t], M_sm, s_sm, md["EXPLUT"])
            acc = p.astype(np.int32) @ v[:, h].astype(np.int32)  # [T, hd]
            y = requant(acc, blk["M_att"][h * hd:(h + 1) * hd].astype(I64),
                        blk["s_att"][h * hd:(h + 1) * hd].astype(I64))
            attn[:, h * hd:(h + 1) * hd] = np.clip(y, -127, 127).astype(np.int8)
        for t in range(T):
            res[t] = linear_i8_acc16(attn[t], blk["Wproj"], blk["bproj"],
                                     st["Mproj"], st["sproj"], res[t])

    def dit_step(self, z_tok, k, s_shift=0, cond=0, z_in=None, mv=None,
                 sv=None, return_vacc=False):
        """One DiT pass. z_in: where the pass READS z (defaults to z_tok;
        CFG passes read the saved pre-step state while both accumulate into
        z_tok). mv/sv: Euler-fold override (CFG per-pass rescaled M_v).
        return_vacc: return raw int32 final-layer accumulator (for the
        precision-preserving guided blend) instead of updating z_tok."""
        md = self.md
        T, D, H, hd = self.tokens, self.dim, self.heads, self.hd
        if z_in is None:
            z_in = z_tok

        x_in = np.empty((T, self.pd), np.int8)
        for t in range(T):
            x_in[t] = requant_i16_to_i8(z_in[t], md["M_zin"][k], md["s_zin"][k])

        res = md["POS"].copy()  # i16 [T, D], residual units
        for t in range(T):
            res[t] = linear_i8_acc16(x_in[t], md["W_emb"], md["b_emb"],
                                     md["M_emb"], md["s_emb"], res[t])

        drop_attn = set(md.get("drop_attn", ()))
        for b in range(self.depth):
            blk = md["blocks"][b]
            st = md["step_tab"][cond][k][b]

            # attention branch -- skipped for blocks in drop_attn (residual
            # passes through untouched, exactly as omitting the weights does)
            if b not in drop_attn:
                self._attn_branch(res, blk, st, md, T, D, H, hd)

            # mlp branch
            for t in range(T):
                xm = rmsnorm_i16_to_i8(res[t], st["G2"], st["B2"])
                if "M_actq" in blk:
                    # relu2 square-requant (v7): square the RAW fc1 accumulator
                    # (no intermediate requant/LUT/clip); u*M stays in int64 via
                    # the 2^12 pre-shift folded into M_actq. Per-channel h1
                    # scales are folded into Wfc2's columns.
                    acc = (blk["Wfc1"].astype(I64) @ xm.astype(I64)
                           + blk["bfc1"].astype(I64))
                    u = (np.maximum(acc, 0) ** 2) >> 12
                    h1 = np.clip(requant(u, blk["M_actq"].astype(I64),
                                         blk["s_actq"].astype(I64)),
                                 -127, 127).astype(np.int8)
                else:
                    h1 = linear_i8(xm, blk["Wfc1"], blk["bfc1"], blk["Mfc1"],
                                   blk["sfc1"])
                    h1 = lut_i8(h1, blk["ACT_LUT"])
                res[t] = linear_i8_acc16(h1, blk["Wfc2"], blk["bfc2"],
                                         st["Mfc2"], st["sfc2"], res[t])

        # final: v-prediction folded Euler update (M_v[k] is negative).
        # s_shift scales the folded dt by 2^s_shift exactly (strided sampling).
        if mv is None:
            mv, sv = md["M_v"][k], md["s_v"][k]
        sva = (sv - s_shift).astype(np.uint8)
        # return_vacc: return the RAW int32 final-layer accumulator per token
        # (W_final @ xf + b_final) WITHOUT requant, so a guided caller can form
        # v_cond - v_null in int32 before the *w scaling (avoids catastrophic
        # cancellation of two separately-requantized int16 velocities).
        if return_vacc:
            Wf, bf = md["W_final"].astype(np.int64), md["b_final"].astype(np.int64)
            acc = np.empty((T, self.pd), np.int64)
            for t in range(T):
                xf = rmsnorm_i16_to_i8(res[t], md["Gf"][cond][k], md["Bf"][cond][k])
                acc[t] = Wf @ xf.astype(np.int64) + bf
            return acc
        for t in range(T):
            xf = rmsnorm_i16_to_i8(res[t], md["Gf"][cond][k], md["Bf"][cond][k])
            z_tok[t] = linear_i8_acc16(xf, md["W_final"], md["b_final"],
                                       mv, sva, z_tok[t])
        return z_tok

    # ---- decoder -----------------------------------------------------------
    def unpatchify(self, z_tok):
        """tokens i16[T][pd] -> HWC i16[zhw][zhw][zch]; row-major patch grid,
        token feature order (c, py, px)."""
        p, g, C = self.patch, self.g, self.zch
        z = np.empty((self.zhw, self.zhw, C), np.int16)
        for y in range(self.zhw):
            for x in range(self.zhw):
                tok = (y // p) * g + (x // p)
                for c in range(C):
                    z[y, x, c] = z_tok[tok, c * p * p + (y % p) * p + (x % p)]
        return z

    def decode(self, z_tok):
        md = self.md
        z = self.unpatchify(z_tok)
        h = requant_i16_to_i8(z, md["M_zdec"], md["s_zdec"])
        feat = None
        for i, L in enumerate(md["dec"]):
            last = i == len(md["dec"]) - 1
            if last:
                feat = h  # 128^2 x 8 feature map: the hires head's tap
            h = conv3x3_i8(h, L["W"], L["b"], L["M"], L["s"],
                           relu=not last, upsample_in=bool(L["up"]),
                           out_u8=last)
        if "hires" not in md:
            return h  # [H,W,img_ch] uint8
        return self.decode_hires(feat, h)

    def decode_hires(self, feat, img128):
        """ESPCN head (hires models): conv+ReLU then a linear conv whose i8
        output is a delta in pixel LSB units [H,W,12]; depth-to-space 2x2
        over the u8 base: out[2y+i, 2x+j, c] = clip(img128[y,x,c] +
        delta[y,x, c*4 + i*2 + j], 0, 255)."""
        H0, H1 = self.md["hires"]
        h = conv3x3_i8(feat, H0["W"], H0["b"], H0["M"], H0["s"], relu=True)
        d = conv3x3_i8(h, H1["W"], H1["b"], H1["M"], H1["s"], relu=False)
        hh, ww, ch = img128.shape
        base = img128.astype(np.int16)
        out = np.empty((2 * hh, 2 * ww, ch), np.uint8)
        for c in range(ch):
            for i in range(2):
                for j in range(2):
                    dd = d[:, :, c * 4 + i * 2 + j].astype(np.int16)
                    out[i::2, j::2, c] = np.clip(
                        base[:, :, c] + dd, 0, 255).astype(np.uint8)
        return out

    def generate(self, seed, k_steps=None, cond=0, w_idx=-1):
        """seed -> (uint8 image [H,W,img_ch], z trajectory taps for debugging).
        k_steps: power-of-2 divisor of the exported K (stride subsampling of
        the step tables; dt rescale via exact shift). Default: full K.
        cond: table-set index for conditional models (class, or n_classes
        for the null class); must be 0 when n_cond == 1.
        w_idx: index into meta.cfg_w for classifier-free guidance (two DiT
        passes per step, blend folded into per-pass M_v). -1 (or cond = the
        null class, where guidance is a no-op) = plain single-pass."""
        assert 0 <= cond < self.n_cond
        k_steps = k_steps or self.K
        stride = self.K // k_steps
        assert stride * k_steps == self.K and (stride & (stride - 1)) == 0
        s_shift = stride.bit_length() - 1
        n_w = len(self.md["meta"].get("cfg_w", []))
        guided = 0 <= w_idx < n_w and cond != self.n_cond - 1
        st = pcg32_init(seed)
        st, g = gaussian_clt12(st, self.tokens * self.pd)
        z_tok = sat16(g.reshape(self.tokens, self.pd))
        taps = [z_tok.copy()]
        md = self.md
        wq = None
        if guided:
            wq = int(round(self.md["meta"]["cfg_w"][w_idx] * 256))  # w in Q8
        for i in range(k_steps):
            k = i * stride
            if guided:
                # precision-preserving guided blend (matches the engine): form
                # v_cond - v_null in int32 BEFORE the *w scaling, then requant
                # once with the base (unguided) M_v. Avoids catastrophic
                # cancellation of two separately-requantized int16 velocities at
                # high w. wq = w in Q8 (round(w*256)).
                z_in = z_tok.copy()
                acc_c = self.dit_step(z_in, k, s_shift, cond, z_in, return_vacc=True)
                acc_n = self.dit_step(z_in, k, s_shift, self.n_cond - 1, z_in,
                                      return_vacc=True)
                diff = acc_c - acc_n
                acc_g = acc_n + ((wq * diff) >> 8)
                mv, sv = md["M_v"][k], md["s_v"][k]
                sva = (sv - s_shift).astype(np.uint8)
                for t in range(self.tokens):
                    y = requant(acc_g[t], mv.astype(np.int64), sva.astype(np.int64))
                    z_tok[t] = sat16(z_tok[t].astype(np.int64) + y)
            else:
                z_tok = self.dit_step(z_tok, k, s_shift, cond)
            taps.append(z_tok.copy())
        img = self.decode(z_tok)
        return img, taps
