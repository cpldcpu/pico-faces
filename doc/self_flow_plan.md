# Self-Flow for pico-faces — assessment and implementation plan

Paper: *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*
(Chefer, Esser et al., Black Forest Labs, arXiv 2603.06507, Mar 2026 —
local copy: `doc/2603.06507v1.pdf`).

Status: **Stages 0+1 implemented and RUN on this branch — gate NOT met,
Stage 2 not launched.** The per-token conditioning lives in
`train/dit/model.py` / `train/common/sincos.py`, the training paths in
`train/dit/train_dit.py` (yaml-gated, default path byte-identical), and the
twin bake-off configs are `models/sf0_base` (control), `models/sf0_dualt`
(Dual-Timestep only), `models/sf1_selfflow` (full Self-Flow) — run via
`scripts/sf_bakeoff.sh`. Results below (section 0); the rest of the
document is the original assessment and plan.

## 0. Bake-off results (2026-07-24, 80k twins, depth-8, seed 0)

| variant | final val v-loss | vs control | fixed-seed grids |
|---|---|---|---|
| `sf0_base` | **1.17762** | — | reference |
| `sf0_dualt` | 1.17824 | +0.0006 (tie) | wash — no systematic difference |
| `sf1_selfflow` | 1.18332 | +0.0057 (behind) | K=64 near-twins; K=4 slightly NOISIER on several cells |

Sanity check that made the twins meaningful: `sf0_base` reproduced the
dev repo's historical 80k bake-off val loss to all five decimals
(1.17762) — the default path really is untouched.

Observations:

- **Dual-Timestep alone is cost-free but not beneficial here.** It lagged
  the control by ~0.002 through mid-training (learning the heterogeneous
  task), converged to parity by 60k, and its grids are indistinguishable.
  The paper's Fig. 2b "slight improvement" did not materialize at 1.6M
  params / 70k aligned faces.
- **Full Self-Flow slightly REGRESSES at equal budget.** The −0.006 val
  gap held stable to the end (unlike dual-t's, it never closed), and the
  K=4 grids show mildly more texture damage on the harder cells.
- **The alignment task saturates instantly** — student↔teacher cosine hit
  0.96 by step 1.5k and stayed flat for the remaining 78k steps. At
  dim 128 / 64 tokens the projection head aligns shallow-to-deep features
  trivially, so the information asymmetry never forces new representation
  structure. In the paper, alignment keeps strengthening for a long time —
  that dynamic is what pays, and it is absent at this scale. This is the
  scale-gap risk from section 4 materializing, with a mechanism attached.

**Aggressive probe (`sf1_hard`, 2026-07-24, stopped at 15k by design):**
every knob turned toward making the mechanism fight — mask 0.5, student
layer 1 → teacher 6 (the largest asymmetry depth 8 allows), γ 1.5. The
alignment cosine followed the default run's trajectory point-for-point
(0.56 @ 500, 0.92 @ 1k, 0.96 @ 1.5k, flat after) and the val curve matched
the default Self-Flow run to ~4 decimals (1.2757 vs 1.2757 @ 15k). The
task is trivially solvable at dim 128 no matter how it is posed; the run
was stopped early as conclusive.

**Verdict: the ~400× scale-down kills the effect.** Section 10's negative
blog chapter is the outcome: a clean data point on where representation
alignment stops paying. `models/sf2_deep` (the full depth-12 Self-Flow
config) is committed but was NOT launched. If anyone revisits: the knobs
most likely to matter are a harder alignment task (mask 0.5, shallower
student layer), lower γ (0.4), and the shifted p(t) — but the instant
cosine saturation suggests the mechanism itself has no room to work at
this width. Evidence: `artifacts/sf_compare_strip.png`, the three
`artifacts/sf*/runs/dit/grid_080000_k{4,64}.png`, train logs alongside.

---

## 1. TL;DR

Self-Flow is a **training-only** recipe that improves flow-matching models by
adding a self-supervised representation objective — no external encoder, no
architecture change at inference. It targets exactly the failure class we have
fought hardest on this project (global facial structure: symmetry, the class-3
smile collapse) and costs **zero bytes of flash and zero device cycles**. A
better fp checkpoint propagates straight through the existing distill-QAT to
int8.

The honest caveat: the paper's smallest model is 290M params trained on
1.28M–200M images; ours is 1.6–2.4M params on 70k single-domain faces — a
~400× scale-down nobody has tested. The effect may shrink to nothing. But the
first stage of the method (Dual-Timestep Scheduling alone) is nearly free to
implement and the paper shows it helps *by itself*, so this is worth a cheap
staged experiment with twin-run discipline. Either outcome is a blog chapter.

## 2. The paper in one page

Two stacked ideas, both applied only during training:

**Dual-Timestep Scheduling.** Instead of noising all tokens at one timestep
`t`, sample two timesteps `t, s ~ p(t)` and a random token mask `M`
(mask ratio `R_M`); tokens in `M` are noised at `s`, the rest at `t`. The
flow loss is unchanged (`v = eps − x0` per token regardless of its τ). The
cleaner tokens create an information asymmetry: the model can — and learns
to — use them to infer the noisier tokens, which forces *global* relations
instead of local denoising shortcuts. Crucially the per-token marginal
t-distribution is preserved, so unlike diffusion-forcing / full-masking there
is no train–inference gap. This alone improves FID (their Fig. 2b).

**Self-Flow (the representation loss).** Keep an EMA teacher. The teacher
sees the input noised *everywhere* at `τ_min = min(t, s)` (the cleaner
level); the student sees the mixed-noised input. The student's layer-ℓ
features, passed through a small train-only MLP projection head, must match
the teacher's layer-k features (k > ℓ) under cosine similarity, with the
teacher stop-gradded. Total loss `L = L_gen + γ · L_rep`.

Their settings (appendix A.3): γ = 0.8, image mask ratio `R_M = 0.25`,
student layer at 0.3·depth aligned to teacher layer at 0.7·depth, EMA decay
0.9999 (the EMA copy doubles as teacher and eval model), lightweight
projection head (~1.6% of model params).

Results and ablation facts worth carrying over:

- Beats REPA (external DINOv2 alignment) on ImageNet FID (5.70 vs 5.89) and
  converges ~2.8× faster; stronger external encoders paradoxically *hurt*.
- Removing `L_rep` costs the most (~4 FID); removing the masking while
  keeping `L_rep` still costs >1 FID — both components matter.
- Constraining `s` to be only slightly cleaner than `t` (s ∈ [t, t−0.2])
  is nearly as bad as no masking: **sample both timesteps from the full
  distribution.**
- ℓ1 instead of cosine for `L_rep` goes numerically unstable late in
  training (feature norms grow) — **use cosine.**
- Dual-timestep noising shifts the average SNR toward the middle, so their
  method prefers a slightly *higher* timestep shift than vanilla FM — the
  `p(t)` may need a nudge.
- Qualitative claim directly relevant to us: gains concentrate in
  "structural coherence, particularly for challenging structures like faces
  and hands."

## 3. Why it maps onto pico-faces

- **It targets our weakest axis.** Every hard defect this project has hit on
  the DiT side has been a *global-coherence* failure: the class-3 smile
  collapse, plain-mode structural damage, asymmetric faces at few steps.
  With 64 tokens each covering 16×16 output pixels of an aligned face,
  "reconstruct the masked eye region from the visible eye" is a meaningful
  and well-posed self-supervised task.
- **Zero device cost.** The teacher, the projection head, and per-token
  timesteps exist only in training. At inference all tokens share one `t`,
  so the adaLN folding, the per-step tables, the v8 blob, and the C engine
  are all untouched — the exported semantics are byte-identical to today.
- **Gains propagate to int8 automatically.** Our shipping QAT is
  self-distillation against the fp teacher (`quant/qat.py --distill
  --traj-pool`); the int8 model faithfully tracks whatever fp
  checkpoint we hand it. Improve fp → improve device, no new quant work.
- **It is the only representation-alignment method applicable here.**
  REPA-style external encoders are a non-starter for 128×128 grayscale
  faces in a custom 16×16×8 latent — and the paper's own scaling study shows
  external alignment degrades as encoders get stronger. Self-Flow is
  self-contained.
- **Cheap to falsify.** Our bake-off runs are ~1–2 h on the 5090; the extra
  teacher pass is no-grad (~1.3–1.4× step cost); we already maintain an EMA.

## 4. Risks

- **Scale gap (the big one).** 290M–1B params / 200M images in the paper vs
  1.6–2.4M params / 70k images here. At our size the auxiliary loss competes
  for scarce capacity, and aligned FFHQ already bakes much of "global
  structure" into the positional embedding. The effect could be zero or
  negative. This is precisely what Stage 0/1 twin runs answer cheaply.
- **Part of the paper's headline is convergence speed.** We already train to
  saturation with a cosine cooldown, so only the *final-quality* component
  matters for us. Expect the realized gain to be smaller than headline
  numbers.
- **Measurement noise.** Plausible gains (~1–2 FID fp) are near run-to-run
  variance. Mitigation: identical-seed twin runs, fixed `grid_noise`, and
  the standing lesson that eyeballs on plain-mode grids catch what FID
  misses (the smile).
- **`p(t)` interaction.** Dual-timestep shifts SNR coverage; our
  logit-normal(0,1) may want a small positive mean shift. One extra knob to
  ablate, not a blocker.

## 5. What changes / what does not

| Area | Change |
|---|---|
| `train/dit/model.py` | Per-token conditioning path (τ vector), guarded so the scalar-`t` path is byte-identical |
| `train/dit/train_dit.py` | Dual-timestep noising; Stage 1 adds teacher forward, projection head, cosine rep loss; class dropout moves into the loop |
| `train/common/sincos.py` | `timestep_embedding` accepts `(B, N)` |
| `models/*/dit.yaml` | New optional keys (`dual_timestep`, `self_flow: {...}`) |
| **Unchanged** | `engine/` (all of it), `quant/fold.py`, `quant/export.py`, blob format v8, `quant/calibrate.py`, `quant/qat.py` recipe (teacher just re-pointed at the new fp ckpt), `train/dit/sample.py`, checkpoint key layout, device inference |

## 6. Design constraints (the sharp edges)

1. **Checkpoint compatibility is sacred.** `fold.py`, `qat.py`, and
   `sample.py` all do strict `load_state_dict` on `ckpt["model"]` /
   `ckpt["ema"]`. Therefore the projection head lives in the **trainer**,
   not inside `DiT` — saved under its own ckpt key (e.g. `"sf_head"`), never
   polluting the model state dict. `device_params()` needs no change.
2. **Scalar-`t` path stays byte-identical.** `DiT.forward` guards on
   `t.dim()`: `(B,)` → today's path exactly (`c` is `(B, dim)`, broadcast
   `[:, None]` as in [model.py:73](train/dit/model.py#L73)); `(B, N)` → the
   new per-token path (`c` is `(B, N, dim)`, `self.mod(c)` needs no
   broadcast). Sampling, calibration, folding, and reflow-style tooling all
   pass scalar `t` and see zero behavioral change.
3. **Token ↔ latent granularity.** τ is per *token* (64 = 8×8 patch grid),
   but noising happens on the 16×16×8 latent. Expand the 8×8 τ map by 2×
   nearest-neighbor to 16×16 and broadcast over channels — each 2×2 latent
   patch gets its token's timestep, matching the patchify contract
   ([model.py:110](train/dit/model.py#L110)).
4. **The velocity target does not change.** `v = eps − x0` per token, whatever
   its τ ([train_dit.py:173](train/dit/train_dit.py#L173) stays as-is).
5. **Teacher = live EMA module.** Our `EMA` class is a state-dict shadow
   ([ema.py](train/common/ema.py)); calling `copy_to()` every step is
   wasteful. Maintain the teacher as a real module whose params are lerped
   in place each step (mathematically identical to the shadow update); the
   existing `EMA` object can stay for checkpointing, or the teacher replaces
   it as the single EMA of record.
6. **Class dropout moves from `model.forward` to the loop** so teacher and
   student see *identical* labels (currently it is inside the model,
   [model.py:129](train/dit/model.py#L129); the qat.py `--distill` path
   already established this pattern).
7. **Rep-loss layers at our depths.** 0.3·D → 0.7·D maps to: depth 8 —
   student after block 2, teacher after block 6; depth 12 — student after
   block 4, teacher after block 8. Grab features via forward hooks or an
   optional `return_feats` list on `DiT.forward` (train-only argument,
   default off).

## 7. Staged implementation plan

### Stage 0 — Dual-Timestep Scheduling alone (~½ day + ~2–3 h GPU)

No teacher, no head, no rep loss — the paper shows this component helps by
itself, and it derisks the per-token conditioning plumbing.

1. `sincos.timestep_embedding`: accept `(B,)` or `(B, N)` (flatten/reshape).
2. `DiT.forward` / `DiTBlock.forward` / final mod: per-token `c` path per
   constraint #2. `y_emb` broadcasts over tokens.
3. `train_dit.py` noising block ([train_dit.py:164-168](train/dit/train_dit.py#L164-L168)):
   sample `s` like `t`, draw per-token mask at `R_M = 0.25`, build τ `(B, 64)`,
   expand to the latent grid, noise per-pixel; forward with τ.
4. Config key `dual_timestep: true` (+ `mask_ratio`), default off; the
   default path must reproduce today's training byte-for-byte.
5. **Bake-off:** twin 80k runs at the m3_long arch (depth 8), same seed,
   vanilla vs dual-t. Cheap, fast signal.

Gate → Stage 1: fixed-seed grids at K=64 and K=8 visibly no worse, and
val v-loss / FID(N=5000 fp protocol) not regressed. (A *neutral* Stage 0 is
acceptable — its job in the paper is mostly to enable the rep loss.)

### Stage 1 — full Self-Flow (~1 day + ~1–2 h GPU per variant)

1. Live EMA teacher (constraint #5); teacher input = all tokens at
   `τ_min = min(t, s)`, forward no-grad.
2. Projection head in the trainer: 2-layer MLP `dim → dim → dim` (~33K
   params at dim 128), applied to student features at 0.3·D.
3. `L_rep` = negative cosine per token between projected student features
   and stop-grad teacher features at 0.7·D; `L = L_gen + γ·L_rep`, γ = 0.8.
4. Class dropout into the loop (constraint #6).
5. Config block:
   ```yaml
   self_flow:
     gamma: 0.8
     mask_ratio: 0.25
     l_student: 2      # 0.3 * depth (depth-8 bake-off)
     l_teacher: 6      # 0.7 * depth
   ```
6. **Ablation matrix** (each an 80k depth-8 twin): γ ∈ {0.4, 0.8},
   `R_M` ∈ {0.1, 0.25, 0.5}, and one run with the logit-normal mean shifted
   +0.3 (the SNR-shift note from their App. B). Run the promising corner,
   not the full cross product — ~4–5 runs.

Gate → Stage 2: FID improvement beyond twin-seed noise **and** no plain-mode
structural regression by eyeball. If FID is neutral but plain-mode structure
(symmetry, smiles) is visibly better, that still passes — the project's
standing lesson is that FID at w=4 has blind spots exactly there.

### Stage 2 — full retrain + downstream ladder (overnight GPU + ~½ day pipeline)

Only if Stage 1 wins:

1. Retrain the flagship arch (m3_deep: depth 12, 500k + 100k cosine
   cooldown per [models/m3_deep/dit.yaml](models/m3_deep/dit.yaml)) with the
   winning Self-Flow recipe (`l_student: 4`, `l_teacher: 8`). Step cost
   ~1.3–1.4× → plan for an overnight run.
2. Downstream is the existing scripted ladder, unchanged in kind:
   regenerate union calib → distill-QAT 45k with `--teacher-ckpt` pinned to
   the **new** base fp checkpoint (`--distill --cosine --traj-pool 512`) →
   frozen-calib v8 fold → FID + K×W grids (N=5000 device protocol: seeds
   1e6.., K=8 w=4, cls = seed % 4) → UF2 rebuild and golden update.
3. Ship gate: beats the current flagship (int8 FID 53.8, fp 52.44) or shows
   clearly better plain-mode structure at equal FID.

## 8. Hyperparameter mapping (paper → ours)

| Knob | Paper | Ours (start) | Note |
|---|---|---|---|
| γ (rep-loss weight) | 0.8 | 0.8 | ablate 0.4 |
| Mask ratio `R_M` | 0.25 (image) | 0.25 | ablate 0.1 / 0.5 |
| Student / teacher layer | 0.3·D / 0.7·D | 2/6 (d8), 4/8 (d12) | |
| Rep metric | cosine | cosine | ℓ1 is unstable — don't |
| Projection head | ~1.6% of params | MLP 128→128→128 | trainer-owned, never exported |
| EMA decay | 0.9999 | keep 0.9995 | our runs are 10× shorter; slower teacher would lag |
| `p(t)` | task-tuned shift | logit-normal(0,1) | one ablation with mean +0.3 |
| Timestep sampling | both from full `p(t)` | same | never constrain `s` near `t` |

## 9. Open questions

- Does the projection head matter at dim 128, or can the student features
  align raw? (Paper always uses a head; keep it — it's 33K train-only params.)
- Is 64 tokens enough masking granularity? MAE-style methods usually run
  196+; our fallback is `R_M = 0.5` (paper's audio setting, 250 tokens) if
  0.25 leaves too weak a signal.
- Should the cooldown finetune also run under Self-Flow, or vanilla?
  Default: keep the recipe identical for the whole run (the cooldown is
  `cooldown_frac` of the same run, not a separate stage).
- Interaction with CFG training is expected to be nil (class dropout is the
  only mechanism, and it is shared teacher/student) — verify on the w-sweep
  grids in Stage 1 anyway.

## 10. Blog angle

"Does a 2026 BFL representation-learning trick survive a 400× scale-down?"
is a self-contained chapter either way: positive → the smallest Self-Flow
model ever trained, running on a $5 microcontroller; negative → a clean
data point on where representation alignment stops paying, with twin-run
evidence. Figures fall out of the existing tooling (fixed-seed grids, FID
ladder, K×W sheets).
