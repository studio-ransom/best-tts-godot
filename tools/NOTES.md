# Model notes (dev tooling)

Everything in `tools/` is **development-only**. The shipped Godot addon is pure
GDScript + GLSL and never runs any of this.

## The model

`addons/best-tts/assets/model_q8f16.onnx` — Kokoro-82M, int8-quantized matmuls
and convs with fp16 weights elsewhere.

| | |
|---|---|
| inputs | `input_ids` int64 `[1,T]`, `style` f32 `[1,256]`, `speed` f32 `[1]` |
| output | `waveform` f32 `[1,N]` @ 24 kHz |
| nodes | 3973 → 28 constant-foldable, 409 CPU shape-domain, 3536 GPU |
| op types | 55 total; 53 need a GPU path, 13 a CPU path |
| initializers | 775 (494 fp16, 230 int8/uint8), 84.5 MB |

Frame math: durations sum to `F` frames; the decoder upsamples ×2, then the
generator ×60, then iSTFT hop 5 → **300 output samples per duration frame**.

Vocab is the 114-symbol map in `hexgrad/Kokoro-82M/config.json` (ids up to 177).
`input_ids = [0, *phoneme_ids, 0]`, and the style row is
`voice_pack[len(phoneme_ids) - 1]`.

Voice `.pt` files are torch zips whose single `<name>/data/0` entry is raw
little-endian float32 `[510, 1, 256]`. No unpickling needed — see
`voice_util.load_voice_pack`.

## onnxruntime quirks (why the tooling looks the way it does)

These cost real debugging time; do not rediscover them.

1. **`ORT_ENABLE_ALL` segfaults** on this model. Only `ORT_DISABLE_ALL` is safe.
2. **Exposing intermediate tensors corrupts the run.** ORT's CPU EP has no fp16
   kernels, so a mandatory `InsertCastTransformer` rewrites fp16 regions to
   fp32. Adding graph outputs pins those tensors to their declared fp16 type and
   defeats the rewrite — results silently go NaN and the waveform changes
   length. `probe.py` therefore exposes only a few tensors at a time and
   **refuses to report anything unless the waveform still matches golden**.
3. Because of (2), per-node ground truth comes from `ref_runtime.py` (numpy),
   not from ORT intermediates.

## Precision policy

The graph casts int32 quantized accumulators directly to fp16, which overflows
if taken literally — ORT never does, because it eliminates those casts. So:

> **fp16 is a storage format only. All compute is fp32, and `Cast(to=float16)`
> is a no-op.**

This is what ORT effectively does, it is what `ref_runtime.py` does by default,
and it is what the GLSL kernels must do. It also makes our output *closer to the
original unquantized Kokoro* than ORT's own, since ORT rounds every node output
back to fp16 and that error compounds through the 12 shared ALBERT layers.

`DynamicQuantizeLSTM` genuinely quantizes its activations to uint8 at runtime.
We dequantize the weights to fp32 instead and skip that — simpler kernel, less
error. `--qlstm` emulates ORT's behaviour and is only for verification.

## Verification status

`op_test.py` extracts each node into a single-node model, feeds it the exact
tensors the numpy runtime produced, and compares against ORT. This isolates ops
from accumulated precision drift.

```bash
python tools/op_test.py --tokens 16 --per-op 3     # 134/134 pass
```

Result: **every one of the 55 op types is verified against onnxruntime**, most
bit-exact. Notably `DynamicQuantizeLSTM` matches to 1.4e-7 once activation
quantization is emulated, confirming gate order (i, o, f, c), the transposed
`W [dir, input, 4H]` / `R [dir, H, 4H]` layout, and per-direction scalar scales.

End-to-end, our fp32 waveform differs from ORT's by accumulated fp16 rounding.
For the 16-token probe the predicted durations track ORT to ~0.02 mean, with one
phoneme landing on the other side of a rounding boundary (64 vs 65 frames, i.e.
12.5 ms). That is expected and inaudible, not a bug.

`compare_speech.py` makes that quantitative for a whole utterance. On a
44-token sentence the Godot runtime sits at spectral 0.986 / envelope 0.994
against the numpy reference, while the numpy reference itself sits at 0.941 /
0.809 against ORT and Godot at 0.926 / 0.803. **Always read an ORT comparison
against that control** (`--control` prints it): the reference passes 134/134
per-op tests, so the distance it shows from ORT is the fp32-policy floor, not
an implementation error.

Do not gate on sample-by-sample correlation. The vocoder integrates F0 through
a `CumSum` to build its excitation, so a tiny frequency difference accumulates
into a growing phase offset — waveforms decorrelate completely while carrying
identical speech. Compare magnitude spectrograms instead.

### Bugs this chain actually caught

- **uint8 weight zero points read as float.** `MatMulInteger` / `ConvInteger`
  take their zero points as buffers, but weight zero points are int8/uint8
  initializers stored packed. Reinterpreting a byte like 114 as float32 gives
  a denormal that rounds to zero, silently dropping the correction — a ~17%
  error in the first `ConvInteger`. Fixed by decoding packed zero points to
  float before binding (`_zp_rid`).
- **`pow()` with a negative base.** GLSL leaves it undefined and returns NaN;
  ONNX allows it. The iSTFT branch squares signed spectra, so this poisoned
  the whole vocoder tail. The tap comparison missed it at first because NaN
  loses every comparison, so `max()` ignored it — the checker now counts
  non-finite values explicitly.
- **Executor not reusable.** Scalar-constant pseudo-tensors were wiped by the
  per-run reset while the cache still handed out their ids, and aliased
  buffers could enter the free pool twice. Both only show up on the second
  `speak()`; `tests/test_repeat.gd` exists to keep them fixed.
- **Nearest Resize off by one at frame boundaries.** `floor(float(l)/scale)`
  landed a hair below the integer on NVIDIA at exactly `l = k*ratio`, so every
  upsampled tensor carried one stale sample per frame — 72 of 108 frames on
  the 9-token case. Now integer arithmetic (`l / (out_dim/dim)`). Effect on
  the waveform turned out to be under 0.25 dB in any band, so this was a
  correctness fix rather than an audible one; the tap test found it because a
  pure gather that is not a gather shows up as a huge `max|err|` on a tensor
  whose input matches.

### Metrics that missed real defects

Three separate times a metric said "fine" while something was wrong. Each gap
is now covered, and the pattern is worth internalizing: **a metric that
aggregates hides exactly the defects that are narrow or systematic.**

| what was wrong | what missed it | what catches it now |
|---|---|---|
| NaN through the vocoder | `max()` — NaN loses every comparison | explicit non-finite count in `test_taps.gd` |
| a tensor uniformly 1.6x hot in a low-energy band | peak-relative `max\|err\|` | relative RMS (`RMS_TOL`) in `test_taps.gd` |
| a 4800 Hz tone 25 dB above the floor | spectral correlation (0.985) and per-band level | `analyze_tone.py`, and a Goertzel gate in `test_speak.gd` |

Also: **regenerate artefacts before measuring them.** A +5 dB high-frequency
tilt "found" in `tests/out/*.wav` turned out to be two stale files from an
earlier session that were never produced by the code under test. Freshly
generated, the same measurement read 0.6 dB. Always re-run
`test_export_speech.gd` before trusting a comparison.

### The ring at 4800 Hz

Kokoro's exported iSTFT overlap-adds with hop 5, and its window does not sum to
a constant, so the output carries a steady tone at `24000/5 = 4800 Hz` and its
octave at 9600 Hz, ~25 dB above the local spectrum. **The numpy reference has
it identically (+24.8 dB), so it is the model's, not the runtime's** — this is
the control that matters, and it is why the addon notches rather than trying to
"fix" the graph. Two 160 Hz-wide notches (1023-tap Blackman-windowed FIR,
`kernels/fir.glsl`) take it to −52 dB for ~1 ms of GPU time and 0.06 dB of
overall level.

A noise gate does not help here and was considered and rejected: the tone rides
along with the speech rather than sitting in the pauses, and the pauses are
already at −118 dB — digital silence, nothing to gate.

### GPU memory

Memory used to climb without bound — 3 GB after five utterances. Three causes,
all fixed; `tests/test_memory.gd` keeps them fixed:

1. The buffer pool was keyed on the **exact** byte size, so a run of a
   different length shared nothing with the run before it. Now bucketed to four
   significant bits (≤6% slack).
2. Anything built with `gpu.create_buffer_f32` during a run was never entered
   into `_live`, so `_release` refused it and it leaked for the life of the
   engine — a few hundred buffers per utterance. Per-run allocations now go
   through `_alloc_f32`.
3. The uniform-set cache was unbounded and, worse, held RIDs that
   RenderingDevice had already freed underneath it when their buffer went away.
   `free_buffer` now purges them and `trim_uniform_sets` caps the rest.

`buffer_update` is illegal while a compute list is open, so `_alloc_f32` closes
the list first. That is not a submission — the recorded dispatches stay queued
— so it costs a barrier, not a stall.

A fourth cause survived all of that and only showed up in a 100-utterance soak:
**33 KB per run, forever.** `_run_dynamic_quantize` allocated an 8-byte min/max
buffer that no tensor id owned, so `_reset_activations` — which walks `_buf` —
never saw it. 139 nodes x one 256-byte bucket, every utterance. The same shape
of bug sat in the LSTM's unused `Y_h`/`Y_c` outputs. Both now go on `_scratch`.

The lesson is the generalizable part: **an allocation that no tensor id owns is
invisible to a reset that walks tensor ids.** `tests/test_leak.gd` prints
buffers, pool, live, const and packed counts per run side by side, which makes
that class of bug a single glance — `buffers` and `live` climbing together
while `pool` stays flat is exactly this.

Watch out for `np.ascontiguousarray`: it promotes 0-d arrays to 1-d, which
silently changes the rank of every scalar constant. It bit us twice (a bogus
`Unsqueeze` failure, and wrong shapes in the packed model). Always
`.reshape(orig_shape)` after calling it.

## The packed model

`pack_model.py` digests the .onnx into `addons/best-tts/assets/model.kpk`, which
is the only model file the addon needs at runtime:

```bash
python tools/pack_model.py          # -> model.kpk, 85.0 MB
python tools/kpk_runtime.py         # proves it is lossless
```

It constant-folds 28 nodes, drops them, keeps 3945 (409 CPU / 3536 GPU), assigns
integer tensor ids, and writes a 534 KB JSON graph plus one flat 84.5 MB weight
blob. Weights stay in their stored dtype (fp16 / int8) and are unpacked on the
GPU. `kpk_runtime.py` reproduces the ONNX-driven waveform with **max|err| = 0**,
so `model_q8f16.onnx` can be deleted from the addon once the GDScript runtime
lands. Keep it out of the Godot export filter's exclusion list — `.kpk` is not a
recognized resource extension, so it must be added explicitly.

## Godot-side gotchas

- **`--headless` gives you no RenderingDevice.** It selects the dummy rendering
  driver, so `create_local_rendering_device()` returns null and no compute can
  run — passing `--rendering-driver vulkan` alongside it does not help. GPU
  tests must run with the real display driver:

  ```bash
  godot --rendering-driver vulkan --script res://tests/test_gpu_elementwise.gd
  ```

  Pure-CPU tests (the loader) are fine under `--headless`.
- **Kernels must be imported before they can load.** `.glsl` files become
  `RDShaderFile` resources with SPIR-V baked in at import time. Run
  `godot --headless --import` after adding or editing a kernel, or
  `ResourceLoader.exists()` reports the kernel missing.
- **Keep large binaries out of the project tree.** `godot --headless --import`
  hung for over ten minutes scanning the 86 MB `.onnx`. It now lives in
  `tools/model_src/` next to a `.gdignore`, which is also where it belongs since
  the addon ships only `model.kpk`.
- GDScript's `%` formatting has **no `%e`**; use `String.num_scientific()`.
- Inner classes cannot be named after built-ins — a `class Node` inside a script
  silently resolves to Godot's `Node` and fails with a baffling
  "Value of type Node cannot be assigned to a variable of type Node".
- Push constants are guaranteed only to 128 bytes. The elementwise kernel packs
  op/count/rank/alpha plus four `uvec4`s into 80 bytes; rank never exceeds 4 in
  this graph and the executor collapses contiguous axes before dispatching.
- **GDScript lambdas capture locals by value.** `var done := false` followed by
  `sig.connect(func(_x): done = true)` writes to a copy and reads back `false`
  forever. This wrote two test results that looked like real product bugs
  before the third one made the pattern obvious. Box it: `var done := [false]`.
- **A submitted Vulkan queue cannot be cancelled.** The executor records ~3945
  dispatches in a few hundred milliseconds and the GPU then runs them for over
  a second, so a cancel arriving during that window has nothing to interrupt.
  Bounding the wait means submitting in segments — `cancel_flush_nodes`, 512 by
  default, which trades 2.3% throughput for a 3.3x better worst case.
- **Never iterate a Dictionary another thread can write.** `total_buffer_bytes`
  walked `_buffers` from whatever thread asked for a memory reading while the
  engine thread was still allocating and freeing, and threw on the first key
  that vanished mid-loop — intermittently, and only under load, which is the
  worst way to find out. Both that and `pooled_bytes` are maintained counters
  now: a few kilobytes stale at worst, and reading an int cannot fault. Anything
  a debug overlay might poll needs the same treatment.
- **GDScript cannot override a native method.** `func play()` on an
  AudioStreamPlayer subclass parses, and is then never called — not by the
  engine, and not by a script calling `player.play()` either. Godot raises a
  `NATIVE_METHOD_OVERRIDE` warning, but `debug/gdscript/warnings/exclude_addons`
  is on by default, so a file under `res://addons/` gets no warning at all and
  the method simply does nothing. That is why the voice players expose
  `speak()` and `prepare()` instead of overriding `play()`, and why
  `prepare_on_ready` exists: it puts real audio in `stream` so the native
  `play()` is the right thing to call.
- **`super()` is unavailable for engine virtuals.** `_instantiate_playback()`
  on an `AudioStreamGenerator` subclass cannot call the parent implementation —
  "Cannot call the parent class' virtual function because it hasn't been
  defined". Combined with `AudioStreamPlayback._mix` taking a raw buffer pointer
  (`_mix(buffer: int, ...)`) that GDScript cannot write through, a custom
  AudioStream that synthesizes during playback is impossible without
  GDExtension. Both were checked directly against 4.7 before the node API was
  designed around them.
- **`Engine.get_main_loop()` is null inside `SceneTree._init()`.** The main loop
  is not registered until the tree has finished constructing, so anything that
  reaches for the tree from there — like a static singleton parenting itself to
  the root — silently gets nothing. Tests that use `BestTTS` need one
  `await process_frame` first, which is also what a real game has.
- **A `Thread` must be joined before its owner is freed**, and a
  `RenderingDevice` may only be freed by the thread that created it. Together
  those make `_exit_tree` a blocking join. The only lever is making the thread
  finish sooner, which is why `shutdown()` cancels before it joins: 3.8 s down
  to 0.46 s busy, 0.13 s idle.

## Tools

| script | purpose |
|---|---|
| `analyze.py` | classify nodes CONST / CPU / GPU; op inventory |
| `voice_util.py` | vocab, tokenization, `.pt` voice loading (mirrors GDScript) |
| `golden.py` | snapshot ORT's waveform as the end-to-end reference |
| `ref_runtime.py` | numpy interpreter — the oracle and the executable spec |
| `op_test.py` | per-node ORT comparison; `--export` dumps `.npz` for GLSL tests |
| `probe.py` | read a few ORT intermediates safely |
| `bisect_ref.py` | locate the first tap where two runtimes diverge |
| `dump_intermediates.py` | bulk shape/dtype manifest (values unreliable, see quirk 2) |
| `pack_model.py` | .onnx → `model.kpk`, the shipped artifact |
| `kpk_runtime.py` | run the packed model in numpy; proves the pack is lossless |
| `export_test_case.py` | ids + style + reference waveform for `test_executor.gd` |
| `export_taps.py` | per-tensor reference values for `test_taps.gd` |
| `compare_speech.py` | whole-utterance Godot vs ORT/numpy: durations, spectra, tilt, .wav |
| `analyze_tone.py` | sustained tonal artefacts in a .wav; finds what correlation cannot |

## Loudness across the voice packs

The 50 packs are not mastered to a common level — **12.3 dB** between the
quietest (`af_nova`, -27.6 LUFS) and the loudest (`if_sara`, -15.4). A game that
switches speaker mid-scene hears it as a volume step.

`tests/test_calibrate_voices.gd` measures each voice and writes
`assets/voice_levels.json`; the runtime applies one constant gain per voice
through the `gain` push constant `pcm16.glsl` already had, so it costs nothing.
Across twelve voices on a sentence the calibration never saw, the spread closes
from 12.2 dB to 2.1 dB.

Three things worth keeping in mind if this is ever revisited:

- **Gate the measurement.** Ungated RMS over a whole clip is partly a measure of
  how much of it was silence, so a voice that pauses between clauses reads
  quieter than an equally loud voice that does not — and the gain built from it
  would be wrong in proportion to the speaking rate. BS.1770's absolute -70 LUFS
  and relative -10 LU gates are the fix, and they matter far more here than the
  K-weighting does.
- **The published K-weighting coefficients are for 48 kHz.** At 24 kHz they are
  simply the wrong filter; the biquads have to be designed for the actual rate.
- **One gain per voice, not per clip.** Per-clip normalization would flatten the
  variation *within* a voice, which is the model responding to content and worth
  keeping. That variation — 0.5 to 4.4 dB depending on the voice, measured across
  three sentence lengths — is also exactly what the residual 2.1 dB is. Chasing
  it further would be fixing the wrong thing.

The gain is capped by each voice's measured peak (`PEAK_CEILING`, 0.99) because
`pcm16.glsl` hard-clamps: an overshooting gain does not just get loud, it turns
a peak into a click. One voice of 50 is peak-limited, under-corrected by 0.9 dB.
