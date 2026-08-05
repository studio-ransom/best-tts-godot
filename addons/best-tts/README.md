# best-tts — Kokoro-82M TTS for Godot

Neural text-to-speech running entirely inside Godot. **No GDExtension, no C++,
no native binaries, no onnxruntime** — the model executes in pure GDScript
driving hand-written GLSL compute shaders.

> **Status: working.** Text goes in, speech comes out, verified against
> onnxruntime. A 3.3 s sentence synthesizes in ~600 ms on an RTX 4070 Ti
> (5–9x realtime). What is left is polish and wider G2P coverage — see the
> roadmap.

## Try it now

`demo/demo.tscn` is the project's main scene: four panels walking through the
API in the order you would meet it, with the code for each panel on screen next
to the button that runs it. Press **F5** in the editor, or:

```bash
godot --rendering-driver vulkan
```

`demo/test_bench.tscn` is the other scene — a capability bench covering the
packed model, all 50 voice packs, the tokenizer and chunker, GPU kernel
correctness against a CPU reference, a dispatch benchmark, the
GPU → PCM16 → `AudioStreamWAV` output stage and text → IPA conversion. Add
`--autorun` to run everything and exit with a non-zero code on failure:

```bash
godot --rendering-driver vulkan --autorun
```

## Requirements

- Godot **4.4+** (developed against 4.7)
- Renderer set to **Forward+** or **Mobile**. The Compatibility/GL backend has
  no `RenderingDevice` and therefore cannot run compute shaders.
- A GPU with Vulkan support.

## The short way

Three layers. Most games never leave the first one.

### Nodes

`BestVoicePlayer`, `BestVoicePlayer2D` and `BestVoicePlayer3D` are
AudioStreamPlayers that speak. `text`, `voice` and `speed` are inspector fields;
the voice is a dropdown of the 50 packs.

```gdscript
$Merchant/Voice.speak("I have wares, if you have coin.")
```

With `prepare_on_ready` on (the default), the line is synthesized while the
scene loads and lands in `stream` as ordinary audio. After that the node is an
ordinary player: `play()`, `seek()`, `stop()` and `finished` all behave exactly
as they always do.

**`speak()` is the verb, not `play()`.** GDScript cannot override a native
method — Godot parses the override and then never calls it, not even from a
script calling `player.play()` — so an unprepared line has nothing to play.
`speak()` synthesizes if it has to; `play()` is for audio that is already there.

| | |
|---|---|
| `speak(line := "")` | say it now, streaming from the first clause |
| `prepare()` | fill `stream` without playing — awaitable |
| `stop_speaking()` | cancel what is queued and stop what is playing |
| `cancel()` | drop an undelivered line, leave playback alone |
| `is_speaking()` | true from queued until the audio ends |
| `prepared` | signal: `stream` now holds audio |

### A line as a resource

`BestVoiceStream` extends `AudioStreamWAV`, so once rendered any stock player
plays it. Text and voice are saved with the resource, which makes dialogue
something a writer can edit as a `.tres`.

```gdscript
var line := BestVoiceStream.of("The bridge is out ahead.", "bm_george")
await line.render()
$AnyOldPlayer.stream = line
$AnyOldPlayer.play()
```

Editing `text`, `voice` or `speed` discards the audio, so a stale render can
never be played by mistake.

### The shared engine

`BestTTS` is a static class — no node, no autoload. The first call creates the
one engine the process needs and parents it to the tree root.

```gdscript
BestTTS.warm_up()                                   # on the menu screen
BestTTS.speak_into($Guard/Voice, "Halt!", "bm_george")
BestTTS.stream_into($Narrator, chapter_text)        # starts on clause one
BestTTS.precache(BARK_LINES, "am_michael")
BestTTS.cancel_all()
```

`BestTTS.configure(fn)` applies settings before the engine exists, which is the
only way to set the cache options in time:

```gdscript
BestTTS.configure(func(tts):
    tts.cache_on_disk = true
    tts.default_voice = "bm_george")
```

`BestTTS.shutdown()` releases the ~190 MB; the next call builds a fresh engine.

## The engine node

`KokoroTTS` is what the two layers above are built on. Add one yourself for
per-scene settings, a separate cache, or control over when the model loads.

```gdscript
@onready var tts: KokoroTTS = $KokoroTTS

func _ready() -> void:
    var stream := await tts.speak("Hello from Godot.", "af_heart")
    $AudioStreamPlayer.stream = stream
    $AudioStreamPlayer.play()
```

`await` is the shape to reach for in a cutscene. Everywhere else — barks,
reactions, anything a player can interrupt — use `say()`, which hands back a
`Request` immediately:

```gdscript
var r := tts.say("Did you hear that?", "am_michael")
r.finished.connect(func(clip): $Voice.stream = clip; $Voice.play())
...
r.cancel()   # player walked away
```

| method | returns |
|---|---|
| `speak(text, voice, speed)` | `AudioStreamWAV`, 24 kHz mono — await it |
| `speak_phonemes(ipa, voice, speed)` | bypass G2P, feed IPA directly |
| `say(text, voice, speed, priority)` | a `Request`, right now — no await |
| `say_phonemes(ipa, voice, speed, priority)` | same, for IPA |
| `speak_into(player, text, …)` | plays into your own 2D/3D player |
| `stream_into(player, text, …)` | same, starting on the first clause |
| `speak_streaming(text, voice, speed)` | player that starts on the first chunk |
| `precache(lines, voice)` | warm a set of barks at low priority |
| `cancel_all()` | drop the queue, abandon the GPU run |
| `is_busy()` / `queue_size()` | what the engine is doing |
| `has_g2p(voice)` | can `speak()` drive this voice from text? |
| `phonemize(text, voice)` | `{phonemes, oov, lang, words}` — no synthesis |
| `list_voices()` / `describe_voice(v)` | the 50 bundled voice names |
| `has_voice(name)` | is this a real pack? case-sensitive on purpose |
| `cache_stats()` / `clear_cache(also_disk)` | cache introspection |
| `KokoroAudio.measure_loudness(clip)` | `{lufs, peak}`, gated BS.1770 |
| `gpu_memory_bytes()` | what the device is holding — cheap enough to poll |
| `release_memory()` | hand idle GPU memory back early |
| `shutdown()` | stop the engine and free the device |

A `Request` carries `stream`, `error`, `timings`, `oov`, `phonemes` and
`from_cache`, has `cancel()` / `is_done()` / `is_ok()`, and emits
`finished(clip)` plus `chunk_ready(clip, index, total)`. **`finished` always
fires exactly once** — on success, failure, cancellation and shutdown — so an
`await` can never be stranded.

Node signals: `engine_ready`, `engine_failed(reason)`,
`synthesis_progress(chunk, total)`, `went_idle`.

`went_idle` fires once per busy-to-idle transition — after the queue drains and
idle GPU memory has gone back to the driver. It fires even when every request
was served from cache and the worker never woke, so `precache(...); await
went_idle` is safe on a second playthrough when there is nothing left to
synthesize.

The model, the GPU device and the text-to-phoneme frontend all live on a worker
thread that owns its own `RenderingDevice`. Calling `speak()` before loading
finishes just waits.

### In a game

```gdscript
# An NPC line, positioned, interruptible, cached after the first time.
var line: KokoroTTS.Request

func greet() -> void:
    if line: line.cancel()
    line = tts.speak_into($Voice, "I have wares if you have coin.", "bm_george")

func _on_player_left() -> void:
    if line: line.cancel()
```

```gdscript
# Something urgent, jumping whatever ambient chatter is queued.
tts.say("Look out!", "am_michael", 0.0, KokoroTTS.Priority.HIGH)
```

```gdscript
# A long piece of narration: starts on the first clause, not the last.
var r := tts.stream_into($Narrator, chapter_text)
await r.finished
```

```gdscript
# Subtitles that follow the voice.
tts.compute_timings = true
var r := tts.say(line_text)
await r.finished
$Voice.stream = r.stream
$Voice.play()
for w in r.timings.words:
    await get_tree().create_timer(w.start - $Voice.get_playback_position()).timeout
    $Subtitle.highlight(w.text)
```

```gdscript
# Loading screen: warm the barks, then they cost nothing all session.
tts.cache_on_disk = true
tts.precache(BARK_LINES, "am_michael")
await tts.went_idle
```

### What it costs the calling thread

Everything below was measured by `tests/test_stress.gd` on an RTX 4070 Ti.

| on the main thread | |
|---|---|
| `say()` — one bark | 0.07 ms |
| `say()` — a full page of text | 0.06 ms |
| `say()` — a cache hit, clip returned | 0.15 ms |
| freeing the node while idle | 126 ms |
| freeing the node mid-utterance | 463 ms |

G2P used to run on the caller, which cost 9.5 ms for a page and a one-off 90 ms
the first time a lexicon was parsed. Both now happen on the worker.

Teardown has to abandon the GPU submission in flight and then free the device,
and a `RenderingDevice` may only be freed by the thread that made it — so it is
not free, but it is no longer the 3.8 s it was when it waited for the whole
utterance.

### Cancelling

`cancel_all()`, or `Request.cancel()` for one line. Queued requests are dropped
instantly; a run already on the GPU is abandoned at the next submission
boundary, because Vulkan has no way to un-queue work that has been submitted.

That boundary is `gpu_submit_nodes`, and it is the throughput/latency dial.
On a 25 s passage:

| `gpu_submit_nodes` | full run | worst-case cancel |
|---|---|---|
| 0 (one submission) | 3650 ms | 1149 ms |
| 1024 | 3650 ms | 532 ms |
| **512 (default)** | **3734 ms** | **348 ms** |
| 256 | 3817 ms | 248 ms |

Below 256 the sync bubbles cost more than they save.

### Caching

Games repeat themselves, and synthesizing "I have wares if you have coin" for
the fortieth time costs 400 ms and half a gigabyte of VRAM traffic for a
byte-identical result. On by default:

| | |
|---|---|
| cold synthesis | 596 ms |
| memory hit | 0.16 ms — same frame, on the calling thread |
| disk hit | 17 ms, on the worker |

Keyed on text, voice, speed, `remove_vocoder_tone` and the normalization gain
— everything that changes the samples, so nothing stale is ever served. Bounded by `cache_memory_mb` (32 MB) with LRU eviction. Set
`cache_on_disk` to keep clips under `cache_dir` as well, and a game's fixed
dialogue is only ever synthesized on the player's first playthrough.

`precache(lines)` queues at LOW priority, so warming a hundred barks during a
loading screen cannot delay the line the player is waiting on.

### Loudness

Kokoro's voice packs are not mastered to a common level: across the 50 bundled
voices the gated loudness spans **12.3 dB**, from `af_nova` at −27.6 LUFS to
`if_sara` at −15.4. Switching speaker mid-scene steps the volume audibly, and
before this there was nothing to do about it but ride the gain by hand.

`normalize_loudness` (on by default) applies one constant gain per voice, from
the table in `assets/voice_levels.json`. On a sentence the calibration never
saw, across twelve voices picked from both ends of the range:

| | spread |
|---|---|
| raw | 12.2 dB |
| normalized | **2.1 dB** |

Constant per voice, deliberately — the variation *within* a voice is the model
responding to content and worth keeping, so a short exclamation still lands
louder than a long calm line. That variation is also what the residual 2.1 dB
is: one gain cannot track it, and flattening it would be the wrong fix.

Nothing can clip. The gain is capped by each voice's measured peak, so it is
applied by the existing multiply in `pcm16.glsl` at no runtime cost, and the
loudest sample measured after normalizing was 0.57 of full scale. Exactly one
of the 50 voices is peak-limited, under-corrected by 0.9 dB.

`loudness_target_lufs` overrides the target; 0 means the table's own, which is
the median voice — chosen so corrections stay small in both directions.

Regenerate the table after changing the vocoder, the notch filter or the packs:

```bash
godot --rendering-driver vulkan --script res://tests/test_calibrate_voices.gd
```

Measurement is ITU-R BS.1770 — K-weighted, gated at −70 LUFS absolute and
−10 LU relative. The gating is the part that matters: ungated RMS over a whole
clip measures how much of it was silence, so a voice that pauses between
clauses reads quieter than an equally loud voice that does not, and a gain
built on that would be wrong in proportion to the speaking rate.

### Queue and priority

One engine, one GPU, one line at a time. Requests are served in priority order
(`HIGH` / `NORMAL` / `LOW`), newest last within a priority, and the backlog is
capped at `max_queued` (32). Past that the lowest-priority waiting requests are
dropped newest-first and fail with a reason — a queue is a latency budget, and
audio that arrives a minute late is worse than audio that never arrives.

### Timings for subtitles and lip sync

Set `compute_timings` and every `Request` comes back with:

```gdscript
{
  "words":    [{"text": "bridge", "start": 0.51, "end": 0.83}, ...],
  "phonemes": [{"symbol": "b",    "start": 0.51, "end": 0.55}, ...],
  "duration": 3.40,
}
```

These are the duration predictor's own numbers — the same ones the model uses
to stretch the alignment — so they describe the audio that was actually
produced rather than an estimate over it. On *"The old bridge collapsed under
the weight of the caravan."*: 10 words for 10 words, first at 0.33 s, last
ending at 3.38 s of a 3.40 s clip, and the span the timings call speech
measures 32 dB above the lead-in silence. They survive chunking (each chunk is
offset by the audio before it) and are stored with the clip, so a cache hit
brings its timings along.

Cost is one extra readback of a few dozen floats per chunk.

### Memory

The engine holds **~190 MB** on the GPU at rest: 87 MB of weights, a 64 MB
recycling pool, and the dequantized LSTM weights. A single synthesis pass peaks
higher, and that peak is set by `chunk_phonemes`, not by how long the text is —
longer text is split and the buffers are reused. On a 33 s passage:

| `chunk_phonemes` | peak | throughput |
|---|---|---|
| 510 | 1.14 GB | 9.0x realtime |
| 160 (default) | 0.69 GB | 9.0x realtime |
| 96 | 0.55 GB | 8.0x realtime |

Between utterances the pool is trimmed back to `gpu_memory_budget_mb` (64 MB by
default) and `went_idle` fires. Call `release_memory()` to force that sooner —
before a level load, say.

Over 100 warm utterances the device does not move at all. It used to creep
33 KB per utterance — invisible in a ten-run test and a third of a gigabyte
over a long session — because the dynamic-quantize path allocated a scratch
min/max buffer per node that no tensor id owned, so nothing ever released it.
`tests/test_leak.gd` prints the counters that would show it coming back.

### Audio quality

`remove_vocoder_tone` (on by default) notches out a steady tone at 4800 Hz and
9600 Hz. Kokoro's exported iSTFT overlap-adds with hop 5 and its window does not
sum flat, so both frequencies sit ~25 dB above the surrounding spectrum and are
heard as a ring behind the speech. The numpy reference has the identical tone,
so this is the model's artefact, not the runtime's — the filter is a deliberate
departure from bit-faithfulness in favour of sounding better. Turn it off to
hear exactly what the graph produced.

Voice names encode language and gender: `af_heart` is American female,
`bm_george` British male, then `e` Spanish, `i` Italian, `p` Portuguese,
`f` French, `h` Hindi, `j` Japanese, `z` Chinese.

### Which languages `speak()` handles

The voice name picks the frontend. English (`a`/`b`) uses the misaki gold
lexicons — 90 k entries — with a normalizer for numbers, currency, times,
ordinals and abbreviations, regular-inflection lookup for words like
"walked" and "making", letter-name spelling for unknown initialisms ("GPU"),
and a letter-to-sound fallback for anything still unknown. Spanish, Italian
and Portuguese (`e`/`i`/`p`) use rule tables — those orthographies are
regular enough to be intelligible, but this is best effort, not misaki.

French, Hindi, Japanese and Chinese have no G2P here. `speak()` refuses
rather than mispronouncing; feed those voices IPA through
`speak_phonemes()`.

## How it works

```
text ──► G2P ──► phoneme ids ──┐
voice.pt ──► style row [256] ──┤
                               ▼
                    graph executor (GDScript)
                    ├── 409 CPU nodes: shapes and indices
                    └── 3536 GPU nodes: compute dispatches
                               │
                               ▼
                 float waveform ──► PCM16 on GPU ──► AudioStreamWAV
```

`assets/model.kpk` is the whole network: a 16-byte header, a 534 KB JSON graph
(3945 nodes, integer tensor ids) and one 84.5 MB weight blob. It is produced
from the original ONNX by `tools/pack_model.py` and verified bit-exact. Weights
stay in their stored dtype (fp16 / int8) and are unpacked on the GPU.

Precision note: fp16 is treated as a storage format only and all compute is
fp32, which makes output slightly *closer* to the original unquantized Kokoro
than onnxruntime's own fp16 intermediates.

## Layout

```
addons/best-tts/
  kokoro_tts.gd             the engine node (threaded)
  best_tts.gd               BestTTS — the shared engine, no setup
  nodes/best_voice_player*.gd   AudioStreamPlayers that speak (plain, 2D, 3D)
  nodes/best_voice_stream.gd    a line as an AudioStreamWAV resource
  nodes/voice_player_impl.gd    logic the three players share
  assets/model.kpk          packed model — the only file needed at runtime
  assets/voices/*.pt        50 voice style packs
  assets/lexicons/*.json    misaki gold lexicons, US and GB
  g2p/g2p.gd                text -> IPA: normalizer, lexicon, fallback rules
  runtime/kpk_model.gd      packed-model loader (loads 85 MB in ~55 ms)
  runtime/gpu.gd            RenderingDevice, pipelines, buffers, dispatch
  runtime/executor.gd       the graph executor
  runtime/ops.gd            op codes + broadcasting/stride helpers
  runtime/{tokenizer,voice_loader,audio,shapes,cpu_ops}.gd
  kernels/*.glsl            compute shaders
tests/                      test scripts
tools/                      dev-only Python; see tools/NOTES.md
```

## Running the tests

```bash
godot --headless --import
```

```bash
godot --headless --script res://tests/test_load_model.gd
```

GPU tests need a real rendering driver — `--headless` selects the dummy one:

```bash
godot --rendering-driver vulkan --script res://tests/test_gpu_elementwise.gd
```

## Roadmap

- [x] Graph analysis, CPU/GPU node classification
- [x] numpy reference runtime — **all 55 op types verified against onnxruntime**
- [x] `model.kpk` packer, verified lossless (max|err| = 0)
- [x] GDScript packed-model loader
- [x] `RenderingDevice` wrapper and `elementwise.glsl` (~2600 of 3536 GPU nodes)
- [x] Voice loader (`ZIPReader`), vocab, tokenizer, chunker
- [x] Audio output stage: `pcm16.glsl` → `AudioStreamWAV`
- [x] Interactive capability bench (`demo/test_bench.tscn`)
- [x] All 17 compute kernels written (`matmul`, `matmul_int8`, `conv`,
      `conv_int8`, `lstm`, `reduce`, `layernorm`, `softmax`, `gather`, `copy`,
      `minmax`, `quantize`, `stft`, `misc`, `pcm16`, `elementwise`, `fir`)
- [x] CPU op set (13 shape/index ops) and shape inference for all 55 ops
- [x] Graph executor — runs all 3945 nodes, 3602 dispatches
- [x] **Numerical parity** — see "How close is it?" below
- [x] G2P: English via the misaki gold lexicons, rule-based es/it/pt
- [x] `KokoroTTS` node: threaded engine, `speak`, `speak_streaming`,
      `speak_phonemes`, chunking of long text
- [x] Bounded GPU memory: bucketed buffer pool, idle trim, `went_idle`
- [x] Notch out the model's 4800/9600 Hz iSTFT tone (`kernels/fir.glsl`)
- [x] Production API: `Request` handles, cancellation, priority queue, a
      bounded backlog, a two-tier clip cache, word/phoneme timings, and a
      teardown that does not stall a frame
- [x] Loudness normalization across voices (12.3 dB spread -> 2.1 dB)
- [x] Zero-setup node layer: `BestVoicePlayer` (+2D/3D), `BestVoiceStream`,
      `BestTTS`, and a demo scene that walks through all of it
- [ ] G2P for fr/hi/ja/zh (those voices need IPA via `speak_phonemes()`)
- [ ] English heteronyms ("read", "lead") need part-of-speech context
- [ ] Optional: keep the waveform on the GPU for `AudioStreamGenerator`
      instead of a readback
- [ ] Blocked, not planned: a custom `AudioStream` that synthesizes during
      playback. `AudioStreamPlayback._mix` hands over a raw buffer pointer
      GDScript cannot write to, and `super()` is unavailable for engine
      virtuals, so this needs GDExtension — which the project rules out.

## How close is it?

`tools/compare_speech.py` runs the same token ids through Godot and through
Python and reports the difference. For "Hello from Godot. This is Kokoro
speaking." (44 tokens):

| | spectral corr. | envelope corr. |
|---|---|---|
| Godot vs numpy reference | 0.986 | 0.994 |
| Godot vs onnxruntime | 0.926 | 0.803 |
| numpy reference vs onnxruntime | 0.941 | 0.809 |

The third row is the one that matters. The numpy reference passes 134/134
per-op tests against onnxruntime, so the distance *it* sits from ORT is the
floor imposed by the fp32 precision policy — and Godot sits at the same
distance. Against the reference it actually shares a policy with, Godot
matches to 0.99.

Predicted phoneme durations agree to a mean of 0.02 frames. On that sentence
exactly one phoneme of 44 rounded differently, and only because the reference
predicted 3.5081 frames — eight thousandths from a coin-flip tie. ONNX `Round`
is half-to-even, so a tie is decided by numerical noise; one frame is 25 ms.

Sample-by-sample correlation is *not* a useful metric here and the script
deliberately does not gate on it: the vocoder integrates F0 through a `CumSum`
to build its excitation, so a tiny frequency difference accumulates into a
growing phase offset. The waveforms drift apart while carrying identical
speech, which is exactly what a magnitude spectrogram is for.

`compare_speech.py` also prints a **per-band spectral tilt** and gates on it,
because correlation is blind to spectral shape — a whole band can be 60% hot
while correlation stays at 0.98. Worst band is currently 0.6 dB.

Neither metric sees a narrow tone, which is why `tools/analyze_tone.py` exists
separately: it takes the median spectrum across frames, so only *sustained*
peaks survive, and measures each against its own neighbourhood.

### The test suite

GPU tests need a real rendering driver, so they run with `--script` rather
than `--headless`:

| script | checks |
|---|---|
| `tests/test_load_model.gd` | the packed model parses (headless) |
| `tests/test_demo_scene.gd` | the demo scene builds, and survives having no GPU (headless) |
| `tests/test_nodes.gd` | `BestTTS`, `BestVoiceStream` and the three players |
| `tests/test_g2p.gd` | text → IPA, every symbol in the vocabulary (headless) |
| `tests/test_gpu_elementwise.gd` | kernels against a CPU reference |
| `tests/test_executor.gd` | the final waveform against the reference |
| `tests/test_taps.gd` | the earliest tensor that disagrees, in graph order |
| `tests/test_repeat.gd` | the executor is reusable and deterministic |
| `tests/test_speak.gd` | text in, `.wav` out; also gates the 4800/9600 Hz tone |
| `tests/test_memory.gd` | GPU memory plateaus instead of climbing per run |
| `tests/test_stress.gd` | the production suite — see below |
| `tests/test_leak.gd` | per-run buffer counters, for hunting a slow leak |
| `tests/test_calibrate_voices.gd` | regenerates the per-voice loudness table |
| `tests/test_dump_taps.gd` | dumps named tensors for Python to inspect |

`test_stress.gd` is the one that answers "will this survive a shipped game". It
runs ten sections, and `--only=cache,timings` picks a subset:

| section | asks |
|---|---|
| `abuse` | 28 kinds of hostile text: emoji, NULs, 600-letter words, markup, a 300-sentence page. Speak it or refuse with a reason — never return silence |
| `voices` | every bundled pack, bad voice names, unknown IPA, and speeds from −1 to 100 |
| `concurrency` | ten NPCs on one frame keep their own results; HIGH overtakes a backlog; the queue stays bounded |
| `cancel` | a run on the GPU stops in 348 ms, cancelling one request leaves its neighbours alone, the engine still works after |
| `cache` | hits are byte-identical and free, voice/speed/filter changes miss, the budget is enforced, the disk tier survives losing memory |
| `loudness` | the 12 dB spread across packs closes to ~2 dB on a sentence the calibration never saw, and nothing clips |
| `timings` | word counts, ordering, and an energy measurement proving the spans line up with the audio rather than merely looking ordered |
| `main_thread` | nothing costs the caller more than 2 ms |
| `soak` | 100 utterances: no memory creep, no slowdown, identical bytes for identical input |
| `lifecycle` | two engines at once, freeing the node mid-utterance, requests after `shutdown()` |

`test_speak.gd --raw` disables the notch filter, which is both an A/B listen and
a check that the tone gate is real (it goes from −40 dB to +29 dB).

### Debugging workflow

When a number is wrong, find the first node where it goes wrong:

```bash
python tools/export_test_case.py --tokens 9 && python tools/export_taps.py --tokens 9 --max-elems 60000
```

```bash
godot --rendering-driver vulkan --script res://tests/test_taps.gd
```

`test_taps.gd` walks the graph in order and reports the earliest divergence —
that is always where the bug is. It flags non-finite values separately,
because NaN loses every comparison and would otherwise slip through a
max-error check. `test_executor.gd` takes `--nodes=N` and `--verbose-nodes`
for bisecting.

When the numbers are fine but it *sounds* wrong, compare whole utterances:

```bash
godot --rendering-driver vulkan --script res://tests/test_export_speech.gd
```

```bash
python tools/compare_speech.py --engine ort --control
```

That writes `tests/out/godot.wav` and `tests/out/reference.wav` to listen to
side by side, and prints the duration table plus the metrics above.

Each kernel is also validated against real tensors captured from the model by
`tools/op_test.py --export`, so nothing is written blind.

## Licenses

- this addon — Apache-2.0, Copyright 2026 Studio Ransom. Free for any use,
  commercial included; the audio it produces carries no obligation.
- Kokoro-82M — Apache-2.0 (hexgrad)
- misaki G2P lexicons — MIT (hexgrad), once bundled

See `LICENSE`, `NOTICE` and `THIRD_PARTY.md` at the repository root.
