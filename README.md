<div align="center">

# best-tts

**High-quality neural text-to-speech, running entirely inside Godot.**

Pure GDScript and GLSL compute shaders.
No C++. No GDExtension. No executables. No Python. No server. No API key.

[![Godot 4.4+](https://img.shields.io/badge/Godot-4.4%2B-478cbf?logo=godotengine&logoColor=white)](https://godotengine.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Renderer](https://img.shields.io/badge/renderer-Forward%2B%20%7C%20Mobile-lightgrey)](#requirements)
[![Voices](https://img.shields.io/badge/voices-50-4fd1c5)](#the-voices)

</div>

## Hear it

### ▶ **[Listen to the voices](https://studio-ransom.github.io/best-tts-godot/)**

Every clip came out of Godot on an RTX 4070 Ti — same line, different voice
pack, nothing touched afterwards. No EQ, no compression, no editing.

> *"Hello from Godot. This is Kokoro speaking."*

| voice | | |
|---|---|---|
| `am_adam` | American English, male | [▶ play](https://github.com/studio-ransom/best-tts-godot/raw/main/samples/am_adam.mp3) |
| `af_alloy` | American English, female | [▶ play](https://github.com/studio-ransom/best-tts-godot/raw/main/samples/af_alloy.mp3) |
| `am_onyx` | American English, male | [▶ play](https://github.com/studio-ransom/best-tts-godot/raw/main/samples/am_onyx.mp3) |
| `am_fenrir` | American English, male | [▶ play](https://github.com/studio-ransom/best-tts-godot/raw/main/samples/am_fenrir.mp3) |
| `em_alex` | Spanish pack reading English, for the accent | [▶ play](https://github.com/studio-ransom/best-tts-godot/raw/main/samples/em_alex.mp3) |

Five of fifty. The rest ship with the addon.

<!-- ---------------------------------------------------------------------
     WANT INLINE PLAYERS RIGHT HERE IN THE README?

     GitHub will not play repo-hosted media inline, whatever the markup:
       * <audio> is stripped by the HTML sanitizer outright.
       * <video> survives, but GitHub serves .mp4 from raw/ as
         Content-Type: application/octet-stream, so the browser refuses it.
         (Verified: .png -> image/png, .mp3 -> audio/mpeg, .mp4 -> octet-stream.)

     The only thing that works is media GitHub itself hosts. To get it:

       1. Open a new issue on this repo — do not submit it.
       2. Drag samples/am_adam.mp4 into the comment box and wait for upload.
       3. It becomes a URL like
            https://github.com/user-attachments/assets/<uuid>
       4. Repeat for the other four .mp4 files.
       5. Close the issue without submitting, and paste each URL here on its
          own line. GitHub renders each as a real player, waveform and all.

     The .mp4 files in samples/ exist for exactly this — they are the .mp3s
     with a waveform rendered over them, sized for a README.
     --------------------------------------------------------------------- -->

## Use it

```gdscript
$Voice.speak("I have wares, if you have coin.")
```

`$Voice` is a **BestVoicePlayer** — an AudioStreamPlayer with `text` and `voice`
fields in the inspector. That is the entire integration. No engine node to add,
no autoload to register, no plugin to enable, nothing to initialize.

---

## ⛏️ Support this project

best-tts is free and Apache-licensed. It was built for **Anthracite**, our co-op
survival horror game — a wishlist is the most useful thing you can give back,
and it costs nothing.

### 👉 **[Wishlist Anthracite on Steam](https://store.steampowered.com/app/4027400/Anthracite/)**

> *A co-op survival horror game where every expedition changes your survivor.
> Scavenge strange items, endure unpredictable effects, and extract alive.*

---

## Why pure GDScript matters

Every other way to get neural TTS into Godot drags something along with it: a
GDExtension you have to compile per platform, an onnxruntime DLL, a bundled
Python, a subprocess, or a cloud endpoint your players' firewall can block and
your budget pays for by the character.

This has none of that. The model is a data file; the runtime is `.gd` and
`.glsl` files that Godot already knows how to load.

| | |
|---|---|
| **Exports like any other project** | No per-platform native builds, no `.dll`/`.so`/`.dylib` to sign, notarize or ship |
| **No supply chain** | Nothing to `pip install`, nothing to vendor, no binary anyone has to trust |
| **Offline and unmetered** | No API key, no rate limit, no per-character cost, no outage |
| **Readable and hackable** | The whole executor is GDScript you can set a breakpoint in |
| **No player-side install** | They download your game; that's it |

The tradeoff is honest: it needs a Vulkan-capable GPU and the Forward+ or Mobile
renderer, because the compute shaders have to run somewhere.

## The voices

50 packs across nine languages' worth of speakers, 511 KB each — delete the ones
you don't ship. Voice names encode language and gender: `af_heart` is American
female, `bm_george` British male, then `e` Spanish, `i` Italian, `p` Portuguese,
`f` French, `h` Hindi, `j` Japanese, `z` Chinese.

**On quality.** The model is [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
an 82-million-parameter model that sounds far better than its size suggests —
which is exactly why it is the one worth running locally. The samples above are
the honest evidence; listen before you commit.

What this runtime adds on top of the model:

- **fp32 compute throughout.** Weights are stored quantized (int8/fp16) but every
  intermediate is fp32, which puts the output slightly *closer* to the original
  unquantized Kokoro than onnxruntime's own fp16 path.
- **The vocoder ring is gone.** Kokoro's exported iSTFT leaves a steady tone at
  4800 Hz and its octave, ~25 dB above the surrounding spectrum, audible as a
  ring behind the speech. It is in the reference implementation too. A notch
  filter removes it by default; `remove_vocoder_tone = false` to hear the raw
  graph output.
- **Matched loudness.** The packs are not mastered to a common level — across
  all 50 the gated loudness spans **12.3 dB**. One measured gain per voice
  (ITU-R BS.1770, gated, not RMS) closes that to **2.1 dB**, peak-limited so
  nothing clips. Switching speaker mid-scene no longer steps the volume.

## Try the demo

Clone, open, press **F5**. The main scene walks through the API in four panels —
the node, the resource, free-form text, and what happens under load — with the
code for each panel on screen next to the button that runs it.

```bash
git clone https://github.com/studio-ransom/best-tts-godot
```

The clone is ~200 MB; the packed model is most of it, and there is no way to
make a neural network small.

`demo/test_bench.tscn` is the other scene: a capability bench that exercises the
model loader, every GPU kernel against a CPU reference, the tokenizer and the
output stage, and reports what passed.

## Install into your own project

1. Copy `addons/best-tts/` into your project's `addons/` folder.
2. Set **Project Settings → Rendering → Renderer** to **Forward+** or **Mobile**.
3. Add a **BestVoicePlayer** node and type a line into it.

There is no plugin to enable. The nodes are `class_name` scripts, so they appear
in the Create Node dialog as soon as Godot rescans.

### Requirements

| | |
|---|---|
| Godot | **4.4+** (developed against 4.7) |
| Renderer | **Forward+** or **Mobile** — the Compatibility/GL backend has no `RenderingDevice` and cannot run compute shaders |
| GPU | Vulkan capable |
| Disk | 113 MB — an 85 MB model, 26 MB of voices, 5.6 MB of English lexicons |
| VRAM | ~190 MB at rest; a synthesis pass peaks around 0.7 GB and is tunable |

## The API

Three layers. Start at the top and go down only when you need to.

### 1 · Nodes

**BestVoicePlayer**, **BestVoicePlayer2D** and **BestVoicePlayer3D** are
AudioStreamPlayers that speak. `text`, `voice` and `speed` are inspector fields.

```gdscript
$Merchant/Voice.speak("I have wares, if you have coin.")
$Merchant/Voice.stop_speaking()
```

With `prepare_on_ready` left on, the line is synthesized while the scene loads
and lands in `stream` as ordinary audio — after that the node is an ordinary
player, and `play()`, `seek()` and `finished` all behave exactly as they always
do.

> **`speak()` is the verb, not `play()`.** GDScript cannot override a native
> method — Godot parses the override and then never calls it, not even from
> script — so `play()` on a line that has not been prepared has nothing to play.
> `speak()` always works; it synthesizes if it has to.

**BestVoiceStream** is a line authored as a resource: text and voice saved in a
`.tres` a writer can edit, or set on the player inside the scene file. It
extends `AudioStreamWAV`, so once rendered any stock AudioStreamPlayer plays it.

```gdscript
var line := BestVoiceStream.of("The bridge is out ahead.", "bm_george")
await line.render()
$AnyOldPlayer.stream = line
$AnyOldPlayer.play()
```

### 2 · The shared engine

**BestTTS** is a static class — no node, no autoload, no setup. The first call
creates the one engine the process needs and parents it to the tree root.

```gdscript
BestTTS.speak_into($Guard/Voice, "Halt! Who goes there?", "bm_george")
BestTTS.stream_into($Narrator, chapter_text)      # starts on the first clause
BestTTS.precache(BARK_LINES, "am_michael")        # during a loading screen
BestTTS.cancel_all()
```

| | |
|---|---|
| `warm_up()` | start loading the model without speaking |
| `speak(text, voice, speed)` | awaitable, returns an `AudioStreamWAV` |
| `say(text, voice, speed, priority)` | a `Request`, right now — no await |
| `speak_into(player, …)` / `stream_into(player, …)` | into a player you own |
| `precache(lines, voice)` | warm a set of barks at low priority |
| `cancel_all()` / `is_busy()` / `is_ready()` | engine state |
| `list_voices()` / `has_voice()` / `describe_voice()` | works with no engine |
| `configure(fn)` | set cache and voice defaults before the engine exists |
| `shutdown()` | release the ~190 MB; the next call builds a fresh one |

### 3 · The engine node

**KokoroTTS** is the node the other two layers use. Add one yourself when you
want per-scene settings, a separate cache, or control over when the model loads.
Every export — chunk size, GPU budget, cache tiers, loudness target, submission
granularity, timings — lives here.

The full table, the `Request` lifecycle, and the measured cost of everything are
in **[addons/best-tts/README.md](addons/best-tts/README.md)**.

## What you get for shipping it

**It costs the calling thread nothing.** Queueing a full page of text takes
0.06 ms on the main thread; a cache hit returns the clip in 0.16 ms. G2P, the
model and the GPU device all live on a worker.

**Cancellation that works.** Vulkan cannot un-queue submitted work, so the
executor submits in segments: a run already on the GPU is abandoned in ~350 ms,
and the throughput cost of that responsiveness is 2%. The dial is exposed.

**A clip cache.** Games repeat themselves. Memory hits are free and byte-identical;
an optional disk tier means a game's fixed dialogue is only ever synthesized on
the player's first playthrough.

**Word and phoneme timings** for subtitles and lip sync, taken from the duration
predictor's own numbers — so they describe the audio that was produced, not an
estimate over it. They survive chunking and ride along with cache hits.

**Bounded memory.** The buffer pool trims to a budget between utterances. Over
100 warm utterances the device does not grow at all.

**A priority queue** with a bounded backlog, because audio that arrives a minute
late is worse than audio that never arrives.

## Languages

The voice name picks the frontend. English (`a` US / `b` GB) uses the misaki
gold lexicons — 90 k entries — with a normalizer for numbers, currency, times,
ordinals and abbreviations, plus letter-to-sound fallback for anything unknown.
Spanish, Italian and Portuguese (`e`/`i`/`p`) use rule tables; those
orthographies are regular enough to be intelligible, but it is best effort.

French, Hindi, Japanese and Chinese voices have **no G2P here**. `speak()`
refuses rather than mispronouncing them — feed those voices IPA through
`speak_phonemes()`.

## Known limitations

- Compatibility/GL renderer is unsupported and always will be — there is no
  compute API to run on.
- English heteronyms ("read", "lead") need part-of-speech context this does not
  have yet.
- Web export is untested.
- A custom `AudioStream` that synthesizes during playback is not possible in
  pure GDScript: `AudioStreamPlayback._mix` hands over a raw buffer pointer that
  GDScript cannot write to. Hence `prepare()` rather than true on-demand
  streaming from the resource itself.

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
and one 84.5 MB weight blob, produced from the original ONNX by
`tools/pack_model.py` and verified bit-exact. Weights stay in their stored dtype
and are unpacked on the GPU. 17 compute kernels and a small CPU op set cover
every op type the graph uses.

Output is verified against onnxruntime: **0.986 spectral correlation** against a
numpy reference that itself passes 134/134 per-op tests, with predicted phoneme
durations agreeing to a mean of 0.02 frames. The methodology, and why
sample-by-sample correlation is the wrong metric for a vocoder, are written up
in the [addon README](addons/best-tts/README.md#how-close-is-it).

## Repository layout

```
addons/best-tts/     the addon — this is the only folder you need
demo/                the demo scene, and a capability bench
samples/             the audio at the top of this page
tests/               headless and GPU test scripts
tools/               dev-only Python: model packer, reference runtime,
                     comparison harness. Never ships, never runs at runtime.
```

## Running the tests

```bash
godot --headless --script res://tests/test_demo_scene.gd
```

GPU tests need a real rendering driver, since `--headless` selects the dummy one:

```bash
godot --rendering-driver vulkan --script res://tests/test_stress.gd
```

`test_stress.gd` is the suite that answers "will this survive a shipped game":
28 kinds of hostile text, every voice pack, ten concurrent requests, cancellation
mid-GPU-run, cache correctness, loudness, timings-versus-audio-energy,
main-thread cost, a 100-utterance soak, and lifecycle abuse like freeing the node
mid-utterance. `test_nodes.gd` covers the node layer. Full table in the
[addon README](addons/best-tts/README.md#the-test-suite).

## Contributing

Issues and pull requests welcome. The most useful contributions right now are
G2P frontends for French, Hindi, Japanese and Chinese — the voices are already
bundled and working, they just need text-to-IPA.

`tools/NOTES.md` is the engineering log: the non-obvious constraints, the bugs
that were expensive to find, and why several things are the way they are. Read
it before changing the executor.

## Licenses

| | |
|---|---|
| this addon | Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE) |
| Kokoro-82M | Apache-2.0, [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) |
| misaki G2P lexicons | MIT, hexgrad |

Free for any use, commercial included. Full attribution in
[THIRD_PARTY.md](THIRD_PARTY.md); audio you synthesize with it is yours, with
no licence obligation attached.

---

<div align="center">

If this saved you a voice-acting budget, **[wishlist Anthracite](https://store.steampowered.com/app/4027400/Anthracite/)**.

</div>
