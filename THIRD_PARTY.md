# Third-party components

best-tts bundles the following. Everything here is redistributable under a
permissive licence, and audio synthesized with it may be used commercially.

## Kokoro-82M — Apache-2.0

The neural network itself, and all 50 voice style packs.

- Original model: https://huggingface.co/hexgrad/Kokoro-82M
- Author: hexgrad
- Licence: Apache License 2.0

`addons/best-tts/assets/model.kpk` is a repacked form of this model — the graph
and weights rearranged into a single file the GDScript runtime can load in one
read. It is verified bit-exact against the source (max absolute error 0). The
repacking is performed by `tools/pack_model.py`.

`addons/best-tts/assets/voices/*.pt` are the upstream voice tensors, unmodified.

### Quantized ONNX export

The repack starts from a Q8F16 ONNX export of the above (int8 dynamically
quantized matmuls and convolutions, fp16 weights elsewhere):

- https://huggingface.co/tonythethompson/Kokoro-82M-Q8F16-ONNX

## misaki G2P lexicons — MIT

The English pronunciation dictionaries, ~90,000 entries across US and GB.

- Project: https://github.com/hexgrad/misaki
- Author: hexgrad
- Licence: MIT

Bundled as `addons/best-tts/assets/lexicons/us_gold.json` and `gb_gold.json`,
converted from the upstream format. The normalizer, inflection handling and
letter-to-sound fallback in `addons/best-tts/g2p/g2p.gd` are an independent
GDScript implementation informed by misaki's behaviour, not a port of its code.

## Not bundled

- **onnxruntime**, **PyTorch**, **numpy**, **scipy** — used only by the
  development tooling under `tools/`, which never ships and never runs at
  runtime. The addon has no Python dependency of any kind.

---

Everything else in this repository — the GDScript runtime, the GLSL compute
kernels, the model packer, the test suite — is original work,
Copyright 2026 Studio Ransom, licensed Apache-2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

Free for any use including commercial, with no restriction on what you build
with it or what you do with the audio it produces.
