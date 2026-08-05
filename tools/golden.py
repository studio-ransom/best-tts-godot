"""Dev-only: capture ORT's output as the golden reference.

Only one ORT configuration is trustworthy for this model: graph optimizations
fully disabled, with no intermediate tensors exposed. ENABLE_ALL segfaults and
exposing intermediates perturbs ORT's mandatory fp16->fp32 cast insertion,
which silently corrupts the run. So we snapshot just the final waveform here
and use tools/ref_runtime.py for per-node ground truth.
"""
import argparse
import os

import numpy as np
import onnxruntime as ort

from voice_util import ROOT, load_voice, sample_tokens

MODEL = os.path.join(ROOT, "tools", "model_src", "model_q8f16.onnx")
OUTDIR = os.path.join(ROOT, "tools", "_ref")


def run(ids, voice="af_heart", speed=1.0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
    style = np.ascontiguousarray(load_voice(voice, len(ids)).reshape(1, 256),
                                 dtype=np.float32)
    return sess.run(None, {
        "input_ids": np.array([ids], dtype=np.int64),
        "style": style,
        "speed": np.array([speed], dtype=np.float32),
    })[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    ids = sample_tokens(args.tokens)
    wav = run(ids, args.voice, args.speed)
    tag = f"{args.voice}_t{args.tokens}_s{args.speed:g}"
    np.save(os.path.join(OUTDIR, f"golden_{tag}.npy"), wav)
    np.save(os.path.join(OUTDIR, f"tokens_{tag}.npy"), np.array(ids, np.int64))
    print(f"golden_{tag}: {wav.shape} min={wav.min():.4f} max={wav.max():.4f} "
          f"rms={np.sqrt((wav.astype(np.float64)**2).mean()):.4f}")


if __name__ == "__main__":
    main()
