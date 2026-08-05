"""Dev-only: run the model with every intermediate tensor exposed as an output.

Produces tools/_ref/<sanitized>.npy for every tensor plus a manifest, giving us
ground truth (shape + dtype + values) for every node in the graph.
"""
import argparse
import json
import os
import re

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

from voice_util import ROOT, load_voice, sample_tokens

MODEL = os.path.join(ROOT, "tools", "model_src", "model_q8f16.onnx")
OUTDIR = os.path.join(ROOT, "tools", "_ref")


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def build_exposed(path, want=None):
    """Add intermediates as graph outputs, carrying their inferred types.

    Exposing a tensor with an undeclared type makes ORT mistype the uint8
    quantization tensors and silently corrupt the run, so anything shape
    inference cannot type is left alone.
    """
    m = onnx.load(path)
    m = onnx.shape_inference.infer_shapes(m, strict_mode=False,
                                          data_prop=True)
    g = m.graph
    typed = {vi.name: vi for vi in g.value_info}
    existing = {o.name for o in g.output}
    inits = {t.name for t in g.initializer}

    skipped = 0
    for n in g.node:
        for o in n.output:
            if not o or o in existing or o in inits:
                continue
            if want is not None and not any(s in o for s in want):
                continue
            vi = typed.get(o)
            if vi is None or not vi.type.tensor_type.HasField("elem_type"):
                skipped += 1
                continue
            g.output.append(vi)
            existing.add(o)
    return m, [o.name for o in g.output], skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=9)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--save-values", action="store_true",
                    help="write .npy per tensor (large); otherwise manifest only")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; only save matching tensors")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    only = args.only.split(",") if args.only else None
    m, out_names, skipped = build_exposed(MODEL, only)
    print(f"exposing {len(out_names)} tensors ({skipped} untypable, skipped)")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])

    ids = sample_tokens(args.tokens)
    style = load_voice(args.voice, len(ids))
    feeds = {
        "input_ids": np.array([ids], dtype=np.int64),
        "style": np.ascontiguousarray(style.reshape(1, 256), dtype=np.float32),
        "speed": np.array([1.0], dtype=np.float32),
    }

    vals = sess.run(out_names, feeds)

    manifest = {"tokens": ids, "voice": args.voice, "tensors": {}}
    for name, v in zip(out_names, vals):
        v = np.asarray(v)
        entry = {"shape": list(v.shape), "dtype": str(v.dtype)}
        if v.size and v.dtype.kind == "f":
            entry["min"] = float(v.min())
            entry["max"] = float(v.max())
        manifest["tensors"][name] = entry
        if args.save_values and (only is None or any(s in name for s in only)):
            np.save(os.path.join(OUTDIR, sanitize(name) + ".npy"), v)

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(out_names)} tensors -> {OUTDIR}/manifest.json")
    wav = manifest["tensors"].get("waveform")
    print("waveform:", wav)


if __name__ == "__main__":
    main()
