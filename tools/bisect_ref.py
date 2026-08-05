"""Dev-only: find where our numpy runtime first diverges from onnxruntime.

Walks the graph in execution order, probing ORT for a sample of typed float
tensors and comparing each against the numpy reference. The first tap whose
relative error jumps is where the bug lives.
"""
import argparse
import os

import numpy as np
import onnx

from probe import probe
from ref_runtime import RefRuntime
from voice_util import ROOT, load_voice, sample_tokens

MODEL = os.path.join(ROOT, "tools", "model_src", "model_q8f16.onnx")


def typed_float_tensors():
    """Tensor names ORT can safely expose, in graph execution order."""
    m = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=False,
                                          data_prop=True)
    typed = {vi.name: vi for vi in m.graph.value_info}
    order = []
    for n in m.graph.node:
        for o in n.output:
            vi = typed.get(o)
            if vi is None:
                continue
            et = vi.type.tensor_type.elem_type
            if et in (1, 10):                     # float32 / float16
                order.append((o, n.op_type, n.name))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stride", type=int, default=None,
                    help="sample every Nth tap (default: ~40 taps total)")
    ap.add_argument("--filter", default=None,
                    help="only taps whose name contains this substring")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = sample_tokens(args.tokens)
    style = load_voice(args.voice, len(ids))
    print("running numpy reference...")
    env = RefRuntime().run(ids, style)

    taps = [t for t in typed_float_tensors() if t[0] in env]
    if args.filter:
        taps = [t for t in taps if args.filter in t[0]]
    stride = args.stride or max(1, len(taps) // 40)
    taps = taps[::stride]
    if args.limit:
        taps = taps[:args.limit]
    print(f"{len(taps)} taps (stride {stride})")

    rows = []
    for i in range(0, len(taps), args.batch):
        chunk = taps[i:i + args.batch]
        names = [c[0] for c in chunk]
        try:
            ref = probe(names, ids, args.voice)
        except RuntimeError as exc:
            print(f"  ! batch perturbed the run, retrying singly ({exc})")
            ref = {}
            for n in names:
                try:
                    ref.update(probe([n], ids, args.voice))
                except RuntimeError:
                    print(f"  ! skipping untappable {n}")
        for name, op, nname in chunk:
            if name not in ref:
                continue
            a = np.asarray(ref[name]).astype(np.float64)
            b = np.asarray(env[name]).astype(np.float64)
            if a.shape != b.shape:
                rows.append((name, op, float("nan"), f"shape {a.shape}|{b.shape}"))
                continue
            scale = np.abs(a).mean() + 1e-12
            rel = np.abs(a - b).mean() / scale
            rows.append((name, op, rel, f"max={np.abs(a-b).max():.4g}"))
            print(f"  {rel:9.3e}  {op:24s} {name[-58:]}")

    print("\n--- first taps exceeding 1% relative error ---")
    for name, op, rel, extra in rows:
        if not (rel == rel) or rel > 0.01:
            print(f"  {rel:9.3e}  {op:24s} {name[-58:]}  {extra}")


if __name__ == "__main__":
    main()
