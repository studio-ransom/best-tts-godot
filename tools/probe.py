"""Dev-only: read a few ORT intermediates without corrupting the run.

Exposing many intermediates perturbs ORT's mandatory fp16->fp32 cast insertion
and silently changes results, so this exposes only the requested tensors and
refuses to report anything unless the final waveform still matches the golden
run bit-for-bit.
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort

from voice_util import ROOT, load_voice, sample_tokens

MODEL = os.path.join(ROOT, "tools", "model_src", "model_q8f16.onnx")


def probe(names, ids, voice="af_heart", speed=1.0, verify=True):
    m = onnx.shape_inference.infer_shapes(onnx.load(MODEL), strict_mode=False,
                                          data_prop=True)
    g = m.graph
    typed = {vi.name: vi for vi in g.value_info}
    have = {o.name for o in g.output}
    added = []
    for n in names:
        if n in have:
            continue
        vi = typed.get(n)
        if vi is None:
            raise KeyError(f"no type info for {n}")
        g.output.append(vi)
        added.append(n)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(m.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    out_names = [o.name for o in g.output]
    style = np.ascontiguousarray(load_voice(voice, len(ids)).reshape(1, 256),
                                 dtype=np.float32)
    vals = sess.run(out_names, {
        "input_ids": np.array([ids], dtype=np.int64),
        "style": style,
        "speed": np.array([speed], dtype=np.float32),
    })
    res = dict(zip(out_names, vals))

    if verify:
        tag = f"{voice}_t{len(ids)}_s{speed:g}"
        gp = os.path.join(ROOT, "tools", "_ref", f"golden_{tag}.npy")
        if os.path.exists(gp):
            gold = np.load(gp)
            got = res["waveform"]
            if gold.shape != got.shape or not np.array_equal(gold, got):
                raise RuntimeError(
                    "exposing these tensors perturbed the run "
                    f"(golden {gold.shape} vs probed {got.shape}); "
                    "probe fewer tensors")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    ids = sample_tokens(args.tokens)
    res = probe(args.names, ids, args.voice)
    for n in args.names:
        v = np.asarray(res[n])
        print(f"\n{n}\n  shape={v.shape} dtype={v.dtype}")
        flat = v.reshape(-1)
        print("  " + np.array2string(flat[:40], precision=5, max_line_width=100))
        if args.save:
            import re
            fn = re.sub(r"[^A-Za-z0-9_.-]", "_", n) + ".npy"
            np.save(os.path.join(ROOT, "tools", "_ref", fn), v)


if __name__ == "__main__":
    main()
