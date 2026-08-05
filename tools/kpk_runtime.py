"""Dev-only: run the model straight from `model.kpk`, with no onnx involved.

This is the reference implementation of the *loader* -- it does exactly what the
GDScript runtime has to do (read the header, parse the JSON graph, slice weights
out of the blob, walk the nodes) while reusing the already-verified op
semantics. If this reproduces the ONNX-driven waveform, the packed file is
lossless and the GDScript port only has to match this.
"""
import argparse
import json
import os
import struct

import numpy as np

from ref_runtime import RefRuntime
from voice_util import ROOT, load_voice, sample_tokens

KPK = os.path.join(ROOT, "addons", "best-tts", "assets", "model.kpk")

CODE_TO_NP = {
    0: np.float32, 1: np.int8, 2: np.uint8, 3: np.int32,
    4: np.int64, 5: np.bool_, 6: np.float16,
}


class PackedModel:
    def __init__(self, path=KPK):
        with open(path, "rb") as f:
            data = f.read()
        if data[:4] != b"KPK1":
            raise ValueError("not a kpk file")
        version, json_len, blob_len = struct.unpack_from("<III", data, 4)
        head = 16
        self.version = version
        self.desc = json.loads(data[head:head + json_len].rstrip(b"\0"))
        blob = data[head + json_len: head + json_len + blob_len]

        self.names = self.desc["tensors"]
        self.ops = self.desc["ops"]
        self.nodes = self.desc["nodes"]
        self.inputs = self.desc["inputs"]
        self.output = self.desc["output"]

        self.weights = {}
        for sid, w in self.desc["weights"].items():
            dt = CODE_TO_NP[w["d"]]
            arr = np.frombuffer(blob, dtype=dt, count=w["n"] // np.dtype(dt).itemsize,
                                offset=w["o"])
            self.weights[int(sid)] = arr.reshape(w["s"])

    def run(self, input_ids, style, speed=1.0, quantized_lstm=False):
        rt = RefRuntime.__new__(RefRuntime)
        rt.quantized_lstm = quantized_lstm
        rt.fp16_boundaries = False
        rt.fp16_outputs = set()

        # The op layer is keyed by tensor name, so hand it ids as strings.
        env = {str(k): v for k, v in self.weights.items()}
        env[str(self.inputs["input_ids"])] = np.asarray(
            input_ids, dtype=np.int64).reshape(1, -1)
        env[str(self.inputs["style"])] = np.asarray(
            style, dtype=np.float32).reshape(1, 256)
        env[str(self.inputs["speed"])] = np.asarray([speed], dtype=np.float32)

        for op_id, ins, outs, at, _place in self.nodes:
            attrs = {}
            for k, v in (at or {}).items():
                if isinstance(v, dict) and "__t" in v:
                    v = np.array(v["v"], dtype=CODE_TO_NP[v["__t"]]).reshape(v["s"])
                attrs[k] = v
            try:
                rt.dispatch(self.ops[op_id],
                            ["" if i < 0 else str(i) for i in ins],
                            ["" if o < 0 else str(o) for o in outs],
                            attrs, env)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.ops[op_id]} (tensors "
                    f"{[self.names[i] for i in ins if i >= 0]}): {exc}") from exc
        return env[str(self.output)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    ids = sample_tokens(args.tokens)
    style = load_voice(args.voice, len(ids))

    pm = PackedModel()
    print(f"kpk v{pm.version}: {len(pm.nodes)} nodes, "
          f"{len(pm.weights)} weight tensors")
    got = pm.run(ids, style, args.speed)
    print(f"packed waveform {got.shape} min={got.min():.4f} max={got.max():.4f}")

    want = RefRuntime().run(ids, style, args.speed)["waveform"]
    if want.shape != got.shape:
        print(f"SHAPE MISMATCH onnx={want.shape} kpk={got.shape}")
        return 1
    err = float(np.abs(want - got).max())
    print(f"vs onnx-driven run: max|err| = {err:.3e} "
          f"({'lossless' if err == 0 else 'DIFFERS'})")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
