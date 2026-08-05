"""Dev-only: turn the .onnx into `model.kpk`, the artifact the addon ships.

The runtime is pure GDScript and must not parse protobuf, so all the digestion
happens here: constant folding, topological ordering, CPU/GPU placement, integer
tensor ids, and a single flat weight blob that GDScript uploads to the GPU with
one `FileAccess.get_buffer` per tensor.

Once the packed model is verified, `model_q8f16.onnx` can be deleted from the
addon -- nothing at runtime reads it.

File layout (little-endian):

    "KPK1"          magic
    u32 version
    u32 json_len
    u32 blob_len
    json_len bytes  graph description, utf-8 JSON, zero-padded to 4 bytes
    blob_len bytes  weight data, each tensor 4-byte aligned
"""
import argparse
import collections
import json
import os
import struct

import numpy as np
import onnx
from onnx import numpy_helper

import ref_runtime
from ref_runtime import MODEL, RefRuntime, _attrs
from voice_util import ROOT

OUT = os.path.join(ROOT, "addons", "best-tts", "assets", "model.kpk")
VERSION = 1

# dtype codes shared with the GDScript loader
DT_F32, DT_I8, DT_U8, DT_I32, DT_I64, DT_BOOL, DT_F16 = 0, 1, 2, 3, 4, 5, 6
NP_TO_CODE = {
    np.dtype("float32"): DT_F32, np.dtype("int8"): DT_I8,
    np.dtype("uint8"): DT_U8, np.dtype("int32"): DT_I32,
    np.dtype("int64"): DT_I64, np.dtype("bool"): DT_BOOL,
    np.dtype("float16"): DT_F16,
}

PLACE_CPU, PLACE_GPU = 0, 1

# Ops the GDScript side runs on the CPU: they move shapes and indices around,
# never bulk tensor data. Mirrors tools/analyze.py.
SHAPE_SAFE = {
    "Shape", "Gather", "Concat", "Unsqueeze", "Squeeze", "Slice", "Cast",
    "Add", "Sub", "Mul", "Div", "Mod", "Range", "Equal", "Where", "Greater",
    "Less", "GreaterOrEqual", "LessOrEqual", "Not", "And", "Or", "Min", "Max",
    "ReduceProd", "Floor", "Ceil", "Clip", "CumSum", "NonZero", "Transpose",
    "Tile", "Split",
}


def fold_constants(graph, inits):
    """Evaluate nodes whose inputs are all constant; return folded values."""
    rt = RefRuntime.__new__(RefRuntime)
    rt.model, rt.graph, rt.inits = None, graph, inits
    rt.fp16_boundaries = False
    rt.quantized_lstm = False
    rt.fp16_outputs = set()

    const = dict(inits)
    folded = {}
    for node in graph.node:
        ins = [i for i in node.input if i]
        if not ins or not all(i in const for i in ins):
            continue
        if node.op_type in ("RandomNormal", "RandomUniform"):
            continue
        try:
            rt._exec(node, const)
        except Exception:
            continue
        for o in node.output:
            if o and o in const:
                folded[o] = const[o]
    return folded


def classify(graph, const_names):
    """Split remaining nodes into CPU shape-domain and GPU tensor-math."""
    shape_dom = set()
    for n in graph.node:
        if n.op_type == "Shape":
            shape_dom.update(o for o in n.output if o)
    changed = True
    while changed:
        changed = False
        for n in graph.node:
            if n.op_type not in SHAPE_SAFE:
                continue
            outs = [o for o in n.output if o]
            if not outs or all(o in shape_dom for o in outs):
                continue
            ins = [i for i in n.input if i]
            if not ins or not all(i in shape_dom or i in const_names
                                  for i in ins):
                continue
            # For these, only a non-data operand being shape-domain is not
            # enough -- the data operand decides.
            if n.op_type in ("Slice", "Gather", "Transpose", "Split", "CumSum",
                             "Where", "NonZero"):
                if n.input[0] not in shape_dom and n.input[0] not in const_names:
                    continue
            shape_dom.update(outs)
            changed = True
    return shape_dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--keep-unused", action="store_true")
    args = ap.parse_args()

    model = onnx.load(MODEL)
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    print(f"{len(graph.node)} nodes, {len(inits)} initializers")

    folded = fold_constants(graph, inits)
    print(f"constant-folded {len(folded)} tensors")

    consts = dict(inits)
    consts.update(folded)
    dead = {n.output[0] for n in graph.node
            if n.output and n.output[0] in folded}

    live_nodes = [n for n in graph.node
                  if not (n.output and all(o in folded for o in n.output if o))]
    shape_dom = classify(graph, set(consts))

    # ---- assign tensor ids ------------------------------------------------
    names = []
    tid = {}

    def get_tid(name):
        if name not in tid:
            tid[name] = len(names)
            names.append(name)
        return tid[name]

    for name in ("input_ids", "style", "speed"):
        get_tid(name)
    for n in live_nodes:
        for i in n.input:
            if i:
                get_tid(i)
        for o in n.output:
            if o:
                get_tid(o)

    # ---- weights ----------------------------------------------------------
    used = {i for n in live_nodes for i in n.input if i}
    blob = bytearray()
    weights = {}
    for name, arr in consts.items():
        if name not in used and not args.keep_unused:
            continue
        # ascontiguousarray promotes 0-d to 1-d, which would silently change
        # the rank of every scalar constant, so restore the original shape.
        shape = arr.shape
        arr = np.ascontiguousarray(arr).reshape(shape)
        if arr.dtype not in NP_TO_CODE:
            arr = arr.astype(np.float32)
        while len(blob) % 4:
            blob.append(0)
        weights[str(get_tid(name))] = {
            "o": len(blob), "n": arr.nbytes,
            "d": NP_TO_CODE[arr.dtype], "s": list(arr.shape),
        }
        blob += arr.tobytes()

    # ---- nodes ------------------------------------------------------------
    op_names, op_id = [], {}

    def get_op(op):
        if op not in op_id:
            op_id[op] = len(op_names)
            op_names.append(op)
        return op_id[op]

    nodes = []
    place_counts = collections.Counter()
    for n in live_nodes:
        outs = [o for o in n.output if o]
        p = (PLACE_CPU if (n.op_type == "Shape"
                           or all(o in shape_dom for o in outs))
             else PLACE_GPU)
        place_counts[p] += 1
        at = _attrs(n)
        clean = {}
        for k, v in at.items():
            if isinstance(v, np.ndarray):
                clean[k] = {"__t": NP_TO_CODE[v.dtype] if v.dtype in NP_TO_CODE
                            else DT_F32,
                            "v": v.reshape(-1).tolist(), "s": list(v.shape)}
            elif isinstance(v, bytes):
                clean[k] = v.decode()
            else:
                clean[k] = v
        nodes.append([
            get_op(n.op_type),
            [get_tid(i) if i else -1 for i in n.input],
            [get_tid(o) if o else -1 for o in n.output],
            clean or None,
            p,
        ])

    desc = {
        "version": VERSION,
        "ops": op_names,
        "tensors": names,
        "weights": weights,
        "nodes": nodes,
        "inputs": {"input_ids": tid["input_ids"], "style": tid["style"],
                   "speed": tid["speed"]},
        "output": tid["waveform"],
        "sample_rate": 24000,
        "samples_per_frame": 300,
    }

    payload = json.dumps(desc, separators=(",", ":")).encode("utf-8")
    payload += b"\0" * (-len(payload) % 4)

    with open(args.out, "wb") as f:
        f.write(b"KPK1")
        f.write(struct.pack("<III", VERSION, len(payload), len(blob)))
        f.write(payload)
        f.write(blob)

    total = os.path.getsize(args.out)
    print(f"nodes kept {len(nodes)} "
          f"(cpu {place_counts[PLACE_CPU]}, gpu {place_counts[PLACE_GPU]})")
    print(f"weights {len(weights)} tensors, {len(blob)/1e6:.1f} MB")
    print(f"json {len(payload)/1e3:.0f} KB")
    print(f"-> {args.out}  ({total/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
