"""Dev-only: verify each op implementation against onnxruntime in isolation.

End-to-end comparison is confounded by accumulated fp16 rounding, so instead we
extract single nodes into minimal models, feed them the exact tensors the numpy
runtime produced, and check our result for that one node. A wrong op shows up
immediately and unambiguously.

The captured (inputs, expected output) pairs are also what the GLSL kernels are
tested against later, via ``--export``.
"""
import argparse
import collections
import os
import re

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

import ref_runtime
from ref_runtime import MODEL, RefRuntime
from voice_util import ROOT, load_voice, sample_tokens

NP_TO_ONNX = {
    np.dtype("float32"): TensorProto.FLOAT,
    np.dtype("float16"): TensorProto.FLOAT16,
    np.dtype("uint8"): TensorProto.UINT8,
    np.dtype("int8"): TensorProto.INT8,
    np.dtype("int32"): TensorProto.INT32,
    np.dtype("int64"): TensorProto.INT64,
    np.dtype("bool"): TensorProto.BOOL,
    np.dtype("float64"): TensorProto.DOUBLE,
}

OUTDIR = os.path.join(ROOT, "tools", "_ref", "ops")


def build_single_node(node, inits, feed_names):
    """A minimal model containing just `node`."""
    inputs = []
    for name in feed_names:
        inputs.append(helper.make_empty_tensor_value_info(name))
    graph = helper.make_graph(
        [node], "single", inputs,
        [helper.make_empty_tensor_value_info(o) for o in node.output if o],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 20),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = 9
    return model


def typed_input(name, arr):
    return helper.make_tensor_value_info(
        name, NP_TO_ONNX[arr.dtype], list(arr.shape))


def run_node(node, inits, feeds):
    inputs = [typed_input(n, v) for n, v in feeds.items()]
    graph = helper.make_graph(
        [node], "single", inputs,
        [helper.make_empty_tensor_value_info(o) for o in node.output if o],
        initializer=inits)
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 20),
        helper.make_opsetid("com.microsoft", 1),
    ])
    model.ir_version = 9
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def compare(got, want):
    a = np.asarray(want).astype(np.float64)
    b = np.asarray(got).astype(np.float64)
    if a.shape != b.shape:
        return None, f"shape {a.shape} vs {b.shape}"
    if a.size == 0:
        return 0.0, "empty"
    denom = np.abs(a).mean() + 1e-9
    rel = np.abs(a - b).mean() / denom
    return rel, f"max={np.abs(a - b).max():.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--per-op", type=int, default=2,
                    help="how many nodes to test per op type")
    ap.add_argument("--op", default=None, help="only this op type")
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--export", action="store_true",
                    help="save inputs/outputs as .npz for the GLSL tests")
    ap.add_argument("--fp32-lstm", action="store_true",
                    help="use the shipping fp32 LSTM instead of ORT's "
                         "activation-quantizing one (will show ~1e-2 error)")
    args = ap.parse_args()

    ids = sample_tokens(args.tokens)
    style = load_voice(args.voice, len(ids))
    print("running numpy reference to capture live tensors...")
    rt = RefRuntime(quantized_lstm=not args.fp32_lstm)
    env = rt.run(ids, style)

    graph = rt.graph
    init_by_name = {t.name: t for t in graph.initializer}

    picked = collections.defaultdict(list)
    for node in graph.node:
        if args.op and node.op_type != args.op:
            continue
        if len(picked[node.op_type]) >= args.per_op:
            continue
        if not all(i in env for i in node.input if i):
            continue
        picked[node.op_type].append(node)

    if args.export:
        os.makedirs(OUTDIR, exist_ok=True)

    results = []
    for op in sorted(picked):
        for node in picked[op]:
            feeds, inits = {}, []
            for i in node.input:
                if not i:
                    continue
                if i in init_by_name:
                    t = init_by_name[i]
                    if t.data_type == TensorProto.FLOAT16:
                        # Feed fp16 weights as fp32, which is what ORT's CPU EP
                        # does internally; otherwise the single-node model has
                        # fp16 weights against fp32 activations and ORT rejects
                        # it on type-binding grounds.
                        t = numpy_helper.from_array(
                            numpy_helper.to_array(t).astype(np.float32), t.name)
                    inits.append(t)
                else:
                    # ascontiguousarray promotes 0-d to 1-d; keep scalars scalar
                    v = np.asarray(env[i])
                    v = np.ascontiguousarray(v).reshape(v.shape)
                    if v.dtype not in NP_TO_ONNX or v.dtype == np.float16:
                        v = v.astype(np.float32)
                    if v.ndim == 0:
                        # A 0-d graph input reads as unknown-rank to ORT and it
                        # infers the wrong output rank; as an initializer the
                        # scalar-ness is unambiguous.
                        inits.append(numpy_helper.from_array(v, i))
                    else:
                        feeds[i] = v
            try:
                outs = run_node(node, inits, feeds)
            except Exception as exc:
                results.append((op, node.name, None, f"ORT failed: {exc}"))
                continue

            worst, detail = 0.0, ""
            for oname, got in zip([o for o in node.output if o], outs):
                rel, d = compare(env[oname], got)
                if rel is None:
                    worst, detail = None, d
                    break
                if rel > worst:
                    worst, detail = rel, d
            results.append((op, node.name, worst, detail))

            if args.export:
                payload = {f"in::{k}": v for k, v in feeds.items()}
                for t in inits:
                    payload[f"init::{t.name}"] = numpy_helper.to_array(t)
                for oname, got in zip([o for o in node.output if o], outs):
                    payload[f"out::{oname}"] = np.asarray(got)
                fn = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{op}__{node.name}")[:150]
                np.savez_compressed(os.path.join(OUTDIR, fn + ".npz"), **payload)

    bad = []
    print(f"\n{'op':26s} {'rel err':>10s}  detail")
    print("-" * 78)
    for op, name, rel, detail in results:
        flag = ""
        if rel is None or rel > args.tol:
            flag = "  <== FAIL"
            bad.append((op, name, rel, detail))
        shown = "shape" if rel is None else f"{rel:10.3e}"
        print(f"{op:26s} {shown}  {detail}{flag}")

    print(f"\n{len(results) - len(bad)}/{len(results)} node tests within "
          f"rel tol {args.tol:g}")
    if bad:
        print("\nfailing ops:")
        for op, name, rel, detail in bad:
            print(f"  {op:24s} {name}\n      {detail}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
