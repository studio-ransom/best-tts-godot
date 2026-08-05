"""Dev-only: classify every node of the Kokoro ONNX graph.

Splits the graph three ways:
  CONST  - computable from initializers alone -> folded away at pack time
  CPU    - shape/index domain (small int tensors) -> runs in GDScript
  GPU    - bulk tensor math -> runs as a compute dispatch

Prints a compact inventory so we know exactly which kernels to write.
"""
import collections
import sys

import onnx

MODEL = "tools/model_src/model_q8f16.onnx"


def main():
    g = onnx.load(MODEL).graph

    inits = {t.name for t in g.initializer}
    graph_inputs = {i.name for i in g.input}

    producer = {}
    for n in g.node:
        for o in n.output:
            if o:
                producer[o] = n

    # ---- pass 1: constant folding reachability -------------------------------
    const = set(inits)
    # iterate to fixpoint (graph is topologically sorted, but be safe)
    changed = True
    while changed:
        changed = False
        for n in g.node:
            if all(o in const for o in n.output if o):
                continue
            if n.op_type in ("RandomNormal", "RandomUniform"):
                continue
            ins = [i for i in n.input if i]
            if ins and all(i in const for i in ins):
                for o in n.output:
                    if o:
                        const.add(o)
                changed = True

    # ---- pass 2: shape domain ------------------------------------------------
    # Seeded by Shape outputs; propagates forward through int-only ops.
    SHAPE_SAFE = {
        "Shape", "Gather", "Concat", "Unsqueeze", "Squeeze", "Slice", "Cast",
        "Add", "Sub", "Mul", "Div", "Mod", "Range", "ConstantOfShape", "Equal",
        "Where", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "Not",
        "And", "Or", "Min", "Max", "ReduceProd", "Expand", "Reshape", "Floor",
        "Ceil", "Clip", "CumSum", "NonZero", "Transpose", "Tile", "Split",
    }
    shape_dom = set()
    for n in g.node:
        if n.op_type == "Shape":
            shape_dom.update(o for o in n.output if o)
    changed = True
    while changed:
        changed = False
        for n in g.node:
            if n.op_type not in SHAPE_SAFE:
                continue
            outs = [o for o in n.output if o]
            if all(o in shape_dom for o in outs):
                continue
            ins = [i for i in n.input if i]
            # every non-constant input must itself be shape-domain
            if ins and all((i in shape_dom or i in const) for i in ins):
                # Reshape/Expand/Tile: only the *shape operand* is shape-domain,
                # the data operand is real data -> the node itself is not.
                if n.op_type in ("Reshape", "Expand", "Tile", "ConstantOfShape",
                                 "Slice", "Gather", "Transpose", "Split",
                                 "CumSum", "Where", "NonZero"):
                    if n.op_type == "ConstantOfShape":
                        pass  # output really is data-shaped; treat as data
                    data_in = n.input[0]
                    if data_in not in shape_dom and data_in not in const:
                        continue
                    if n.op_type in ("Reshape", "Expand", "Tile", "ConstantOfShape"):
                        continue
                for o in outs:
                    shape_dom.add(o)
                changed = True

    # ---- classify nodes ------------------------------------------------------
    kind = {}
    for n in g.node:
        outs = [o for o in n.output if o]
        if all(o in const for o in outs):
            kind[id(n)] = "CONST"
        elif n.op_type == "Shape" or all(o in shape_dom for o in outs):
            kind[id(n)] = "CPU"
        else:
            kind[id(n)] = "GPU"

    counts = collections.Counter(kind.values())
    print("=" * 64)
    print(f"total nodes {len(g.node)}   " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("=" * 64)

    for want in ("CONST", "CPU", "GPU"):
        c = collections.Counter(n.op_type for n in g.node if kind[id(n)] == want)
        print(f"\n--- {want} ({sum(c.values())} nodes, {len(c)} op types) ---")
        line = []
        for op, k in sorted(c.items(), key=lambda x: -x[1]):
            line.append(f"{op}:{k}")
        print("  " + "  ".join(line))

    # ops we must implement on the GPU
    gpu_ops = sorted({n.op_type for n in g.node if kind[id(n)] == "GPU"})
    cpu_ops = sorted({n.op_type for n in g.node if kind[id(n)] == "CPU"})
    print(f"\nGPU op types to implement ({len(gpu_ops)}): {' '.join(gpu_ops)}")
    print(f"CPU op types to implement ({len(cpu_ops)}): {' '.join(cpu_ops)}")

    # size of surviving initializers
    used = set()
    for n in g.node:
        if kind[id(n)] != "CONST":
            used.update(i for i in n.input if i)
    tot = kept = 0
    for t in g.initializer:
        nbytes = len(t.raw_data) if t.raw_data else 0
        tot += nbytes
        if t.name in used:
            kept += nbytes
    print(f"\ninitializer bytes: total {tot/1e6:.1f}MB  directly-used {kept/1e6:.1f}MB")


if __name__ == "__main__":
    sys.exit(main())
