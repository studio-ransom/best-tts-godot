"""Dev-only: dump reference values for small tensors so Godot can self-check.

The executor and the numpy reference run the same graph, so the first tensor
where they disagree is where the bug is. Only modest tensors are exported,
which is enough to localize any divergence without a huge file.

Format: int32 count, then per tensor: int32 name_len, utf-8 name, int32 n,
float32[n].
"""
import argparse
import os
import struct

import numpy as np

from ref_runtime import RefRuntime
from voice_util import ROOT, load_voice, sample_tokens

OUT = os.path.join(ROOT, "tests", "_taps.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=9)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--max-elems", type=int, default=2048)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ids = sample_tokens(args.tokens)
    style = load_voice(args.voice, len(ids))
    rt = RefRuntime()
    env = rt.run(ids, style, 1.0)

    # Graph order matters: we want the *first* divergence, not just any.
    order = []
    seen = set()
    for node in rt.graph.node:
        for o in node.output:
            if o and o in env and o not in seen:
                seen.add(o)
                order.append(o)

    rows = []
    for name in order:
        v = np.asarray(env[name])
        if v.dtype.kind not in "fiub" or v.size == 0 or v.size > args.max_elems:
            continue
        rows.append((name, v.astype(np.float32).reshape(-1)))

    with open(args.out, "wb") as f:
        f.write(struct.pack("<i", len(rows)))
        for name, v in rows:
            nb = name.encode("utf-8")
            f.write(struct.pack("<i", len(nb)))
            f.write(nb)
            f.write(struct.pack("<i", v.size))
            f.write(v.tobytes())

    print(f"{args.out}: {len(rows)} tapped tensors "
          f"(of {len(order)}), {os.path.getsize(args.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
