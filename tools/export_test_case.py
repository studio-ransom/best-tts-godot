"""Dev-only: write a reference case the Godot executor can check itself against.

Format (little-endian): n_ids, n_style, n_samples as int32, then the token ids
as int32, the style row as float32, and the expected waveform as float32.
"""
import argparse
import os
import struct

import numpy as np

from ref_runtime import RefRuntime
from voice_util import ROOT, load_voice, sample_tokens

OUT = os.path.join(ROOT, "tests", "_case.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=9)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ids = sample_tokens(args.tokens)
    style = np.ascontiguousarray(load_voice(args.voice, len(ids)),
                                 dtype=np.float32)
    wav = RefRuntime().run(ids, style, args.speed)["waveform"].reshape(-1)

    with open(args.out, "wb") as f:
        f.write(struct.pack("<iii", len(ids), style.size, wav.size))
        f.write(np.array(ids, dtype="<i4").tobytes())
        f.write(style.astype("<f4").tobytes())
        f.write(wav.astype("<f4").tobytes())

    print(f"{args.out}: {len(ids)} ids, {wav.size} samples "
          f"(peak {np.abs(wav).max():.4f})")


if __name__ == "__main__":
    main()
