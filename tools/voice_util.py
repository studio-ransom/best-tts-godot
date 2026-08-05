"""Dev-only helpers shared by the reference/compare tooling.

Mirrors exactly what the GDScript runtime will do, so any mismatch is a real
bug rather than a difference in how the two sides prepared their inputs.
"""
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VOICES = os.path.join(ROOT, "addons", "best-tts", "assets", "voices")
CONFIG = os.path.join(HERE, "_kokoro_config.json")

_vocab = None


def vocab():
    """phoneme character -> token id (114 entries, ids up to 177)."""
    global _vocab
    if _vocab is None:
        with open(CONFIG, encoding="utf-8") as f:
            _vocab = json.load(f)["vocab"]
    return _vocab


def phonemes_to_ids(ps):
    v = vocab()
    return [v[c] for c in ps if c in v]


def load_voice_pack(name):
    """Read a torch .pt voice into a [510, 256] float32 array.

    The file is a torch zip whose single storage entry ``<name>/data/0`` holds
    510*1*256 little-endian float32 values; no real unpickling is required.
    """
    path = os.path.join(VOICES, name + ".pt")
    with zipfile.ZipFile(path) as z:
        entry = next(n for n in z.namelist() if n.endswith("/data/0"))
        raw = z.read(entry)
    arr = np.frombuffer(raw, dtype="<f4")
    assert arr.size == 510 * 256, f"unexpected voice size {arr.size}"
    return arr.reshape(510, 256).copy()


def load_voice(name, n_input_ids):
    """Style row for a token sequence, matching Kokoro's KPipeline indexing.

    ``input_ids`` is ``[0, *phonemes, 0]`` and KPipeline uses
    ``pack[len(phonemes) - 1]``.
    """
    pack = load_voice_pack(name)
    idx = max(0, min(509, n_input_ids - 3))
    return pack[idx]


def sample_tokens(n=9):
    """A deterministic phoneme id sequence of length n (incl. both 0 pads)."""
    # "hˈɛlOʊ wˈɜɹld" style filler drawn from the real vocab
    body = [50, 156, 86, 54, 57, 135, 16, 65, 156, 87, 123, 54, 46, 4]
    body = (body * ((n // len(body)) + 1))[: max(0, n - 2)]
    return [0] + body + [0]
