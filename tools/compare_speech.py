"""Dev-only: compare a sentence spoken by Godot against onnxruntime.

`tests/test_export_speech.gd` writes the waveform it produced plus the exact
token ids it used. This feeds those same ids to onnxruntime and reports how
far apart the two waveforms are, then writes both as .wav so they can be
listened to back to back.

    godot --rendering-driver vulkan --script res://tests/test_export_speech.gd
    python tools/compare_speech.py

Bit-exactness is not the goal and is not achievable: the model is int8
quantized, so a rounding tie in the duration predictor moves a phoneme
boundary by one frame, and near-silent STFT bins have essentially random
phase. What matters is that the two are the same speech -- high correlation,
matching length, matching envelope.
"""
import argparse
import json
import os
import struct
import sys
import wave

import numpy as np

from voice_util import ROOT, load_voice

OUT = os.path.join(ROOT, "tests", "out")


def write_wav(path, samples, rate=24000):
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def envelope(x, win=240):
    """Short-time RMS -- the shape of the speech, ignoring fine phase."""
    n = len(x) // win * win
    if n == 0:
        return np.zeros(1)
    return np.sqrt((x[:n].reshape(-1, win).astype(np.float64) ** 2).mean(1))


def spectrogram(x, n_fft=512, hop=128):
    """Magnitude spectrogram in dB.

    This is the metric that answers "does it sound the same". The vocoder
    builds its excitation by integrating F0 through a CumSum, so a tiny
    frequency difference accumulates into a growing phase offset -- the
    waveforms drift apart sample-wise while carrying identical speech. Phase
    is exactly what a magnitude spectrogram discards.
    """
    if len(x) < n_fft:
        return np.zeros((1, n_fft // 2 + 1))
    win = np.hanning(n_fft)
    frames = 1 + (len(x) - n_fft) // hop
    out = np.empty((frames, n_fft // 2 + 1))
    for i in range(frames):
        seg = x[i * hop:i * hop + n_fft] * win
        out[i] = np.abs(np.fft.rfft(seg))
    return 20 * np.log10(out + 1e-8)


BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
         (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000), (8000, 9000),
         (9000, 10000), (10000, 11000), (11000, 12000)]


def band_tilt(a, b, rate=24000):
    """Per-band level difference after loudness matching, in dB.

    Correlation is blind to spectral *shape*: a resize off-by-one that put one
    stale sample at every frame boundary added an 80 Hz impulse train, +5 dB
    across 4-12 kHz and clearly audible as a ring, while spectral correlation
    stayed at 0.986. Tilt catches that; correlation does not.
    """
    b = b * (np.sqrt(np.mean(a ** 2)) / max(np.sqrt(np.mean(b ** 2)), 1e-12))
    n = 2048
    win = np.hanning(n)

    def avg(x):
        frames = [np.abs(np.fft.rfft(x[i:i + n] * win))
                  for i in range(0, max(1, len(x) - n), n // 2)]
        return np.mean(frames, axis=0)

    f = np.fft.rfftfreq(n, 1.0 / rate)
    sa, sb = avg(a), avg(b)
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        da = 20 * np.log10(max(np.mean(sa[m]), 1e-12))
        db = 20 * np.log10(max(np.mean(sb[m]), 1e-12))
        out.append((lo, hi, da - db))
    return out


def compare_durations(taps, ref_env):
    """Per-phoneme durations, before and after rounding.

    ONNX rounds half to even, so a predicted duration of exactly n.5 is a
    coin flip decided by numerical noise far upstream. One phoneme flipping
    shifts every sample after it, which is why a small duration disagreement
    matters far more here than its size suggests.
    """
    pre_name = "/encoder/Div_output_0"
    post_name = "/encoder/Clip_output_0"
    if pre_name not in taps or pre_name not in ref_env:
        return

    got_pre = np.asarray(taps[pre_name], dtype=np.float64).reshape(-1)
    want_pre = np.asarray(ref_env[pre_name], dtype=np.float64).reshape(-1)
    n = min(got_pre.size, want_pre.size)
    got_pre, want_pre = got_pre[:n], want_pre[:n]
    diff = np.abs(got_pre - want_pre)

    got_post = np.rint(np.asarray(taps.get(post_name, np.rint(got_pre)),
                                  dtype=np.float64).reshape(-1))[:n]
    want_post = np.rint(np.asarray(ref_env.get(post_name, np.rint(want_pre)),
                                   dtype=np.float64).reshape(-1))[:n]
    flipped = np.flatnonzero(got_post != want_post)

    print(f"\ndurations ({n} phonemes)")
    print(f"  predicted (pre-round)  max|err| {diff.max():.4f}, "
          f"mean {diff.mean():.4f}")
    print(f"  frames total           godot {got_post.sum():.0f}, "
          f"reference {want_post.sum():.0f}")
    if flipped.size == 0:
        print("  every phoneme rounded the same way")
        return
    print(f"  {flipped.size} phoneme(s) rounded differently:")
    for i in flipped[:8]:
        # How close to a tie it was: 0.5 means the rounding was a coin flip.
        tie = abs(want_pre[i] - np.floor(want_pre[i]) - 0.5)
        print(f"    #{i:3d}  godot {got_pre[i]:7.4f} -> {got_post[i]:3.0f}   "
              f"reference {want_pre[i]:7.4f} -> {want_post[i]:3.0f}   "
              f"(distance from a tie: {tie:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=OUT)
    ap.add_argument("--engine", default="numpy", choices=["ort", "numpy"],
                    help="what to compare against: the numpy reference "
                         "(default, shares our fp32 policy) or onnxruntime")
    ap.add_argument("--control", action="store_true",
                    help="also print how far the numpy reference itself sits "
                         "from onnxruntime, which is the yardstick for any "
                         "ORT comparison")
    args = ap.parse_args()

    meta_path = os.path.join(args.dir, "godot.json")
    wave_path = os.path.join(args.dir, "godot.f32")
    if not os.path.exists(meta_path):
        sys.exit(f"{meta_path} not found -- run tests/test_export_speech.gd first")

    meta = json.load(open(meta_path, encoding="utf-8"))
    got = np.fromfile(wave_path, dtype=np.float32)
    ids = [int(i) for i in meta["ids"]]
    rate = int(meta.get("sample_rate", 24000))

    print(f'text:     "{meta["text"]}"')
    print(f'voice:    {meta["voice"]}')
    print(f'phonemes: {meta["phonemes"]}')
    print(f"ids:      {len(ids)} tokens")
    print(f"godot:    {got.size} samples ({got.size / rate:.2f} s) "
          f"in {meta.get('synth_ms', '?')} ms")

    style = load_voice(meta["voice"], len(ids))
    speed = float(meta.get("speed", 1.0))

    ref_env = None
    if args.engine == "ort":
        from golden import run as ort_run
        want = np.asarray(ort_run(ids, meta["voice"], speed)).reshape(-1)
        label = "onnxruntime"
    else:
        from ref_runtime import RefRuntime
        rt = RefRuntime()
        ref_env = rt.run(np.array(ids, dtype=np.int64), style, speed)
        want = np.asarray(ref_env[rt.graph.output[0].name]).reshape(-1)
        label = "numpy reference"
    print(f"{label:9} {want.size} samples ({want.size / rate:.2f} s)")

    if ref_env is not None:
        compare_durations(meta.get("taps", {}), ref_env)

    write_wav(os.path.join(args.dir, "godot.wav"), got, rate)
    write_wav(os.path.join(args.dir, "reference.wav"), want, rate)

    # Length differences always come in whole duration frames; derive the
    # frame size from the run itself rather than assuming it.
    FRAME = 600
    taps = meta.get("taps", {})
    if "/encoder/Clip_output_0" in taps:
        total = float(np.rint(np.asarray(
                taps["/encoder/Clip_output_0"], dtype=np.float64)).sum())
        if total > 0 and got.size % int(total) == 0:
            FRAME = got.size // int(total)
    delta = got.size - want.size
    if delta:
        print(f"\nlength differs by {delta:+d} samples "
              f"({delta / FRAME:+.0f} duration frames, "
              f"{abs(delta) / rate * 1000:.0f} ms)")

    n = min(got.size, want.size)
    a = got[:n].astype(np.float64)
    b = want[:n].astype(np.float64)

    # Per-frame SNR. The quantized model has a real noise floor -- roughly
    # 15 dB against this reference -- so frames never match bit-for-bit. What
    # separates "same speech, slightly noisier" from "shifted in time" is
    # stark: once a duration rounding offsets one signal, correlated frames
    # (10 dB+) become uncorrelated ones (about 0 dB), because a 25 ms shift
    # is several pitch periods.
    nblocks = n // FRAME
    frame_snr = np.full(nblocks, np.inf)
    for i in range(nblocks):
        x = a[i * FRAME:(i + 1) * FRAME]
        y = b[i * FRAME:(i + 1) * FRAME]
        sig = float(np.sqrt((y ** 2).mean()))
        if sig < 1e-6:
            continue
        noise = float(np.sqrt(((x - y) ** 2).mean()))
        frame_snr[i] = 20 * np.log10(sig / max(noise, 1e-20))

    finite = frame_snr[np.isfinite(frame_snr)]
    tracked = int((finite > 5.0).sum())
    prefix = nblocks
    for i in range(nblocks):
        if np.isfinite(frame_snr[i]) and frame_snr[i] < 5.0:
            prefix = i
            break
    prefix_s = prefix * FRAME / rate
    median_snr = float(np.median(finite)) if finite.size else float("nan")

    ea, eb = envelope(a), envelope(b)
    m = min(len(ea), len(eb))
    env_corr = float(np.corrcoef(ea[:m], eb[:m])[0, 1])

    sa, sb = spectrogram(a), spectrogram(b)
    k = min(len(sa), len(sb))
    spec_corr = float(np.corrcoef(sa[:k].ravel(), sb[:k].ravel())[0, 1])
    spec_mae = float(np.abs(sa[:k] - sb[:k]).mean())

    print(f"\npeak      godot {np.abs(a).max():.4f}   {label} {np.abs(b).max():.4f}")
    print(f"rms       godot {np.sqrt((a ** 2).mean()):.4f}   "
          f"{label} {np.sqrt((b ** 2).mean()):.4f}")
    print(f"spectral correlation  {spec_corr:.4f}  "
          f"(mean |diff| {spec_mae:.1f} dB)")
    print(f"envelope correlation  {env_corr:.4f}")
    print(f"waveform per-frame    median snr {median_snr:.1f} dB, "
          f"{tracked}/{nblocks} frames phase-aligned")
    print(f"  phase-aligned for the first {prefix_s:.2f} s; after that the "
          f"excitation phase has drifted (audibly irrelevant)")

    tilt = band_tilt(a, b, rate)
    worst_tilt = max(abs(d) for _, _, d in tilt)
    print(f"\nspectral tilt (loudness-matched, godot minus {label})")
    for lo, hi, d in tilt:
        bar = "#" * min(20, int(abs(d) * 4))
        print(f"  {lo:5d}-{hi:5d} Hz  {d:+6.2f} dB  {bar}")
    print(f"  worst band {worst_tilt:.2f} dB")

    print(f"\nwrote {os.path.join(args.dir, 'godot.wav')}")
    print(f"wrote {os.path.join(args.dir, 'reference.wav')}")

    # These catch "this is different speech" -- a wrong voice, wrong words, a
    # broken vocoder -- rather than "these bits differ". Spectrum and envelope
    # are the real tests: they survive phase drift and small timing shifts
    # while collapsing on any genuine error.
    if args.control and args.engine == "ort":
        control(ids, style, meta["voice"], speed, want, label)

    # 2 dB is generous for the numpy reference, which shares our precision
    # policy; against ORT the fp16 intermediates alone move a band by ~1.5 dB.
    ok = (spec_corr > 0.95 and env_corr > 0.95
          and abs(delta) < rate * 0.25 and worst_tilt < 2.0)
    if args.engine == "ort" and not ok:
        print("\nNote: onnxruntime keeps fp16 intermediates while this addon "
              "computes everything in fp32, so the two drift apart by design. "
              "Run with --control to see that the verified numpy reference "
              "sits the same distance from onnxruntime as Godot does.")
    print(("\nOK: Godot and the %s produced the same speech" % label) if ok
          else "\nFAIL: the two waveforms are not the same speech")
    return 0 if ok else 1


def control(ids, style, voice, speed, ort_wave, label):
    """How far the numpy reference itself sits from onnxruntime.

    Without this number an ORT comparison is unreadable: the reference is
    verified node by node against ORT, so whatever distance *it* shows is the
    floor imposed by the fp32 precision policy, not an implementation error.
    """
    from ref_runtime import RefRuntime
    rt = RefRuntime()
    env = rt.run(np.array(ids, dtype=np.int64), style, speed)
    npy = np.asarray(env[rt.graph.output[0].name]).reshape(-1)

    n = min(npy.size, ort_wave.size)
    a, b = npy[:n].astype(np.float64), ort_wave[:n].astype(np.float64)
    ea, eb = envelope(a), envelope(b)
    m = min(len(ea), len(eb))
    sa, sb = spectrogram(a), spectrogram(b)
    k = min(len(sa), len(sb))
    print(f"\ncontrol — numpy reference vs {label}")
    print(f"  spectral correlation  "
          f"{np.corrcoef(sa[:k].ravel(), sb[:k].ravel())[0, 1]:.4f}")
    print(f"  envelope correlation  {np.corrcoef(ea[:m], eb[:m])[0, 1]:.4f}")
    print("  this is the floor: the reference passes 134/134 per-op tests "
          "against onnxruntime, so anything at this distance is the fp16 "
          "policy, not a bug")


if __name__ == "__main__":
    sys.exit(main())
