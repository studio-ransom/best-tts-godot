"""Dev-only: find sustained tonal artefacts in a .wav.

Correlation and per-band level both missed the thing that was actually audible
in this project -- a steady tone at 4800 Hz and 9600 Hz, ~25 dB above the
surrounding spectrum, left by Kokoro's iSTFT overlap-adding with hop 5. It hid
because it is narrow: it barely moves a 1000 Hz-wide band average and it is
present in the reference too, so any comparison against the reference cancels
it out.

    python tools/analyze_tone.py tests/out/godot.wav tests/out/reference.wav

Takes the *median* magnitude across frames, so only tones that persist through
the whole file survive; a vowel formant comes and goes and averages away.
Prominence is measured against the local spectrum rather than the overall
level, so it does not care how loud the clip is.
"""
import sys
import wave

import numpy as np

N_FFT = 8192
## Width of the neighbourhood a peak is judged against.
LOCAL_HZ = 400.0
## Above this, a peak is a tone rather than a formant.
PROMINENT_DB = 12.0


def load(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
        rate = w.getframerate()
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0, rate


def sustained_spectrum(x, rate, n=N_FFT):
    hop = n // 4
    win = np.hanning(n)
    frames = np.array([np.abs(np.fft.rfft(x[i:i + n] * win))
                       for i in range(0, max(1, len(x) - n), hop)])
    db = 20 * np.log10(np.maximum(np.median(frames, axis=0), 1e-14))
    return np.fft.rfftfreq(n, 1.0 / rate), db


def prominence(db, rate, n=N_FFT):
    k = max(3, int(LOCAL_HZ / (rate / n)))
    local = np.array([np.median(db[max(0, i - k):i + k + 1])
                      for i in range(len(db))])
    return db - local


def report(path):
    x, rate = load(path)
    freq, db = sustained_spectrum(x, rate)
    prom = prominence(db, rate)

    print(f"\n=== {path}  ({len(x) / rate:.2f} s, {rate} Hz) ===")
    print(f"  iSTFT hop rate is {rate / 5:.0f} Hz; a peak there and at its "
          f"octave is the overlap-add artefact")
    found = []
    for i in np.argsort(prom)[::-1]:
        if freq[i] < 40 or any(abs(freq[i] - g) < 200 for g in found):
            continue
        found.append(freq[i])
        flag = "  <-- tonal" if prom[i] > PROMINENT_DB else ""
        print(f"    {freq[i]:9.1f} Hz  {prom[i]:+6.1f} dB above local floor"
              f"{flag}")
        if len(found) >= 6:
            break

    hop_tone = max(prom[np.argmin(np.abs(freq - rate / 5))],
                   prom[np.argmin(np.abs(freq - 2 * rate / 5))])
    print(f"  worst hop-rate tone: {hop_tone:+.1f} dB "
          + ("(audible ring)" if hop_tone > PROMINENT_DB else "(clean)"))
    return hop_tone


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for p in sys.argv[1:]:
        report(p)
