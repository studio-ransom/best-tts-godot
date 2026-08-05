@tool
class_name KokoroAudio
extends RefCounted

## Turns a waveform living in a GPU buffer into a playable Godot audio stream.
##
## The float-to-PCM16 conversion happens in `kernels/pcm16.glsl`, so the bytes
## read back from the device are already the stream's payload.

const SAMPLE_RATE := 24000

## Kokoro's iSTFT overlaps with hop 5, so its window ripple lands at
## 24000/5 = 4800 Hz and again at 9600 Hz. Both show up ~25 dB above the local
## noise floor as a steady ring behind the speech.
const TONE_HZ := [4800.0, 9600.0]
## Notch half-width. 80 Hz is wide enough that a 1023-tap window reaches a
## -50 dB null, and narrow enough to take 0.06 dB off the overall level.
const TONE_BW := 80.0
## Odd, so the filter has an exact centre tap and therefore zero phase.
const TONE_TAPS := 1023


## Coefficients for a linear-phase filter that is flat everywhere except for a
## narrow null at each of `tones`.
##
## Built as an impulse minus one windowed-sinc band-pass per tone, which is the
## short way of saying "pass everything, then subtract the parts we do not
## want". Blackman rather than Hamming: the extra stopband depth is what turns
## a 3 dB reduction into a 50 dB one.
static func notch_taps(sample_rate: int, tones := TONE_HZ, bw := TONE_BW,
		n := TONE_TAPS) -> PackedFloat32Array:
	var taps := PackedFloat32Array()
	taps.resize(n)
	var c := (n - 1) / 2
	taps[c] = 1.0
	for f0 in tones:
		var f1: float = (f0 - bw) / sample_rate
		var f2: float = (f0 + bw) / sample_rate
		for k in n:
			var m := float(k - c)
			var w := 0.42 - 0.5 * cos(TAU * k / (n - 1)) \
					+ 0.08 * cos(2.0 * TAU * k / (n - 1))
			taps[k] -= (2.0 * f2 * _sinc(2.0 * f2 * m)
					- 2.0 * f1 * _sinc(2.0 * f1 * m)) * w
	return taps


static func _sinc(x: float) -> float:
	if absf(x) < 1e-9:
		return 1.0
	return sin(PI * x) / (PI * x)


## Converts `sample_count` floats in `wave_buffer` to an AudioStreamWAV.
##
## `gpu` must already be initialized; the pcm16 kernel is loaded on demand.
## Pass `fir_taps` (a buffer of `fir_len` coefficients, see `notch_taps`) to
## filter the waveform on the way out.
static func stream_from_gpu(gpu, wave_buffer: RID, sample_count: int,
		sample_rate := SAMPLE_RATE, gain := 1.0,
		fir_taps := RID(), fir_len := 0) -> AudioStreamWAV:
	if sample_count <= 0:
		return null
	if not gpu.load_kernel("pcm16"):
		return null

	var source := wave_buffer
	var filtered := RID()
	if fir_taps.is_valid() and fir_len > 0 and gpu.load_kernel("fir"):
		filtered = gpu.create_buffer(sample_count * 4)
		var fp := PackedByteArray()
		fp.resize(16)
		fp.encode_u32(0, sample_count)
		fp.encode_u32(4, fir_len)
		fp.encode_u32(8, (fir_len - 1) / 2)
		gpu.begin_compute()
		gpu.dispatch("fir", [wave_buffer, fir_taps, filtered] as Array[RID], fp,
				(sample_count + 63) / 64)
		gpu.end_compute()
		gpu.submit_and_wait()
		source = filtered

	var pairs := (sample_count + 1) / 2
	var out_buf: RID = gpu.create_buffer(pairs * 4)

	var push := PackedByteArray()
	push.resize(16)
	push.encode_u32(0, sample_count)
	push.encode_u32(4, pairs)
	push.encode_float(8, gain)
	push.encode_u32(12, 0)

	gpu.begin_compute()
	gpu.dispatch("pcm16", [source, out_buf] as Array[RID], push,
			(pairs + 63) / 64)
	gpu.end_compute()
	gpu.submit_and_wait()

	var bytes: PackedByteArray = gpu.rd.buffer_get_data(out_buf)
	gpu.free_buffer(out_buf)
	if filtered.is_valid():
		gpu.free_buffer(filtered)

	# The last pair may carry a padding sample beyond the real length.
	bytes.resize(sample_count * 2)
	return from_pcm16(bytes, sample_rate)


## Perceived loudness of a clip, in LUFS, plus its sample peak.
##
## ITU-R BS.1770: K-weight, then take the mean square over 400 ms blocks and
## average only the blocks that are actually speech. The gating is the part
## that matters here — plain RMS over a whole clip measures how much of it was
## silence, so a voice that pauses between clauses reads quieter than an
## identically loud voice that does not, and a per-voice gain built on that
## would be wrong in proportion to the speaking rate.
##
## Returns `{lufs, peak}`. `lufs` is -INF for a clip with no speech in it.
static func measure_loudness(clip: AudioStreamWAV) -> Dictionary:
	var n := clip.data.size() / 2
	if n == 0:
		return {"lufs": -INF, "peak": 0.0}
	var x := PackedFloat32Array()
	x.resize(n)
	var peak := 0.0
	for i in n:
		var v := clip.data.decode_s16(i * 2) / 32768.0
		x[i] = v
		peak = maxf(peak, absf(v))

	var rate := float(clip.mix_rate)
	_biquad(x, _shelf_coeffs(rate))
	_biquad(x, _highpass_coeffs(rate))

	# 400 ms blocks, 75% overlap, as the spec specifies.
	var block := int(0.4 * rate)
	var step := block / 4
	if n < block:
		return {"lufs": _loudness_of(_mean_square(x, 0, n)), "peak": peak}

	var powers := PackedFloat64Array()
	var i := 0
	while i + block <= n:
		powers.append(_mean_square(x, i, i + block))
		i += step

	# Absolute gate at -70 LUFS, then a relative gate 10 LU below the mean of
	# what survived it. Two passes, because the relative threshold depends on
	# the blocks it is about to select.
	var sum := 0.0
	var count := 0
	for p in powers:
		if _loudness_of(p) > -70.0:
			sum += p
			count += 1
	if count == 0:
		return {"lufs": -INF, "peak": peak}
	var relative := _loudness_of(sum / count) - 10.0

	sum = 0.0
	count = 0
	for p in powers:
		var l := _loudness_of(p)
		if l > -70.0 and l > relative:
			sum += p
			count += 1
	if count == 0:
		return {"lufs": -INF, "peak": peak}
	return {"lufs": _loudness_of(sum / count), "peak": peak}


static func _loudness_of(mean_square: float) -> float:
	if mean_square <= 0.0:
		return -INF
	return -0.691 + 10.0 * log(mean_square) / log(10.0)


static func _mean_square(x: PackedFloat32Array, from: int, to: int) -> float:
	var sum := 0.0
	for i in range(from, to):
		sum += float(x[i]) * x[i]
	return sum / maxf(1.0, float(to - from))


## K-weighting stage 1: a +4 dB high shelf at 1681 Hz, standing in for the
## acoustic effect of a head. Designed for the actual sample rate rather than
## using the published 48 kHz coefficients, which are wrong at 24 kHz.
static func _shelf_coeffs(rate: float) -> Array:
	var k := tan(PI * 1681.974450955533 / rate)
	var q := 0.7071752369554196
	var vh := pow(10.0, 3.999843853973347 / 20.0)
	var vb := pow(vh, 0.4996667741545416)
	var d := 1.0 + k / q + k * k
	return [
		(vh + vb * k / q + k * k) / d,
		2.0 * (k * k - vh) / d,
		(vh - vb * k / q + k * k) / d,
		2.0 * (k * k - 1.0) / d,
		(1.0 - k / q + k * k) / d,
	]


## K-weighting stage 2: the RLB high pass at 38 Hz, which discards rumble the
## ear does not weigh.
static func _highpass_coeffs(rate: float) -> Array:
	var k := tan(PI * 38.13547087602444 / rate)
	var q := 0.5003270373238773
	var d := 1.0 + k / q + k * k
	return [1.0, -2.0, 1.0, 2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d]


## Direct form I, in place. `c` is [b0, b1, b2, a1, a2].
static func _biquad(x: PackedFloat32Array, c: Array) -> void:
	var x1 := 0.0
	var x2 := 0.0
	var y1 := 0.0
	var y2 := 0.0
	for i in x.size():
		var xn := float(x[i])
		var yn: float = c[0] * xn + c[1] * x1 + c[2] * x2 - c[3] * y1 - c[4] * y2
		x2 = x1
		x1 = xn
		y2 = y1
		y1 = yn
		x[i] = yn


## Wraps raw little-endian 16-bit mono PCM in an AudioStreamWAV.
static func from_pcm16(bytes: PackedByteArray,
		sample_rate := SAMPLE_RATE) -> AudioStreamWAV:
	var stream := AudioStreamWAV.new()
	stream.format = AudioStreamWAV.FORMAT_16_BITS
	stream.stereo = false
	stream.mix_rate = sample_rate
	stream.data = bytes
	return stream


## CPU fallback, for tests and for waveforms that never reached the GPU.
static func stream_from_floats(samples: PackedFloat32Array,
		sample_rate := SAMPLE_RATE, gain := 1.0) -> AudioStreamWAV:
	var bytes := PackedByteArray()
	bytes.resize(samples.size() * 2)
	for i in samples.size():
		var v := clampf(samples[i] * gain, -1.0, 1.0)
		bytes.encode_s16(i * 2, roundi(v * 32767.0))
	return from_pcm16(bytes, sample_rate)


## Joins several clips end to end, for chunked synthesis.
static func concatenate(streams: Array) -> AudioStreamWAV:
	var bytes := PackedByteArray()
	var rate := SAMPLE_RATE
	for s in streams:
		if s == null:
			continue
		rate = s.mix_rate
		bytes.append_array(s.data)
	return from_pcm16(bytes, rate)
