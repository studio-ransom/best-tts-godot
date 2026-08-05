extends SceneTree

## End-to-end: plain text through the KokoroTTS node to a playable clip, and
## a .wav written out so the result can actually be listened to.
##
##   godot --rendering-driver vulkan --script res://tests/test_speak.gd
##
## Pass --text="..." and --voice=name to try your own line.

const TTSScript := preload("res://addons/best-tts/kokoro_tts.gd")

const OUT_DIR := "user://tts_out/"

var _tts
var _failures := 0


func _init() -> void:
	var text := "Hello from Godot. This is Kokoro, running entirely on your GPU."
	var voice := "af_heart"
	# --raw turns off the notch filter, so the .wav files are exactly what the
	# graph produced. Useful for an A/B listen, and it proves the tone check
	# below is a real gate rather than something that always passes.
	var raw := false
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--text="):
			text = arg.substr(7)
		elif arg.begins_with("--voice="):
			voice = arg.substr(8)
		elif arg == "--raw":
			raw = true

	DirAccess.make_dir_recursive_absolute(OUT_DIR)

	_tts = TTSScript.new()
	_tts.preload_on_ready = false
	_tts.remove_vocoder_tone = not raw
	root.add_child(_tts)

	_tts.engine_failed.connect(func(reason: String):
		printerr("FAIL: engine did not start — " + reason)
		quit(1))

	var t0 := Time.get_ticks_msec()
	_tts.start()
	await _tts.engine_ready
	print("engine ready in %d ms" % (Time.get_ticks_msec() - t0))

	# The phonemes are worth showing: a wrong-sounding clip is usually a G2P
	# problem, not a model problem.
	var ph: Dictionary = _tts.phonemize(text, voice)
	print("\ntext:     \"%s\"" % text)
	print("phonemes: %s" % ph.get("phonemes", ""))

	await _speak_case(text, voice, "speech")

	# A second call on the same engine — the common case in a real game, and
	# the one that would catch state leaking between runs.
	await _speak_case("The quick brown fox jumps over the lazy dog.",
			voice, "second")

	# A different voice, to prove the style pack is really being applied.
	var other := "bm_george" if voice.begins_with("a") else "af_heart"
	await _speak_case("The same sentence in a different voice.", other,
			"voice_" + other)

	# Long enough to be split into chunks and joined.
	var long_text := ""
	for i in 4:
		long_text += ("Sentence number %d in a longer passage that has to be "
				+ "split into chunks and stitched back together. ") % (i + 1)
	await _speak_case(long_text, voice, "long")

	# IPA straight in, bypassing G2P. This is the route for the languages the
	# built-in frontend does not cover, so check it against a voice that has
	# no G2P at all.
	var ipa := "kˈɑkɔɹO ɪz sˈɪnθəsˌIzɪŋ fɹʌm ˈIpˌiˈA."
	var ipa_t0 := Time.get_ticks_msec()
	var clip: AudioStreamWAV = await _tts.speak_phonemes(ipa, "jf_alpha")
	if clip == null:
		printerr("FAIL: speak_phonemes — %s" % _tts.get_error())
		_failures += 1
	else:
		var secs := clip.data.size() / 2.0 / clip.mix_rate
		print("\n[phonemes] %.2f s in %d ms, voice jf_alpha (no G2P needed)"
				% [secs, Time.get_ticks_msec() - ipa_t0])
		_write_wav(OUT_DIR + "phonemes.wav", clip)
		print("      wrote %sphonemes.wav" % OUT_DIR)

	print("\nwav files written to %s" % ProjectSettings.globalize_path(OUT_DIR))
	if _failures == 0:
		print("OK: text in, speech out")
	quit(1 if _failures > 0 else 0)


func _speak_case(text: String, voice: String, label: String) -> void:
	var t0 := Time.get_ticks_msec()
	var clip: AudioStreamWAV = await _tts.speak(text, voice)
	var ms := Time.get_ticks_msec() - t0
	if clip == null:
		printerr("FAIL: %s — %s" % [label, _tts.get_error()])
		_failures += 1
		return

	var samples := clip.data.size() / 2
	var seconds := float(samples) / clip.mix_rate
	var peak := 0
	var energy := 0.0
	for i in samples:
		var s := clip.data.decode_s16(i * 2)
		peak = maxi(peak, absi(s))
		energy += float(s) * s
	var rms := sqrt(energy / maxi(1, samples))

	print("\n[%s] %.2f s of audio in %d ms (%.1fx realtime)"
			% [label, seconds, ms, seconds * 1000.0 / maxf(1.0, ms)])
	print("      peak %d, rms %.0f, %d Hz" % [peak, rms, clip.mix_rate])

	# Silence and clipping are the two ways this fails while still "working".
	if peak < 1000:
		printerr("FAIL: %s is silent (peak %d)" % [label, peak])
		_failures += 1
	elif peak >= 32767:
		printerr("FAIL: %s is clipping" % label)
		_failures += 1
	elif rms < 200.0:
		printerr("FAIL: %s is nearly silent (rms %.0f)" % [label, rms])
		_failures += 1
	else:
		_check_vocoder_tone(label, clip)
		var path := OUT_DIR + label + ".wav"
		_write_wav(path, clip)
		print("      wrote %s" % path)


## Kokoro's iSTFT leaves a tone at the hop rate, 24000/5 = 4800 Hz, and at its
## octave. Unfiltered they sit ~25 dB above the neighbouring spectrum and are
## clearly audible as a ring; `remove_vocoder_tone` notches them out. Measure
## each one against its own neighbourhood so the check does not depend on the
## overall level or on what is being said.
func _check_vocoder_tone(label: String, clip: AudioStreamWAV) -> void:
	# Two seconds is plenty for a steady tone, and keeps a GDScript loop off
	# the critical path for a 25-second clip.
	var n := mini(clip.data.size() / 2, 2 * clip.mix_rate)
	var x := PackedFloat32Array()
	x.resize(n)
	for i in n:
		x[i] = clip.data.decode_s16(i * 2) / 32768.0

	for f0 in [4800.0, 9600.0]:
		var here := _goertzel_db(x, f0, clip.mix_rate)
		var near := (_goertzel_db(x, f0 - 400.0, clip.mix_rate)
				+ _goertzel_db(x, f0 + 400.0, clip.mix_rate)) * 0.5
		var excess := here - near
		print("      %d Hz sits %+.1f dB vs its neighbours" % [int(f0), excess])
		if excess > 10.0:
			printerr("FAIL: %s rings at %d Hz (%+.1f dB above neighbours)"
					% [label, int(f0), excess])
			_failures += 1


## Power at one frequency, in dB. Goertzel rather than a full FFT: a handful of
## bins is all this needs and it is a five-line recurrence.
func _goertzel_db(x: PackedFloat32Array, freq: float, rate: int) -> float:
	if x.is_empty():
		return -240.0
	var coeff := 2.0 * cos(TAU * freq / rate)
	var s1 := 0.0
	var s2 := 0.0
	for v in x:
		var s: float = v + coeff * s1 - s2
		s2 = s1
		s1 = s
	var power := s1 * s1 + s2 * s2 - coeff * s1 * s2
	return 10.0 * log(maxf(power, 1e-20)) / log(10.0)


## AudioStreamWAV holds raw PCM, so a real .wav needs the 44-byte header.
func _write_wav(path: String, clip: AudioStreamWAV) -> void:
	var pcm := clip.data
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		printerr("could not write %s" % path)
		return
	f.store_buffer("RIFF".to_ascii_buffer())
	f.store_32(36 + pcm.size())
	f.store_buffer("WAVEfmt ".to_ascii_buffer())
	f.store_32(16)                      # PCM header size
	f.store_16(1)                       # format: PCM
	f.store_16(1)                       # mono
	f.store_32(clip.mix_rate)
	f.store_32(clip.mix_rate * 2)       # byte rate
	f.store_16(2)                       # block align
	f.store_16(16)                      # bits per sample
	f.store_buffer("data".to_ascii_buffer())
	f.store_32(pcm.size())
	f.store_buffer(pcm)
	f.close()
