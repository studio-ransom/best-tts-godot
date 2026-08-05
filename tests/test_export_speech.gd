extends SceneTree

## Synthesizes a sentence and writes the clip plus the exact token ids that
## produced it, so `tools/compare_speech.py` can run the same input through
## onnxruntime and compare the two waveforms.
##
##   godot --rendering-driver vulkan --script res://tests/test_export_speech.gd
##   python tools/compare_speech.py
##
## Optional: --text="..." --voice=af_heart

const GPUScript := preload("res://addons/best-tts/runtime/gpu.gd")
const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")
const ExecutorScript := preload("res://addons/best-tts/runtime/executor.gd")
const VoiceLoaderScript := preload("res://addons/best-tts/runtime/voice_loader.gd")
const TokenizerScript := preload("res://addons/best-tts/runtime/tokenizer.gd")
const G2PScript := preload("res://addons/best-tts/g2p/g2p.gd")

## Written into the repo, not user://, because the Python side reads it.
const OUT_DIR := "res://tests/out/"


func _init() -> void:
	var text := "Hello from Godot. This is Kokoro speaking."
	var voice := "af_heart"
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--text="):
			text = arg.substr(7)
		elif arg.begins_with("--voice="):
			voice = arg.substr(8)

	DirAccess.make_dir_recursive_absolute(OUT_DIR)

	var g2p = G2PScript.new()
	var tok = TokenizerScript.new()
	tok.load_vocab()
	var voices = VoiceLoaderScript.new()

	var ph: Dictionary = g2p.phonemize(text, voice)
	var phonemes: String = ph.get("phonemes", "")
	var enc: Dictionary = tok.encode(phonemes)
	var ids: PackedInt32Array = enc["ids"]
	var style: PackedFloat32Array = voices.style_for(voice,
			int(enc["phoneme_count"]))

	print("text:     \"%s\"" % text)
	print("voice:    %s" % voice)
	print("phonemes: %s" % phonemes)
	print("ids:      %d tokens" % ids.size())

	var gpu = GPUScript.new()
	var err: String = gpu.initialize()
	if err != "":
		printerr("FAIL: " + err)
		quit(1)
		return
	var model = KpkModelScript.new()
	if model.load_from("res://addons/best-tts/assets/model.kpk") != OK:
		printerr("FAIL: cannot load model.kpk")
		quit(1)
		return
	var ex = ExecutorScript.new()

	# Keep the durations so the Python side can compare them directly. A whole
	# frame of timing difference comes from a Round landing the other way, so
	# the pre-rounding value is the one that tells you whether the predictor
	# itself agrees.
	const TAPS := ["/encoder/Div_output_0", "/encoder/Clip_output_0"]
	var tap_tid := {}
	for i in model.tensor_names.size():
		if model.tensor_names[i] in TAPS:
			tap_tid[model.tensor_names[i]] = i
			ex.trace_tids[i] = true

	var setup_err: String = ex.setup(gpu, model)
	if setup_err != "":
		printerr("FAIL: " + setup_err)
		quit(1)
		return

	var t0 := Time.get_ticks_msec()
	var res: Dictionary = ex.run(ids, style, 1.0)
	var ms := Time.get_ticks_msec() - t0
	if res.is_empty():
		printerr("FAIL: " + ex.last_error)
		gpu.cleanup()
		quit(1)
		return
	var wave: PackedFloat32Array = gpu.read_f32(res["buffer"],
			int(res["samples"]))
	print("synthesized %d samples (%.2f s) in %d ms"
			% [wave.size(), wave.size() / 24000.0, ms])

	# Read the traced tensors while the device is still alive.
	var taps := {}
	for nm in tap_tid:
		var tid: int = tap_tid[nm]
		if ex.has_value(tid):
			taps[nm] = Array(ex.value_of(tid))
	gpu.cleanup()

	# Raw float32 waveform, for an exact numeric comparison.
	var wf := FileAccess.open(OUT_DIR + "godot.f32", FileAccess.WRITE)
	wf.store_buffer(wave.to_byte_array())
	wf.close()

	var meta := {
		"text": text,
		"voice": voice,
		"phonemes": phonemes,
		"ids": Array(ids),
		"speed": 1.0,
		"samples": wave.size(),
		"sample_rate": 24000,
		"synth_ms": ms,
		"taps": taps,
	}
	var mf := FileAccess.open(OUT_DIR + "godot.json", FileAccess.WRITE)
	mf.store_string(JSON.stringify(meta, "  "))
	mf.close()

	print("wrote %s{godot.f32,godot.json}"
			% ProjectSettings.globalize_path(OUT_DIR))
	print("now run: python tools/compare_speech.py")
	quit(0)
