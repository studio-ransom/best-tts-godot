extends Control

## Interactive bench for the best-tts addon.
##
## Exercises every capability that is currently wired up and states plainly
## which ones are not, so the scene doubles as a progress report:
##
##   * packed model loading
##   * voice pack loading and style-row selection
##   * phoneme tokenization
##   * GPU device, kernel compilation and elementwise correctness
##   * the GPU -> PCM16 -> AudioStreamWAV output stage, audible
##
## Requires the Forward+ or Mobile renderer; Compatibility has no compute.

const GPUScript := preload("res://addons/best-tts/runtime/gpu.gd")
const Ops := preload("res://addons/best-tts/runtime/ops.gd")
const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")
const VoiceLoader := preload("res://addons/best-tts/runtime/voice_loader.gd")
const Tokenizer := preload("res://addons/best-tts/runtime/tokenizer.gd")
const KokoroAudioScript := preload("res://addons/best-tts/runtime/audio.gd")
const TTSScript := preload("res://addons/best-tts/kokoro_tts.gd")

const MODEL_PATH := "res://addons/best-tts/assets/model.kpk"

## "hˈɛlō wˈɜɹld" — real IPA the model's vocabulary accepts.
const SAMPLE_PHONEMES := "hˈɛlOʊ wˈɜɹld, ðɪs ɪz kˈOʊkɔɹO ɹˈʌnɪŋ ɪn ɡˈOʊdɑt."

var gpu
var model
var voices := VoiceLoader.new()
var tokenizer := Tokenizer.new()
var player := AudioStreamPlayer.new()

var _log_box: RichTextLabel
var _phoneme_input: TextEdit
var _text_input: TextEdit
var _voice_picker: OptionButton
var _status: Label
var _synth_status: Label
var _last_stream: AudioStreamWAV
var _failures := 0

## The real TTS engine, created on first use — it owns its own GPU device and
## takes a couple of seconds to load the model. Untyped, because the global
## class name only exists after Godot has scanned the addon.
var _tts
var _speaking := false


func _ready() -> void:
	add_child(player)
	_build_ui()
	_check_system()

	# `godot --rendering-driver vulkan --autorun` runs the whole bench and
	# exits with a non-zero code if anything failed, so CI can use this scene.
	if "--autorun" in OS.get_cmdline_args():
		await get_tree().process_frame
		await _run_all()
		await get_tree().create_timer(1.5).timeout
		get_tree().quit(1 if _failures > 0 else 0)


# ---------------------------------------------------------------- ui scaffold

func _build_ui() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)

	var root := MarginContainer.new()
	root.set_anchors_preset(Control.PRESET_FULL_RECT)
	root.add_theme_constant_override("margin_left", 16)
	root.add_theme_constant_override("margin_right", 16)
	root.add_theme_constant_override("margin_top", 12)
	root.add_theme_constant_override("margin_bottom", 12)
	add_child(root)

	var split := VBoxContainer.new()
	split.add_theme_constant_override("separation", 10)
	root.add_child(split)

	var title := Label.new()
	title.text = "best-tts — Kokoro-82M capability bench"
	title.add_theme_font_size_override("font_size", 22)
	split.add_child(title)

	_status = Label.new()
	_status.text = "checking system..."
	_status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	split.add_child(_status)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	split.add_child(scroll)

	var col := VBoxContainer.new()
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	col.add_theme_constant_override("separation", 6)
	scroll.add_child(col)

	_build_model_section(col)
	_build_voice_section(col)
	_build_tokenizer_section(col)
	_build_gpu_section(col)
	_build_audio_section(col)
	_build_synthesis_section(col)

	var log_label := Label.new()
	log_label.text = "Log"
	log_label.add_theme_font_size_override("font_size", 16)
	split.add_child(log_label)

	_log_box = RichTextLabel.new()
	_log_box.bbcode_enabled = true
	_log_box.scroll_following = true
	_log_box.custom_minimum_size = Vector2(0, 220)
	_log_box.add_theme_font_override("normal_font",
			ThemeDB.fallback_font)
	split.add_child(_log_box)

	var bottom := HBoxContainer.new()
	split.add_child(bottom)
	bottom.add_child(_button("Run every test", _run_all))
	bottom.add_child(_button("Clear log", func(): _log_box.clear()))


func _section(parent: Control, heading: String, blurb := "") -> VBoxContainer:
	parent.add_child(HSeparator.new())
	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 4)
	parent.add_child(box)
	var h := Label.new()
	h.text = heading
	h.add_theme_font_size_override("font_size", 16)
	box.add_child(h)
	if blurb != "":
		var b := Label.new()
		b.text = blurb
		b.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		b.modulate = Color(1, 1, 1, 0.65)
		box.add_child(b)
	return box


func _button(text: String, cb: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.pressed.connect(cb)
	return b


func _row(parent: Control) -> HBoxContainer:
	var h := HBoxContainer.new()
	h.add_theme_constant_override("separation", 6)
	parent.add_child(h)
	return h


func _log(msg: String) -> void:
	_log_box.append_text(msg + "\n")
	# The console gets the same text without the markup the RichTextLabel uses.
	print(_strip_bbcode(msg))


static func _strip_bbcode(s: String) -> String:
	var re := RegEx.new()
	re.compile("\\[/?[a-zA-Z][^\\]]*\\]")
	return re.sub(s, "", true)


func _ok(msg: String) -> void:
	_log("[color=#7ee787]PASS[/color]  " + msg)


func _fail(msg: String) -> void:
	_failures += 1
	_log("[color=#ff7b72]FAIL[/color]  " + msg)


func _info(msg: String) -> void:
	_log("[color=#79c0ff]  ..[/color]  " + msg)


func _todo(msg: String) -> void:
	_log("[color=#d29922]TODO[/color]  " + msg)


# ------------------------------------------------------------------- sections

func _build_model_section(col: Control) -> void:
	var box := _section(col, "1 · Packed model",
			"Loads assets/model.kpk — the whole network as one file: a JSON "
			+ "graph plus a flat weight blob. No ONNX parsing at runtime.")
	_row(box).add_child(_button("Load model.kpk", _test_model))


func _build_voice_section(col: Control) -> void:
	var box := _section(col, "2 · Voice packs",
			"Reads a PyTorch .pt voice with ZIPReader and picks the style row "
			+ "for the utterance length.")
	var r := _row(box)
	_voice_picker = OptionButton.new()
	for v in VoiceLoader.list_voices():
		_voice_picker.add_item("%s  (%s)" % [v, VoiceLoader.describe(v)])
		_voice_picker.set_item_metadata(_voice_picker.item_count - 1, v)
	if _voice_picker.item_count == 0:
		_voice_picker.add_item("no voices found")
	r.add_child(_voice_picker)
	r.add_child(_button("Load voice", _test_voice))
	r.add_child(_button("Load all 45", _test_all_voices))


func _build_tokenizer_section(col: Control) -> void:
	var box := _section(col, "3 · Tokenizer",
			"Maps IPA phonemes to the model's 114-symbol vocabulary and adds "
			+ "the boundary tokens.")
	_phoneme_input = TextEdit.new()
	_phoneme_input.text = SAMPLE_PHONEMES
	_phoneme_input.custom_minimum_size = Vector2(0, 54)
	box.add_child(_phoneme_input)
	var r := _row(box)
	r.add_child(_button("Tokenize", _test_tokenize))
	r.add_child(_button("Chunking test", _test_chunking))


func _build_gpu_section(col: Control) -> void:
	var box := _section(col, "4 · GPU compute",
			"Creates a local RenderingDevice, compiles kernels and checks the "
			+ "elementwise kernel against values computed on the CPU.")
	var r := _row(box)
	r.add_child(_button("Kernel correctness", _test_kernels))
	r.add_child(_button("Throughput benchmark", _test_benchmark))


func _build_audio_section(col: Control) -> void:
	var box := _section(col, "5 · Audio output stage",
			"The real final stage of synthesis: a waveform is built on the GPU, "
			+ "converted to PCM16 by kernels/pcm16.glsl, and wrapped in an "
			+ "AudioStreamWAV. This should be audible.")
	var r := _row(box)
	r.add_child(_button("Generate + play tone", _test_audio_chain))
	r.add_child(_button("Play last again", func():
			if _last_stream: player.stream = _last_stream; player.play()))


func _build_synthesis_section(col: Control) -> void:
	var box := _section(col, "6 · Full synthesis",
			"Text in, speech out: G2P, then all 3945 graph nodes on the GPU, "
			+ "then PCM16. Runs on a worker thread, so the UI keeps "
			+ "responding while it works.")
	_text_input = TextEdit.new()
	_text_input.text = "Hello from Godot. This is Kokoro speaking."
	_text_input.custom_minimum_size = Vector2(0, 54)
	box.add_child(_text_input)
	var r := _row(box)
	r.add_child(_button("Speak", _test_speak))
	r.add_child(_button("Speak (streaming)", _test_speak_streaming))
	r.add_child(_button("Show phonemes", _test_phonemes))
	_synth_status = Label.new()
	_synth_status.modulate = Color(1, 1, 1, 0.7)
	box.add_child(_synth_status)


# --------------------------------------------------------------------- checks

func _check_system() -> void:
	_log("[b]best-tts bench[/b]  Godot %s" % Engine.get_version_info().string)
	if gpu != null and gpu.is_ready():
		_ok("RenderingDevice already up on %s" % gpu.rd.get_device_name())
		return
	gpu = GPUScript.new()
	var err: String = gpu.initialize()
	if err != "":
		_status.text = "GPU UNAVAILABLE — " + err
		_status.modulate = Color(1, 0.5, 0.4)
		_fail(err)
		gpu = null
		return
	var adapter: String = gpu.rd.get_device_name()
	var vendor: String = gpu.rd.get_device_vendor_name()
	_status.text = "Renderer OK · %s %s" % [vendor, adapter]
	_status.modulate = Color(0.6, 1.0, 0.6)
	_ok("RenderingDevice ready on %s %s" % [vendor, adapter])


func _test_model() -> void:
	_log("\n[b]— packed model —[/b]")
	var t0 := Time.get_ticks_usec()
	var m = KpkModelScript.new()
	var err: int = m.load_from(MODEL_PATH)
	var ms := (Time.get_ticks_usec() - t0) / 1000.0
	if err != OK:
		_fail("load_from returned %d — did you run tools/pack_model.py?" % err)
		return
	model = m
	var cpu := 0
	for n in m.nodes:
		if n.place == KpkModelScript.Place.CPU:
			cpu += 1
	_ok("loaded %d nodes in %.1f ms" % [m.nodes.size(), ms])
	_info("%d weight tensors, %.1f MB blob" % [m.weights.size(), m.blob.size() / 1e6])
	_info("%d CPU (shape/index) / %d GPU (tensor math) nodes"
			% [cpu, m.nodes.size() - cpu])
	_info("%d distinct op types, output tensor '%s'"
			% [m.ops.size(), m.name_of(m.output_tid)])


func _current_voice() -> String:
	if _voice_picker.item_count == 0:
		return ""
	var meta = _voice_picker.get_item_metadata(_voice_picker.selected)
	return "" if meta == null else str(meta)


func _test_voice() -> void:
	_log("\n[b]— voice pack —[/b]")
	var name := _current_voice()
	if name == "":
		_fail("no voices in assets/voices/")
		return
	var t0 := Time.get_ticks_usec()
	var pack := voices.load_pack(name)
	var ms := (Time.get_ticks_usec() - t0) / 1000.0
	if pack.is_empty():
		_fail("could not load %s" % name)
		return
	_ok("%s — %d floats (%d rows x %d) in %.1f ms"
			% [name, pack.size(), VoiceLoader.ROWS, VoiceLoader.DIM, ms])

	var enc: Dictionary = tokenizer.encode(_phoneme_input.text)
	var style := voices.style_for(name, enc.phoneme_count)
	var lo := INF
	var hi := -INF
	for v in style:
		lo = minf(lo, v)
		hi = maxf(hi, v)
	_info("style row %d for %d phonemes: %d dims, range %.4f .. %.4f"
			% [maxi(0, enc.phoneme_count - 1), enc.phoneme_count,
			style.size(), lo, hi])


func _test_all_voices() -> void:
	_log("\n[b]— all voice packs —[/b]")
	var names := VoiceLoader.list_voices()
	var t0 := Time.get_ticks_usec()
	var bad := 0
	for n in names:
		if voices.load_pack(n).size() != VoiceLoader.ROWS * VoiceLoader.DIM:
			_fail("%s did not load" % n)
			bad += 1
	var ms := (Time.get_ticks_usec() - t0) / 1000.0
	if bad == 0:
		_ok("all %d voices loaded in %.0f ms" % [names.size(), ms])
	else:
		_fail("%d of %d voices failed" % [bad, names.size()])
	voices.clear_cache()


func _test_tokenize() -> void:
	_log("\n[b]— tokenizer —[/b]")
	if not tokenizer.load_vocab():
		_fail("vocab.json missing")
		return
	var enc: Dictionary = tokenizer.encode(_phoneme_input.text)
	var ids: PackedInt32Array = enc.ids
	if ids.size() <= 2:
		_fail("nothing encoded — is the input IPA phonemes?")
		return
	_ok("%d phonemes -> %d ids (boundary zeros included)"
			% [enc.phoneme_count, ids.size()])
	var preview := ""
	for i in mini(24, ids.size()):
		preview += str(ids[i]) + " "
	_info("ids: %s%s" % [preview, "..." if ids.size() > 24 else ""])
	_info("vocab has %d symbols" % tokenizer.vocab.size())
	if enc.unknown.size() > 0:
		_info("[color=#d29922]dropped %d unsupported character(s): %s[/color]"
				% [enc.unknown.size(), " ".join(enc.unknown)])
	if enc.truncated:
		_info("input hit the %d-phoneme limit" % tokenizer.max_phonemes)

	# Round-trip sanity: every id must be inside the embedding table.
	var out_of_range := 0
	for id in ids:
		if id < 0 or id >= 178:
			out_of_range += 1
	if out_of_range == 0:
		_ok("every id lies inside the 178-slot embedding table")
	else:
		_fail("%d ids out of range" % out_of_range)


func _test_chunking() -> void:
	_log("\n[b]— chunking —[/b]")
	tokenizer.load_vocab()
	var long_text := ""
	for i in 40:
		long_text += SAMPLE_PHONEMES + " "
	var chunks := tokenizer.chunk(long_text, 120)
	if chunks.is_empty():
		_fail("chunker returned nothing")
		return
	var worst := 0
	for c in chunks:
		worst = maxi(worst, tokenizer.encode(c).phoneme_count)
	_ok("%d phonemes -> %d chunks, longest %d"
			% [tokenizer.encode(long_text).phoneme_count, chunks.size(), worst])
	if worst <= 120:
		_ok("every chunk respects the limit")
	else:
		_fail("a chunk exceeded the limit (%d)" % worst)


func _test_kernels() -> void:
	_log("\n[b]— GPU kernels —[/b]")
	if gpu == null:
		_fail("no RenderingDevice")
		return
	if not gpu.load_kernel("elementwise"):
		_fail("elementwise.glsl did not compile — run: godot --headless --import")
		return
	_ok("elementwise.glsl compiled")

	var cases := [
		["Mul, broadcast [2,3,4] x [3,1]", Ops.Op.MUL, [2, 3, 4], [3, 1]],
		["Add, scalar right-hand side", Ops.Op.ADD, [5, 7], []],
		["Sub, matching shapes", Ops.Op.SUB, [4, 6], [4, 6]],
		["Div, broadcast [2,1,5] x [2,4,5]", Ops.Op.DIV, [2, 1, 5], [2, 4, 5]],
		["Sqrt", Ops.Op.SQRT, [3, 9], []],
		["Sigmoid", Ops.Op.SIGMOID, [64], []],
	]
	var failed := 0
	for case in cases:
		var label: String = case[0]
		var op: int = case[1]
		var ash: Array = case[2]
		var bsh: Array = case[3]
		var a := _rand(ash, 11)
		var b := _rand(bsh, 22)
		var oshape := Ops.broadcast_shape(PackedInt32Array(ash),
				PackedInt32Array(bsh))
		var want := _cpu_reference(op, ash, a, bsh, b, oshape)
		var got := _run_elementwise(op, ash, a, bsh, b)
		var worst := 0.0
		for i in want.size():
			worst = maxf(worst, absf(got[i] - want[i]))
		if worst > 1e-5:
			_fail("%s — max error %s" % [label, String.num_scientific(worst)])
			failed += 1
		else:
			_info("%s ok (n=%d, err %s)"
					% [label, want.size(), String.num_scientific(worst)])
	if failed == 0:
		_ok("all %d elementwise cases match the CPU reference" % cases.size())


func _test_benchmark() -> void:
	_log("\n[b]— throughput —[/b]")
	if gpu == null or not gpu.load_kernel("elementwise"):
		_fail("no GPU or kernel")
		return
	var n := 1 << 21          # 2M elements, comparable to a long utterance
	var a := PackedFloat32Array()
	a.resize(n)
	a.fill(1.5)
	var buf_a: RID = gpu.create_buffer_f32(a)
	var buf_o: RID = gpu.create_buffer(n * 4)
	var shape := PackedInt32Array([n])
	var st := PackedInt32Array([1])
	var zero := PackedInt32Array([0])
	var push: PackedByteArray = Ops.elementwise_push(
			Ops.Op.MUL, n, shape, st, zero, zero)

	const ITERS := 100
	var t0 := Time.get_ticks_usec()
	gpu.begin_compute()
	for i in ITERS:
		gpu.dispatch("elementwise", [buf_a, buf_a, buf_a, buf_o] as Array[RID],
				push, (n + 63) / 64)
	gpu.end_compute()
	gpu.submit_and_wait()
	var us := Time.get_ticks_usec() - t0

	gpu.free_buffer(buf_a)
	gpu.free_buffer(buf_o)
	_ok("%d dispatches x %d elements in %.1f ms" % [ITERS, n, us / 1000.0])
	_info("%.2f us per dispatch, %.1f Gelem/s"
			% [float(us) / ITERS, float(n) * ITERS / us / 1000.0])
	_info("the full model is ~3536 GPU dispatches per utterance")


func _test_audio_chain() -> void:
	_log("\n[b]— GPU to audio —[/b]")
	if gpu == null:
		_fail("no RenderingDevice")
		return
	if not gpu.load_kernel("elementwise") or not gpu.load_kernel("pcm16"):
		_fail("kernels missing — run: godot --headless --import")
		return

	var rate := 24000
	var n := rate       # one second
	var phase := PackedFloat32Array()
	phase.resize(n)
	# A 220 Hz tone with a slow vibrato, so a wrong PCM conversion is obvious.
	for i in n:
		var t := float(i) / rate
		phase[i] = TAU * (220.0 + 8.0 * sin(TAU * 5.0 * t)) * t

	var buf_phase: RID = gpu.create_buffer_f32(phase)
	var buf_wave: RID = gpu.create_buffer(n * 4)
	var buf_amp: RID = gpu.create_buffer_f32(PackedFloat32Array([0.35]))
	var buf_out: RID = gpu.create_buffer(n * 4)

	var shape := PackedInt32Array([n])
	var one := PackedInt32Array([1])
	var zero := PackedInt32Array([0])

	gpu.begin_compute()
	gpu.dispatch("elementwise",
			[buf_phase, buf_phase, buf_phase, buf_wave] as Array[RID],
			Ops.elementwise_push(Ops.Op.SIN, n, shape, one, zero, zero),
			(n + 63) / 64)
	gpu.barrier()
	gpu.dispatch("elementwise",
			[buf_wave, buf_amp, buf_amp, buf_out] as Array[RID],
			Ops.elementwise_push(Ops.Op.MUL, n, shape, one, zero, zero),
			(n + 63) / 64)
	gpu.end_compute()
	gpu.submit_and_wait()
	_ok("waveform generated on the GPU (sin then scale, 2 dispatches)")

	var stream: AudioStreamWAV = KokoroAudioScript.stream_from_gpu(
			gpu, buf_out, n, rate)
	gpu.free_buffer(buf_phase)
	gpu.free_buffer(buf_wave)
	gpu.free_buffer(buf_amp)
	gpu.free_buffer(buf_out)

	if stream == null:
		_fail("PCM16 conversion produced no stream")
		return
	_ok("pcm16.glsl -> AudioStreamWAV: %d bytes, %d Hz, %.2f s"
			% [stream.data.size(), stream.mix_rate,
			stream.data.size() / 2.0 / stream.mix_rate])

	# Verify the bytes really are a sine and not silence or clipping.
	var peak := 0
	var nonzero := 0
	for i in mini(4000, stream.data.size() / 2):
		var s := stream.data.decode_s16(i * 2)
		peak = maxi(peak, absi(s))
		if s != 0:
			nonzero += 1
	if peak > 8000 and peak < 32767 and nonzero > 3000:
		_ok("PCM sanity: peak %d, %d/4000 non-zero samples" % [peak, nonzero])
	else:
		_fail("PCM looks wrong: peak %d, %d non-zero" % [peak, nonzero])

	_last_stream = stream
	player.stream = stream
	player.play()
	_info("playing — you should hear a one second 220 Hz tone with vibrato")


## Brings the engine up on first use. It owns a second RenderingDevice on a
## worker thread, separate from the one the kernel tests use.
func _ensure_engine() -> bool:
	if _tts != null:
		if _tts.is_ready():
			return true
		_info("engine still loading...")
		await _tts.engine_ready
		return _tts.is_ready()

	_tts = TTSScript.new()
	_tts.preload_on_ready = false
	_tts.name = "KokoroTTS"
	add_child(_tts)
	_tts.synthesis_progress.connect(func(done: int, total: int):
		if total > 1:
			_synth_status.text = "synthesizing chunk %d of %d..." % [done, total])

	_info("loading the model on a worker thread (a few seconds)...")
	var t0 := Time.get_ticks_msec()
	_tts.start()
	await _tts.engine_ready
	if not _tts.is_ready():
		_fail("engine failed: " + _tts.get_error())
		return false
	_ok("engine ready in %d ms" % (Time.get_ticks_msec() - t0))
	return true


func _test_phonemes() -> void:
	_log("\n[b]— text to phonemes —[/b]")
	var text := _text_input.text.strip_edges()
	if text == "":
		_fail("nothing to convert")
		return
	if _tts == null:
		_tts = TTSScript.new()
		_tts.preload_on_ready = false
		add_child(_tts)
	var voice := _current_voice()
	var r: Dictionary = _tts.phonemize(text, voice)
	if r.has("error"):
		_fail(r["error"])
		return
	_ok("%s -> %s" % [VoiceLoader.describe(voice), r["lang"]])
	_info("\"%s\"" % text)
	_info("[color=#7ee787]%s[/color]" % r["phonemes"])
	var oov: PackedStringArray = r["oov"]
	if oov.size() > 0:
		_info("letter-to-sound fallback used for: %s" % " ".join(oov))
	# Phonemes the model cannot represent would be silently dropped.
	var enc: Dictionary = tokenizer.encode(r["phonemes"])
	if enc["unknown"].size() > 0:
		_fail("%d symbol(s) outside the vocabulary: %s"
				% [enc["unknown"].size(), " ".join(enc["unknown"])])
	else:
		_ok("%d phonemes, all in the model's vocabulary"
				% int(enc["phoneme_count"]))


func _test_speak() -> void:
	_log("\n[b]— full synthesis —[/b]")
	var text := _text_input.text.strip_edges()
	if text == "":
		_fail("nothing to speak")
		return
	if _speaking:
		_info("already speaking")
		return
	_speaking = true
	_synth_status.text = "loading..."
	if not await _ensure_engine():
		_speaking = false
		_synth_status.text = ""
		return

	var voice := _current_voice()
	var r: Dictionary = _tts.phonemize(text, voice)
	_info("\"%s\"" % text)
	_info("phonemes: %s" % r.get("phonemes", ""))
	_synth_status.text = "synthesizing..."

	var t0 := Time.get_ticks_msec()
	var clip: AudioStreamWAV = await _tts.speak(text, voice)
	var ms := Time.get_ticks_msec() - t0
	_speaking = false
	_synth_status.text = ""

	if clip == null:
		_fail("synthesis failed: " + _tts.get_error())
		return
	var seconds := clip.data.size() / 2.0 / clip.mix_rate
	_ok("%.2f s of speech in %d ms (%.1fx realtime) as %s"
			% [seconds, ms, seconds * 1000.0 / maxf(1.0, ms),
			VoiceLoader.describe(voice)])

	var peak := 0
	for i in clip.data.size() / 2:
		peak = maxi(peak, absi(clip.data.decode_s16(i * 2)))
	if peak < 1000:
		_fail("output is silent (peak %d)" % peak)
		return
	_info("peak %d of 32767, %d Hz mono" % [peak, clip.mix_rate])

	_last_stream = clip
	player.stream = clip
	player.play()
	_info("[color=#7ee787]playing — you should hear the text spoken[/color]")


func _test_speak_streaming() -> void:
	_log("\n[b]— streaming synthesis —[/b]")
	var text := _text_input.text.strip_edges()
	if text == "":
		_fail("nothing to speak")
		return
	if _speaking:
		_info("already speaking")
		return
	_speaking = true
	_synth_status.text = "loading..."
	if not await _ensure_engine():
		_speaking = false
		_synth_status.text = ""
		return

	# Short chunks make the point: playback starts on the first one while the
	# rest are still being synthesized.
	var previous: int = _tts.chunk_phonemes
	_tts.chunk_phonemes = 80
	_synth_status.text = "synthesizing..."

	var t0 := Time.get_ticks_msec()
	var p = _tts.speak_streaming(text, _current_voice())
	# First audio out is the number that matters for streaming.
	p.finished.connect(func(): pass)
	await get_tree().process_frame
	while not p.playing and _speaking:
		await get_tree().process_frame
		if Time.get_ticks_msec() - t0 > 60000:
			break
	_ok("first audio started after %d ms" % (Time.get_ticks_msec() - t0))
	_info("[color=#7ee787]playing chunk by chunk...[/color]")

	await p.utterance_finished
	_ok("finished, %.2f s total" % ((Time.get_ticks_msec() - t0) / 1000.0))
	_tts.chunk_phonemes = previous
	_synth_status.text = ""
	_speaking = false
	p.queue_free()


func _run_all() -> void:
	_log_box.clear()
	_check_system()
	_test_model()
	_test_voice()
	_test_all_voices()
	_test_tokenize()
	_test_chunking()
	_test_kernels()
	_test_benchmark()
	_test_audio_chain()
	_test_phonemes()
	await _test_speak()
	_log("\n[b]bench complete[/b]")


# ------------------------------------------------------------------- helpers

func _rand(shape: Array, seed: int) -> PackedFloat32Array:
	var n := 1
	for d in shape:
		n *= d
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var out := PackedFloat32Array()
	out.resize(maxi(n, 1))
	for i in out.size():
		out[i] = rng.randf_range(0.5, 3.5)
	return out


## Deliberately naive broadcast, independent of the stride helpers under test.
func _cpu_reference(op: int, ashape: Array, a: PackedFloat32Array,
		bshape: Array, b: PackedFloat32Array,
		oshape: PackedInt32Array) -> PackedFloat32Array:
	var rank := oshape.size()
	var n := Ops.num_elements(oshape)
	var out := PackedFloat32Array()
	out.resize(n)
	for i in n:
		var coord := []
		coord.resize(rank)
		var rem := i
		for d in range(rank - 1, -1, -1):
			coord[d] = rem % oshape[d]
			rem /= oshape[d]
		var x := _pick(ashape, a, coord)
		var y := _pick(bshape, b, coord)
		match op:
			Ops.Op.ADD: out[i] = x + y
			Ops.Op.SUB: out[i] = x - y
			Ops.Op.MUL: out[i] = x * y
			Ops.Op.DIV: out[i] = x / y
			Ops.Op.SQRT: out[i] = sqrt(x)
			Ops.Op.SIGMOID: out[i] = 1.0 / (1.0 + exp(-x))
			_: out[i] = x
	return out


func _pick(shape: Array, data: PackedFloat32Array, coord: Array) -> float:
	if shape.is_empty():
		return data[0]
	var pad: int = coord.size() - shape.size()
	var idx := 0
	for d in shape.size():
		var c: int = coord[pad + d]
		if shape[d] == 1:
			c = 0
		idx = idx * int(shape[d]) + c
	return data[idx]


func _run_elementwise(op: int, ashape: Array, a: PackedFloat32Array,
		bshape: Array, b: PackedFloat32Array) -> PackedFloat32Array:
	var oshape := Ops.broadcast_shape(PackedInt32Array(ashape),
			PackedInt32Array(bshape))
	var n := Ops.num_elements(oshape)
	var sa := Ops.broadcast_strides(PackedInt32Array(ashape), oshape)
	var sb := Ops.broadcast_strides(PackedInt32Array(bshape), oshape)
	var sc := Ops.broadcast_strides(PackedInt32Array(), oshape)
	var packed := Ops.collapse(oshape, [sa, sb, sc])

	var buf_a: RID = gpu.create_buffer_f32(a)
	var buf_b: RID = gpu.create_buffer_f32(b if not b.is_empty()
			else PackedFloat32Array([0.0]))
	var buf_o: RID = gpu.create_buffer(n * 4)
	var push: PackedByteArray = Ops.elementwise_push(op, n, packed.shape,
			packed.strides[0], packed.strides[1], packed.strides[2])

	gpu.begin_compute()
	gpu.dispatch("elementwise", [buf_a, buf_b, buf_b, buf_o] as Array[RID],
			push, (n + 63) / 64)
	gpu.end_compute()
	gpu.submit_and_wait()

	var out: PackedFloat32Array = gpu.read_f32(buf_o, n)
	gpu.free_buffer(buf_a)
	gpu.free_buffer(buf_b)
	gpu.free_buffer(buf_o)
	return out


func _exit_tree() -> void:
	if gpu != null:
		gpu.cleanup()
