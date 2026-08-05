extends SceneTree

## The zero-setup layer: BestTTS, BestVoiceStream and the three voice players.
##
##   godot --rendering-driver vulkan --script res://tests/test_nodes.gd
##
## Needs a real rendering driver for the compute device, and runs with whatever
## audio driver is available — the playback assertions only check that the
## engine accepted the stream and started, not that anything was audible.

var _fail := 0


func _init() -> void:
	# Engine.get_main_loop() is not assigned until this SceneTree has finished
	# constructing, and the shared engine needs it to find the tree root. One
	# frame is also what a real game has before any node calls anything.
	await process_frame

	await _singleton()
	await _stream_resource()
	await _player()
	await _player_with_stream()
	await _positional()
	await _cancelling()
	await _inspector()
	await _teardown()

	if _fail == 0:
		print("\nOK: the node API works")
	quit(1 if _fail > 0 else 0)


func _check(ok: bool, what: String) -> void:
	if ok:
		print("  ok    %s" % what)
	else:
		printerr("  FAIL  %s" % what)
		_fail += 1


# --------------------------------------------------------------------- BestTTS

func _singleton() -> void:
	print("\n[BestTTS] the shared engine")
	_check(BestTTS.list_voices().size() == 50, "50 voices without an engine")
	_check(BestTTS.has_voice("af_heart"), "has_voice")
	_check(not BestTTS.has_voice("AF_HEART"), "voice names are case-sensitive")

	var t0 := Time.get_ticks_msec()
	var tts := BestTTS.engine()
	_check(tts != null, "engine() created one")
	if tts == null:
		return
	_check(BestTTS.engine() == tts, "engine() is a singleton")

	# The node is parented deferred, so it is not in the tree this frame.
	await process_frame
	_check(tts.is_inside_tree(), "engine parented itself to the tree root")

	if not tts.is_ready():
		await tts.engine_ready
	print("        model ready in %d ms" % (Time.get_ticks_msec() - t0))
	_check(BestTTS.is_ready(), "is_ready")

	var clip: AudioStreamWAV = await BestTTS.speak("One line, no setup.")
	_check(clip != null and clip.data.size() > 0, "BestTTS.speak returned audio")


# -------------------------------------------------------------- BestVoiceStream

func _stream_resource() -> void:
	print("\n[BestVoiceStream] a line as a resource")
	var line := BestVoiceStream.of("The bridge is out ahead.", "bm_george")
	_check(line is AudioStreamWAV, "is a real AudioStreamWAV")
	_check(not line.is_rendered(), "starts empty")

	var ok: bool = await line.render()
	_check(ok and line.is_rendered(), "render() filled it in")
	_check(line.mix_rate == 24000, "24 kHz")
	_check(line.format == AudioStreamWAV.FORMAT_16_BITS, "16-bit PCM")
	_check(line.duration() > 1.0, "duration() reports %.2f s" % line.duration())

	# A stock player has to accept it with no special handling at all.
	var plain := AudioStreamPlayer.new()
	root.add_child(plain)
	plain.stream = line
	plain.play()
	_check(plain.playing, "a stock AudioStreamPlayer plays it")
	plain.queue_free()

	var before := line.data.size()
	line.text = "Something else entirely."
	_check(line.data.size() == 0 and before > 0, "editing the text discards audio")
	_check(await line.render(), "and it renders again")

	# Same text, same voice: the engine cache should make this immediate.
	var again := BestVoiceStream.of("The bridge is out ahead.", "bm_george")
	var t0 := Time.get_ticks_msec()
	await again.render()
	var ms := Time.get_ticks_msec() - t0
	_check(again.is_rendered(), "a second copy renders (%d ms, cached)" % ms)


# ------------------------------------------------------------- BestVoicePlayer

func _player() -> void:
	print("\n[BestVoicePlayer] text in the inspector")
	var p := BestVoicePlayer.new()
	p.name = "Voice"
	p.text = "I have wares, if you have coin."
	p.voice = "bm_george"
	p.prepare_on_ready = false
	root.add_child(p)

	_check(not p.is_speaking(), "idle before speaking")
	var req := p.speak()
	_check(req != null, "speak() returned a request")
	_check(p.is_speaking(), "speaking as soon as the line is queued")
	await req.finished
	_check(req.is_ok(), "the line was synthesized")
	_check(p.stream != null and p.stream.data.size() > 0,
			"the player was given the audio")
	_check(p.playing, "and started")
	p.stop()

	# prepare_on_ready is the path that makes the ordinary play() work: the
	# node fills its own `stream` while the scene loads.
	var p2 := BestVoicePlayer.new()
	p2.text = "Prepared while the scene loaded."
	root.add_child(p2)
	await p2.prepared
	_check(p2.stream is AudioStreamWAV and p2.stream.data.size() > 0,
			"prepare_on_ready filled stream with plain audio")
	_check(not p2.playing, "and did not start it")
	p2.play()
	_check(p2.playing, "the native play() works from there")
	p.queue_free()
	p2.queue_free()


func _player_with_stream() -> void:
	print("\n[BestVoicePlayer] a BestVoiceStream in the stream slot")
	var p := BestVoicePlayer.new()
	var line := BestVoiceStream.of("Authored as a resource.", "af_heart")
	p.stream = line
	# The resource must win over `text`, or a scene with both would say both.
	p.text = "This text should be ignored."
	root.add_child(p)

	_check(not line.is_rendered(), "not rendered yet, one frame in")
	await p.prepared
	_check(line.is_rendered(), "prepare_on_ready rendered the resource")
	_check(p.stream == line, "and kept the same resource in the slot")
	_check(not p.playing, "without starting it")

	p.play()
	_check(p.playing, "plays with the ordinary native play()")
	p.stop()

	# speak() on a prepared resource must replay it, not synthesize it again.
	var before := line.data.size()
	p.speak()
	_check(p.playing and line.data.size() == before,
			"speak() replays a prepared resource instead of re-synthesizing")
	p.queue_free()


func _positional() -> void:
	print("\n[BestVoicePlayer2D/3D] positional variants")
	var p2 := BestVoicePlayer2D.new()
	p2.text = "Over here."
	p2.prepare_on_ready = false
	root.add_child(p2)
	var r2 := p2.speak()
	await r2.finished
	_check(r2.is_ok() and p2.stream != null, "2D spoke")

	var p3 := BestVoicePlayer3D.new()
	p3.text = "Behind you."
	p3.voice = "am_michael"
	p3.prepare_on_ready = false
	root.add_child(p3)
	var r3 := p3.speak()
	await r3.finished
	_check(r3.is_ok() and p3.stream != null, "3D spoke")
	p2.queue_free()
	p3.queue_free()


func _cancelling() -> void:
	print("\n[cancel] a line the player no longer wants")
	var p := BestVoicePlayer.new()
	p.text = "A long line that will be abandoned. ".repeat(6)
	p.prepare_on_ready = false
	root.add_child(p)
	var req := p.speak()
	p.cancel()
	_check(not p.is_speaking(), "not speaking after cancel()")
	await req.finished
	_check(not req.is_ok(), "the cancelled line produced no audio")
	_check(req.error != "", "and says why: %s" % req.error)
	p.queue_free()


func _inspector() -> void:
	print("\n[inspector] the voice dropdown")
	for obj in [BestVoicePlayer.new(), BestVoicePlayer2D.new(),
			BestVoicePlayer3D.new(), BestVoiceStream.new()]:
		var found := {}
		for prop in obj.get_property_list():
			if prop.name == "voice":
				found = prop
		var hint_ok: bool = found.get("hint", 0) == PROPERTY_HINT_ENUM \
				and str(found.get("hint_string", "")).contains("af_heart")
		var what: String = obj.get_script().resource_path.get_file()
		_check(hint_ok, "%s exposes a voice dropdown" % what)
		if obj is Node:
			obj.free()


func _teardown() -> void:
	print("\n[shutdown] releasing the shared engine")
	var before := BestTTS.engine().gpu_memory_bytes()
	BestTTS.shutdown()
	_check(before > 0, "engine held %.0f MB" % (before / 1048576.0))
	# A call after shutdown has to build a new engine rather than crash.
	var again := BestTTS.engine()
	_check(again != null, "the next call builds a fresh engine")
	BestTTS.shutdown()
	# queue_free() needs a frame or two to run, and quitting before it does
	# reports every node this test made as a leak.
	for i in 4:
		await process_frame
