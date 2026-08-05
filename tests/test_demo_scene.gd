extends SceneTree

## The demo scene builds, its authored nodes carry the values typed into them,
## and none of it crashes on a machine with no compute device.
##
##   godot --headless --script res://tests/test_demo_scene.gd
##
## Deliberately runnable headless: that is the no-RenderingDevice case, and a
## demo that hard-crashes there would be a bad first impression for anyone on
## the Compatibility renderer. With a real driver it also exercises the engine.

const SCENE := "res://demo/demo.tscn"

var _fail := 0


func _init() -> void:
	await process_frame

	var packed: PackedScene = load(SCENE)
	_check(packed != null, "demo.tscn loads")
	if packed == null:
		quit(1)
		return

	var demo: Node = packed.instantiate()
	_check(demo != null, "and instantiates")
	root.add_child(demo)
	await process_frame

	# The three authored players, with the values the scene file gives them.
	var narrator = demo.get_node_or_null("Narrator")
	_check(narrator is BestVoicePlayer, "Narrator is a BestVoicePlayer")
	_check(narrator.voice == "bm_lewis", "Narrator kept its voice from the scene")
	_check(narrator.text.contains("road was quiet"),
			"Narrator kept its text from the scene")

	var merchant = demo.get_node_or_null("Merchant")
	_check(merchant is BestVoicePlayer, "Merchant is a BestVoicePlayer")
	_check(merchant.stream is BestVoiceStream,
			"Merchant's stream is a BestVoiceStream resource")
	if merchant.stream is BestVoiceStream:
		_check(merchant.stream.text.contains("wares"),
				"the resource kept its text: \"%s\"" % merchant.stream.text)
		_check(merchant.stream.voice == "bm_george",
				"and its voice: %s" % merchant.stream.voice)

	_check(demo.get_node_or_null("FreeForm") is BestVoicePlayer,
			"FreeForm is a BestVoicePlayer")

	# The UI is built in code; if any of it were malformed the frames above
	# would already have thrown. Check it produced something.
	_check(demo.get_child_count() > 3, "the UI was built (%d children)"
			% demo.get_child_count())

	var crowd := 0
	for child in demo.get_children():
		if child is BestVoicePlayer and str(child.name).begins_with("Crowd"):
			crowd += 1
	_check(crowd == 10, "ten crowd players exist")

	# Let the engine either come up or fail; both must be survivable.
	for i in 8:
		await process_frame
	var tts := BestTTS.engine()
	if tts == null:
		print("  note  no engine (no scene tree or editor) — nothing to check")
	elif tts.is_ready():
		print("  note  engine came up; the demo would be interactive")
	else:
		print("  note  engine not ready: %s"
				% (tts.get_error() if tts.get_error() != "" else "still loading"))

	demo.queue_free()
	BestTTS.shutdown()
	for i in 4:
		await process_frame

	if _fail == 0:
		print("\nOK: the demo scene is sound")
	quit(1 if _fail > 0 else 0)


func _check(ok: bool, what: String) -> void:
	if ok:
		print("  ok    %s" % what)
	else:
		printerr("  FAIL  %s" % what)
		_fail += 1
