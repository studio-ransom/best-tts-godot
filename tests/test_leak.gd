extends SceneTree

## The soak showed the device creeping ~33 KB per utterance — small enough to
## look like noise over ten runs and 300 MB over a long play session. This
## prints every counter that could be doing it, so the one that tracks the run
## count instead of the workload is obvious.
##
##   godot --rendering-driver vulkan --script res://tests/test_leak.gd

const TTSScript := preload("res://addons/best-tts/kokoro_tts.gd")

var _tts


func _init() -> void:
	_tts = TTSScript.new()
	_tts.preload_on_ready = false
	# Repeated text would come straight back from the cache and this
	# would measure a dictionary lookup instead of the GPU path.
	_tts.cache_enabled = false
	root.add_child(_tts)
	_tts.engine_failed.connect(func(r: String): printerr(r); quit(1))
	_tts.start()
	await _tts.engine_ready

	print("%4s %10s %8s %7s %7s %7s %7s %7s"
			% ["run", "device KB", "buffers", "sets", "pool", "live", "const",
			"packed"])
	# Identical text every time: anything that grows is counting runs.
	for i in 41:
		var clip: AudioStreamWAV = await _tts.speak("Watch your step.")
		if clip == null:
			printerr("FAIL: " + _tts.get_error())
			quit(1)
			return
		await _tts.went_idle
		if i % 5 == 0:
			_row(i)
	quit(0)


func _row(i: int) -> void:
	var g = _tts._gpu
	var e = _tts._executor
	var pool := 0
	for size in e._pool:
		pool += (e._pool[size] as Array).size()
	print("%4d %10d %8d %7d %7d %7d %7d %7d"
			% [i, _tts.gpu_memory_bytes() / 1024, g._buffers.size(),
			g._uniform_sets.size(), pool, e._live.size(),
			e._const_tids.size(), e._packed.size()])
