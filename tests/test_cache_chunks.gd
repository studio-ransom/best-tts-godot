extends SceneTree

## Assert that both disk and memory cached streams are played as a single chunk.

const TTSScript := preload("res://addons/best-tts/kokoro_tts.gd")

const LINE := "I've said this line already."

var _tts
var _fail := 0


func _init() -> void:
	_tts = TTSScript.new()
	_tts.preload_on_ready = false
	_tts.cache_enabled = true
	_tts.cache_on_disk = false
	root.add_child(_tts)
	_tts.engine_failed.connect(func(reason: String):
		printerr("FAIL: engine did not start — " + reason)
		quit(1))
	_tts.start()
	await _tts.engine_ready

	await _memory_hit()
	await _disk_hit()
	await _stream_into_hit()
	await _speak_streaming_hit()

	_tts.clear_cache(true)
	if _fail == 0:
		print("\nOK: cache hits are delivered as a chunk")
	quit(1 if _fail > 0 else 0)


func _check(ok: bool, what: String) -> void:
	if ok:
		print("  ok    %s" % what)
	else:
		printerr("  FAIL  %s" % what)
		_fail += 1


## Records every chunk_ready, and whether each arrived before `finished`.
func _watch(req) -> Array:
	var chunks := []
	req.chunk_ready.connect(func(clip: AudioStreamWAV, i: int, n: int):
		chunks.append({"clip": clip, "index": i, "total": n,
				"before_finished": not req.is_done()}))
	return chunks


func _check_hit(req, chunks: Array, label: String) -> void:
	_check(req.from_cache, "%s: served from cache" % label)
	_check(req.stream != null, "%s: finished with a clip" % label)
	_check(chunks.size() == 1,
			"%s: exactly one chunk_ready (got %d)" % [label, chunks.size()])
	if chunks.size() != 1:
		return
	var c: Dictionary = chunks[0]
	_check(c["index"] == 0 and c["total"] == 1,
			"%s: chunk is 0 of 1 (got %d of %d)"
			% [label, c["index"], c["total"]])
	_check(c["clip"] == req.stream, "%s: chunk is the finished clip" % label)
	_check(c["before_finished"], "%s: chunk arrived before finished" % label)


func _warm(line: String) -> void:
	var req = _tts.say(line)
	await req.finished
	if req.from_cache or req.stream == null:
		printerr("  FAIL  could not warm the cache: %s" % req.error)
		_fail += 1


func _memory_hit() -> void:
	print("\n[memory tier]")
	_tts.clear_cache(true)
	await _warm(LINE)
	var req = _tts.say(LINE)
	var chunks := _watch(req)
	await req.finished
	_check_hit(req, chunks, "memory hit")


func _disk_hit() -> void:
	print("\n[disk tier]")
	_tts.clear_cache(true)
	_tts.cache_on_disk = true
	_tts._cache.disk_dir = _tts.cache_dir
	await _warm(LINE)
	_tts.clear_cache(false)
	var req = _tts.say(LINE)
	var chunks := _watch(req)
	await req.finished
	_check_hit(req, chunks, "disk hit")
	_tts.clear_cache(true)
	_tts.cache_on_disk = false
	_tts._cache.disk_dir = ""


func _stream_into_hit() -> void:
	print("\n[stream_into]")
	_tts.clear_cache(true)
	await _warm(LINE)
	var player := AudioStreamPlayer.new()
	root.add_child(player)
	var req = _tts.stream_into(player, LINE)
	await req.finished
	_check(req.from_cache, "stream_into: served from cache")
	_check(player.stream != null and player.stream == req.stream,
			"stream_into: the cached clip reached the player")
	player.queue_free()


func _speak_streaming_hit() -> void:
	print("\n[speak_streaming]")
	_tts.clear_cache(true)
	await _warm(LINE)
	var player = _tts.speak_streaming(LINE)
	# The player closes when the request finishes; give the deferred
	# delivery a couple of frames to land.
	for i in 3:
		await process_frame
	_check(player.stream != null, "speak_streaming: the cached clip was queued")
	player.queue_free()
