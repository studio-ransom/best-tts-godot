extends SceneTree

## Does GPU memory plateau, or does it climb with every utterance?
##
##   godot --rendering-driver vulkan --script res://tests/test_memory.gd
##
## Synthesizes a ladder of lengths twice on one engine. The first pass sets the
## high-water mark; the second must not raise it. Growth that tracks the number
## of *runs* rather than the longest run is a leak.
##
## Then sweeps `chunk_phonemes`, because that — not the length of the text — is
## what sets peak memory once the pooling works.
##
## Every measurement waits for `went_idle`: the worker trims after the job
## signals completion, so sampling straight after `await speak()` reads the
## pool mid-edit and reports nonsense.

const TTSScript := preload("res://addons/best-tts/kokoro_tts.gd")

## The second pass may allocate a little — bucketing is not exact — but a real
## leak shows up as tens of percent, not single digits.
const REGROWTH_TOL := 0.10

var _tts
var _failures := 0
var _sentence := "The quick brown fox jumps over the lazy dog. "


func _init() -> void:
	_tts = TTSScript.new()
	_tts.preload_on_ready = false
	_tts.chunk_phonemes = 510
	# Repeated text would come straight back from the cache and this
	# would measure a dictionary lookup instead of the GPU path.
	_tts.cache_enabled = false
	root.add_child(_tts)

	_tts.engine_failed.connect(func(reason: String):
		printerr("FAIL: " + reason)
		quit(1))

	_tts.start()
	await _tts.engine_ready
	var weights := _device_bytes()
	print("weights on the device: %.1f MB\n" % (weights / 1048576.0))

	var pass1 := await _ladder("pass 1")
	var pass2 := await _ladder("pass 2")
	await _sweep_chunks()

	var idle := _device_bytes()
	print("\nweights            %7.1f MB" % (weights / 1048576.0))
	print("peak pass 1        %7.1f MB  (activations)" % (pass1 / 1048576.0))
	print("peak pass 2        %7.1f MB  (activations)" % (pass2 / 1048576.0))
	print("idle on device     %7.1f MB  (everything)" % (idle / 1048576.0))

	var growth := float(pass2 - pass1) / maxf(1.0, float(pass1))
	print("regrowth on pass 2 %+7.1f %%" % (growth * 100.0))
	if growth > REGROWTH_TOL:
		printerr("FAIL: memory still climbs with each run (%+.1f%%)"
				% (growth * 100.0))
		_failures += 1

	var budget: int = _tts.gpu_memory_budget_mb * 1048576
	if idle > weights + budget * 2:
		printerr("FAIL: idle footprint %.1f MB exceeds weights + 2x budget"
				% (idle / 1048576.0))
		_failures += 1

	if _failures == 0:
		print("\nOK: GPU memory plateaus")
	quit(1 if _failures > 0 else 0)


## Runs the ladder of lengths and returns the activation high-water mark.
func _ladder(label: String) -> int:
	print("--- %s ---" % label)
	_tts._executor.reset_peak()
	for reps in [1, 2, 4, 8, 12]:
		var text := ""
		for i in reps:
			text += _sentence
		var t0 := Time.get_ticks_msec()
		var clip: AudioStreamWAV = await _tts.speak(text)
		var ms := Time.get_ticks_msec() - t0
		if clip == null:
			printerr("FAIL: %s" % _tts.get_error())
			_failures += 1
			return 0
		await _tts.went_idle
		_report("%2d sentences, %5.1f s audio, %4d ms"
				% [reps, clip.data.size() / 2.0 / clip.mix_rate, ms])
	var peak: int = _tts._executor.peak_live_bytes
	print("  peak activations this pass: %.1f MB\n" % (peak / 1048576.0))
	return peak


## Peak memory is set by the longest single pass, so this is the knob that
## bounds it. Same text every time — only the chunk size changes.
func _sweep_chunks() -> void:
	print("--- peak vs chunk_phonemes (same 12-sentence text) ---")
	var text := ""
	for i in 12:
		text += _sentence
	for limit in [510, 320, 160, 96]:
		_tts.chunk_phonemes = limit
		_tts._executor.reset_peak()
		var t0 := Time.get_ticks_msec()
		var clip: AudioStreamWAV = await _tts.speak(text)
		var ms := Time.get_ticks_msec() - t0
		if clip == null:
			printerr("FAIL: %s" % _tts.get_error())
			_failures += 1
			return
		await _tts.went_idle
		var secs := clip.data.size() / 2.0 / clip.mix_rate
		print("  chunk %3d  peak %7.1f MB   %5.1f s audio in %4d ms (%.1fx)"
				% [limit, _tts._executor.peak_live_bytes / 1048576.0, secs, ms,
				secs * 1000.0 / maxf(1.0, ms)])
	_tts.chunk_phonemes = 510


## Total bytes the RenderingDevice is holding, weights included.
func _device_bytes() -> int:
	return _tts.gpu_memory_bytes() if _tts._gpu != null else 0


func _report(label: String) -> void:
	var ex = _tts._executor
	var gpu = _tts._gpu
	print("  %-38s device %7.1f MB   pooled %6.1f MB (%3d sizes)   sets %5d"
			% [label, _device_bytes() / 1048576.0,
			ex.pooled_bytes() / 1048576.0, ex._pool.size(),
			gpu._uniform_sets.size()])
