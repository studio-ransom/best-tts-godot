extends SceneTree

## Everything a shipped game will do to this addon that a demo never does.
##
##   godot --rendering-driver vulkan --script res://tests/test_stress.gd
##
## The other tests ask "is the output correct". This one asks "does it survive":
## garbage text, unknown voices, absurd speeds, a hundred calls in a row, a
## player skipping dialogue, and the node being freed while the GPU is
## mid-utterance. Every case here is something a player can cause.
##
## Sections can be run alone with --only=abuse,cache,... — the soak is the slow
## one.

const TTSScript := preload("res://addons/best-tts/kokoro_tts.gd")
const AudioScript := preload("res://addons/best-tts/runtime/audio.gd")

## How many utterances the soak runs. A dialogue-heavy hour is well past this.
const SOAK_RUNS := 100
## Frame-time budget for anything that happens on the main thread. A 60 Hz game
## has 16.7 ms for everything; a TTS call may not eat a whole frame by itself.
const MAIN_THREAD_BUDGET_MS := 2.0

var _tts
var _failures := 0
var _only := PackedStringArray()


func _init() -> void:
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--only="):
			_only = arg.substr(7).split(",")

	_tts = TTSScript.new()
	_tts.preload_on_ready = false
	root.add_child(_tts)
	_tts.engine_failed.connect(func(reason: String):
		printerr("FAIL: engine did not start — " + reason)
		quit(1))

	var t0 := Time.get_ticks_msec()
	_tts.start()
	await _tts.engine_ready
	print("engine ready in %d ms\n" % (Time.get_ticks_msec() - t0))

	for section in ["abuse", "voices", "loudness", "concurrency", "cancel",
			"cache", "timings", "main_thread", "soak", "lifecycle"]:
		if _only.is_empty() or _only.has(section):
			await call("_" + section)

	print("\n" + "=".repeat(66))
	if _failures == 0:
		print("OK: survived everything")
	else:
		printerr("%d stress failure(s)" % _failures)
	quit(1 if _failures > 0 else 0)


func _fail(msg: String) -> void:
	printerr("  FAIL: " + msg)
	_failures += 1


func _section(name: String) -> void:
	print("--- %s ---" % name)


## Peak amplitude, sampled. Enough to tell audio from silence without spending
## a second of GDScript on a 300-sentence clip.
func _peak(clip: AudioStreamWAV) -> int:
	var n := clip.data.size() / 2
	var peak := 0
	var step := maxi(1, n / 4000)
	var i := 0
	while i < n:
		peak = maxi(peak, absi(clip.data.decode_s16(i * 2)))
		i += step
	return peak


func _secs(clip: AudioStreamWAV) -> float:
	return clip.data.size() / 2.0 / maxi(1, clip.mix_rate)


## RMS amplitude over a time window, 0..1.
func _rms(clip: AudioStreamWAV, from_s: float, to_s: float) -> float:
	var n := clip.data.size() / 2
	var a := clampi(int(from_s * clip.mix_rate), 0, n)
	var b := clampi(int(to_s * clip.mix_rate), a, n)
	if b <= a:
		return 0.0
	var sum := 0.0
	for i in range(a, b):
		var v := clip.data.decode_s16(i * 2) / 32768.0
		sum += v * v
	return sqrt(sum / float(b - a))


# --------------------------------------------------------------------- inputs

## Text a player, a localizer or a modder can actually produce. None of these
## may crash, hang, or come back as a clip that is silently wrong; a refusal
## with a reason is a perfectly good answer.
func _abuse() -> void:
	_section("hostile input")
	# {label, text, must_speak} — must_speak false means "clip or clean refusal,
	# both fine", true means "this is real text and has to produce audio".
	var cases := [
		["empty", "", false],
		["spaces only", "     ", false],
		["newlines only", "\n\n\t\n", false],
		["one letter", "a", true],
		["one punctuation", ".", false],
		["all punctuation", "!?.,;:-—…\"'()[]{}", false],
		["emoji", "Hello 👋 world 🎮 okay", true],
		["cjk (no g2p for the voice)", "こんにちは世界", false],
		["cyrillic", "Привет мир", false],
		["accents", "naïve café résumé jalapeño", true],
		# Built rather than written as escapes: GDScript refuses to parse a
		# source file that contains a literal NUL.
		["control chars", "a%sb%sc%sd" % [String.chr(0), String.chr(7),
				String.chr(27)], false],
		["url", "Visit https://example.com/x?y=1&z=2 now.", true],
		["file path", "Open C:\\Users\\pc\\save.dat please.", true],
		["markup left in", "<b>Bold</b> and [color=red]red[/color].", true],
		["numbers", "In 1999 I paid $12.50 for 3.14 kg at 4:30 pm.", true],
		["big number", "The population is 8102345678 people.", true],
		["negative and math", "It was -40 degrees, 5 x 3 = 15.", true],
		["repeated punctuation", "What?!?!?! No..... way,,,,, man!!!!!", true],
		["shouting", "STOP RIGHT THERE CRIMINAL SCUM", true],
		["no spaces", "thequickbrownfoxjumpsoverthelazydog", true],
		# 600 letters with nothing to break on. The chunker used to emit empty
		# runs here and fail the whole utterance.
		["one long word", "a".repeat(600), true],
		["long url", "https://" + "sub.".repeat(90) + "example.com", true],
		["very long text", "The quick brown fox. ".repeat(300), true],
		["single quote soup", "'''\"\"\"'''", false],
		["mixed script", "Hello 世界 привет café", true],
		["tabs in the middle", "Hello\tthere\tfriend", true],
		["nbsp", "Hello\u00a0there", true],
		["zero width", "He\u200bllo", true],
	]

	for c in cases:
		var label: String = c[0]
		var must: bool = c[2]
		var t0 := Time.get_ticks_usec()
		var req = _tts.say(c[1])
		await req.finished
		var ms := (Time.get_ticks_usec() - t0) / 1000.0

		if req.stream == null:
			if must:
				_fail("%s: refused real text — %s" % [label, req.error])
			elif req.error.strip_edges().is_empty():
				_fail("%s: returned null with no reason" % label)
			else:
				print("  %-28s refused: %s" % [label, req.error])
			continue

		if req.stream.data.is_empty():
			_fail("%s: produced an empty clip rather than refusing" % label)
		elif _peak(req.stream) < 200:
			# A clip that is all zeros is worse than no clip: the caller plays
			# silence and never learns anything went wrong.
			_fail("%s: produced %.2f s of near-silence"
					% [label, _secs(req.stream)])
		else:
			print("  %-28s %6.2f s  (%5.0f ms)"
					% [label, _secs(req.stream), ms])


# --------------------------------------------------------------------- voices

## Every bundled voice, plus the names someone will inevitably pass by mistake.
func _voices() -> void:
	_section("voices")
	var names: PackedStringArray = _tts.list_voices()
	var claimed := 0
	for v in names:
		if _tts.has_g2p(v):
			claimed += 1
	print("  %d voice packs, has_g2p() claims %d speak from text"
			% [names.size(), claimed])

	var spoke := 0
	var refused := 0
	for v in names:
		var req = _tts.say("Testing one two three.", v)
		await req.finished
		if req.stream == null:
			refused += 1
			# A voice with no G2P must say what to do instead of just failing.
			if not req.error.contains("speak_phonemes"):
				_fail("%s refused without saying what to do: %s"
						% [v, req.error])
			if _tts.has_g2p(v):
				_fail("has_g2p(%s) said yes but speak() refused" % v)
		elif _peak(req.stream) < 200:
			_fail("%s produced silence" % v)
		else:
			spoke += 1
			if not _tts.has_g2p(v):
				_fail("has_g2p(%s) said no but speak() worked" % v)
	print("  %d spoke, %d refused for lack of a G2P" % [spoke, refused])
	if spoke != claimed:
		_fail("has_g2p() disagrees with reality (%d vs %d)" % [claimed, spoke])

	for bad in ["  ", "nope", "af_heart.pt", "../../etc/passwd", "AF_HEART",
			"zz_nobody"]:
		var req = _tts.say("Hello.", bad)
		await req.finished
		if req.stream != null:
			_fail("voice '%s' was accepted" % bad)
		elif req.error.strip_edges().is_empty():
			_fail("voice '%s' failed with no reason" % bad)
	print("  bad voice names all refused with a reason")

	for ipa in ["hɛlˈO", "ʘǀǃǂ", "", "   ", "hɛlˈO".repeat(400)]:
		var req = _tts.say_phonemes(ipa, "jf_alpha")
		await req.finished
		if req.stream == null and req.error.strip_edges().is_empty():
			_fail("speak_phonemes('%s') failed with no reason"
					% ipa.substr(0, 20))
	print("  speak_phonemes survives unknown symbols")

	_section("speed")
	# Out of range is a caller bug; speed is a divisor in the duration
	# predictor, so 0 has to be clamped rather than used.
	for speed in [-1.0, 0.001, 0.5, 1.0, 2.0, 100.0]:
		var req = _tts.say("Speed check.", "af_heart", speed)
		await req.finished
		if req.stream == null:
			_fail("speed %.3f refused: %s" % [speed, req.error])
			continue
		print("  speed %-8.3f -> %.2f s (clamped to %.2f)"
				% [speed, _secs(req.stream), req.speed])
		if _peak(req.stream) < 200:
			_fail("speed %.3f produced silence" % speed)


# -------------------------------------------------------------------- loudness

## Kokoro's voice packs are not level-matched. A scene that switches speaker
## should not step the volume, and the player should not have to reach for the
## slider when an NPC hands off to another.
func _loudness() -> void:
	_section("loudness")
	_tts.cache_enabled = false
	# Deliberately not one of the lines the calibration measured: a per-voice
	# gain that only works on its own reference sentence is worthless.
	var line := "Careful now, the ice will not hold both of us."
	var sample := ["af_nova", "if_sara", "hf_alpha", "am_puck", "bm_lewis",
			"af_kore", "pm_santa", "zf_xiaoxiao", "af_heart", "bf_isabella",
			"am_eric", "jf_nezumi"]

	var before := await _spread(line, sample, false)
	var after := await _spread(line, sample, true)
	print("  spread across %d voices: %.1f dB raw -> %.1f dB normalized"
			% [sample.size(), before["spread"], after["spread"]])
	print("  quietest %s %.1f LUFS, loudest %s %.1f LUFS"
			% [after["min_voice"], after["min"], after["max_voice"],
			after["max"]])

	if before["spread"] < 6.0:
		_fail("the packs only differ by %.1f dB raw, so this test is not "
				% before["spread"] + "measuring what it claims to")
	if after["spread"] > 3.5:
		_fail("still %.1f dB apart after normalizing" % after["spread"])
	if after["spread"] >= before["spread"] * 0.5:
		_fail("normalizing barely helped (%.1f -> %.1f dB)"
				% [before["spread"], after["spread"]])

	# Nothing may clip: `pcm16.glsl` hard-clamps, so a gain that overshoots
	# turns a peak into a click rather than a loud sample.
	if after["peak"] >= 1.0:
		_fail("normalized output reached full scale (%.4f)" % after["peak"])
	print("  loudest sample after normalizing: %.3f of full scale"
			% after["peak"])
	_tts.cache_enabled = true
	await _tts.went_idle


func _spread(line: String, voices: Array, normalize: bool) -> Dictionary:
	_tts.normalize_loudness = normalize
	var lo := INF
	var hi := -INF
	var lo_v := ""
	var hi_v := ""
	var peak := 0.0
	for v in voices:
		var req = _tts.say(line, v) if _tts.has_g2p(v) \
				else _tts.say_phonemes("kˈɛɹfəl nˈaU, ðə ˈIs wɪl nˈɑt hˈOld "
						+ "bˈOθ ʌv ˈʌs.", v)
		await req.finished
		if req.stream == null:
			_fail("%s: %s" % [v, req.error])
			continue
		var m: Dictionary = AudioScript.measure_loudness(req.stream)
		peak = maxf(peak, m["peak"])
		if m["lufs"] < lo:
			lo = m["lufs"]
			lo_v = v
		if m["lufs"] > hi:
			hi = m["lufs"]
			hi_v = v
	return {"spread": hi - lo, "min": lo, "max": hi, "min_voice": lo_v,
			"max_voice": hi_v, "peak": peak}


# ---------------------------------------------------------------- concurrency

## Ten NPCs speaking on the same frame. Every caller must get its own answer
## back, and the answers must not be swapped or dropped.
func _concurrency() -> void:
	_section("concurrency")
	_tts.cache_enabled = false
	var lines := ["One.", "Two two.", "Three three three.", "Four.", "Five.",
			"Six.", "Seven.", "Eight.", "Nine.", "Ten."]
	var reqs := []
	var t0 := Time.get_ticks_msec()
	for l in lines:
		reqs.append(_tts.say(l))

	var ok := 0
	for i in reqs.size():
		var r = reqs[i]
		if not r.is_done():
			await r.finished
		if r.stream == null:
			_fail("job %d of %d came back empty: %s" % [i, reqs.size(), r.error])
		else:
			ok += 1
	print("  %d/%d concurrent requests resolved in %d ms"
			% [ok, reqs.size(), Time.get_ticks_msec() - t0])

	# Longer text first, then a short one: the short answer must not arrive
	# carrying the long clip's audio.
	var long_req = _tts.say("The quick brown fox jumps over the lazy dog. "
			.repeat(6))
	var short_req = _tts.say("Hi.")
	await long_req.finished
	await short_req.finished
	if long_req.stream == null or short_req.stream == null:
		_fail("ordered pair did not both resolve")
	elif short_req.stream.data.size() >= long_req.stream.data.size():
		_fail("the short request came back with the long one's audio")
	else:
		print("  queued requests keep their own results")

	# A HIGH line must overtake a backlog of ambient chatter.
	var low := []
	for i in 8:
		low.append(_tts.say("Ambient line number %d." % i, "", 0.0,
				TTSScript.Priority.LOW))
	var urgent = _tts.say("Look out!", "", 0.0, TTSScript.Priority.HIGH)
	var order := []
	for r in low:
		r.finished.connect(func(_c): order.append("low"))
	urgent.finished.connect(func(_c): order.append("high"))
	await urgent.finished
	var ahead := order.count("low")
	print("  HIGH landed after %d of 8 queued LOW lines" % ahead)
	# One LOW may already have been on the GPU when the HIGH arrived.
	if ahead > 1:
		_fail("priority ignored: %d low-priority lines went first" % ahead)
	_tts.cancel_all()
	await _tts.went_idle

	# The backlog must be bounded, or a spammy trigger builds minutes of stale
	# audio nobody will ever hear.
	_tts.max_queued = 8
	var many := []
	for i in 40:
		many.append(_tts.say("Overflow line %d." % i, "", 0.0,
				TTSScript.Priority.LOW))
	print("  submitted 40 with max_queued=8, queue holds %d"
			% _tts.queue_size())
	if _tts.queue_size() > 8:
		_fail("queue grew to %d past max_queued" % _tts.queue_size())
	_tts.cancel_all()
	for r in many:
		if not r.is_done():
			await r.finished
	var dropped := 0
	for r in many:
		if r.error.begins_with("dropped"):
			dropped += 1
	print("  %d were dropped with a reason, all 40 resolved" % dropped)
	if dropped == 0:
		_fail("nothing was dropped, so the queue is not actually bounded")
	_tts.max_queued = 32
	_tts.cache_enabled = true
	await _tts.went_idle


# ----------------------------------------------------------------- cancelling

## A player skips a line, walks out of trigger range, or opens the pause menu.
## Whatever is queued has to stop, and it has to stop soon enough to matter.
func _cancel() -> void:
	_section("cancellation")
	_tts.cache_enabled = false
	var text := "The quick brown fox jumps over the lazy dog. ".repeat(8)
	var reqs := []
	for i in 6:
		reqs.append(_tts.say(text))
	# Let one actually reach the GPU, so this tests aborting a run rather than
	# just emptying the queue.
	await create_timer(0.4).timeout

	# The one already on the GPU is the interesting one: the rest are just
	# array removals. Time it on its own so the number means something.
	# Boxed in an array: GDScript lambdas capture locals by value, so writing to
	# a plain `var` from inside one updates a copy and reads back unchanged.
	var abort := [-1]
	var t0 := Time.get_ticks_msec()
	reqs[0].finished.connect(func(_c): abort[0] = Time.get_ticks_msec() - t0)
	var dropped: int = _tts.cancel_all()
	for r in reqs:
		if not r.is_done():
			await r.finished
	var ms := Time.get_ticks_msec() - t0
	print("  cancel_all() hit %d of %d; the running one stopped in %d ms, "
			% [dropped, reqs.size(), abort[0]] + "all six in %d ms" % ms)
	print("  %s" % reqs[0].error)

	var still_speaking := 0
	for r in reqs:
		if r.stream != null:
			still_speaking += 1
	if still_speaking > 0:
		_fail("%d cancelled requests still produced audio" % still_speaking)
	if _tts.is_busy():
		_fail("still busy after cancel_all()")
	# Six eight-sentence passages is ~22 s of synthesis; anything close to that
	# means the cancel only emptied the queue and let the GPU run to the end.
	# The bound is one GPU submission — see `gpu_submit_nodes`, measured at
	# 348 ms worst case — plus a frame of deferred delivery.
	if ms > 700:
		_fail("cancel took %d ms to take effect" % ms)

	# Cancelling one request must not touch the others.
	var a = _tts.say("The first line of three.")
	var b = _tts.say("The second line of three.")
	var c = _tts.say("The third line of three.")
	b.cancel()
	await a.finished
	await b.finished
	await c.finished
	if a.stream == null or c.stream == null:
		_fail("cancelling one request took its neighbours with it")
	elif b.stream != null:
		_fail("a cancelled request still produced audio")
	else:
		print("  cancelling one request leaves the rest alone")

	# Cancelling after the fact, and twice, must be harmless.
	a.cancel()
	a.cancel()
	if a.stream == null:
		_fail("cancelling a finished request wiped its result")

	var clip: AudioStreamWAV = await _tts.speak("Still here.")
	if clip == null:
		_fail("engine unusable after a cancel: %s" % _tts.get_error())
	else:
		print("  engine still works after cancelling")
	_tts.cache_enabled = true
	await _tts.went_idle


# ---------------------------------------------------------------------- cache

## Games repeat themselves. The second time a line is asked for it should not
## touch the GPU at all.
func _cache() -> void:
	_section("cache")
	_tts.cache_enabled = true
	_tts.cache_on_disk = false
	_tts.clear_cache(true)

	var line := "I have wares if you have coin."
	var t0 := Time.get_ticks_usec()
	var first = _tts.say(line)
	await first.finished
	var cold := (Time.get_ticks_usec() - t0) / 1000.0

	t0 = Time.get_ticks_usec()
	var second = _tts.say(line)
	await second.finished
	var warm := (Time.get_ticks_usec() - t0) / 1000.0

	print("  cold %.0f ms, warm %.2f ms (%.0fx faster)"
			% [cold, warm, cold / maxf(0.01, warm)])
	if not second.from_cache:
		_fail("the second identical line was synthesized again")
	elif second.stream == null or second.stream.data != first.stream.data:
		_fail("the cached clip differs from the one it replaced")
	if warm > 5.0:
		_fail("a cache hit took %.1f ms" % warm)

	# Anything that changes the samples has to miss.
	var other_voice = _tts.say(line, "am_michael")
	await other_voice.finished
	if other_voice.from_cache:
		_fail("a different voice was served from cache")
	var other_speed = _tts.say(line, "", 1.5)
	await other_speed.finished
	if other_speed.from_cache:
		_fail("a different speed was served from cache")
	# So does flipping the filter, or the cache serves audio the settings say
	# should not exist.
	_tts.remove_vocoder_tone = false
	var unfiltered = _tts.say(line)
	await unfiltered.finished
	if unfiltered.from_cache:
		_fail("the unfiltered clip was served from the filtered cache")
	_tts.remove_vocoder_tone = true
	print("  voice, speed and filter changes all miss correctly")

	# Bounded, and evicting rather than growing.
	_tts.clear_cache(true)
	_tts._cache.memory_budget = 1 << 20
	for i in 12:
		var req = _tts.say("Cache pressure line number %d, a fair bit of it." % i)
		await req.finished
	var stats: Dictionary = _tts.cache_stats()
	print("  under a 1 MB budget: %d entries, %.2f MB, %d evictions"
			% [stats["entries"], stats["memory_bytes"] / 1048576.0,
			stats["evictions"]])
	if stats["memory_bytes"] > (1 << 20) * 1.5:
		_fail("the cache blew past its budget (%.2f MB)"
				% (stats["memory_bytes"] / 1048576.0))
	if stats["evictions"] == 0:
		_fail("nothing was evicted, so the budget is not enforced")
	_tts._cache.memory_budget = 32 << 20

	# The disk tier: survives losing the memory tier, which is what surviving a
	# restart amounts to.
	_tts.clear_cache(true)
	_tts.cache_on_disk = true
	_tts._cache.disk_dir = _tts.cache_dir
	var disk_line := "This line should come back from disk."
	var w = _tts.say(disk_line)
	await w.finished
	_tts.clear_cache(false)
	t0 = Time.get_ticks_usec()
	var r2 = _tts.say(disk_line)
	await r2.finished
	var disk_ms := (Time.get_ticks_usec() - t0) / 1000.0
	var s2: Dictionary = _tts.cache_stats()
	print("  disk hit in %.1f ms, %d file(s), %.0f KB on disk"
			% [disk_ms, s2["disk_files"], s2["disk_bytes"] / 1024.0])
	if not r2.from_cache:
		_fail("the disk tier did not serve a line it had written")
	elif r2.stream.data != w.stream.data:
		_fail("the clip read back from disk is not the one written")
	if disk_ms > 100.0:
		_fail("a disk hit took %.0f ms" % disk_ms)

	_tts.clear_cache(true)
	_tts.cache_on_disk = false
	_tts._cache.disk_dir = ""

	# precache() is the loading-screen path: warm the barks, then they are free.
	var barks := PackedStringArray(["Halt!", "Over here!", "Take cover!"])
	var warmed: Array = _tts.precache(barks)
	for r in warmed:
		if not r.is_done():
			await r.finished
	var instant := 0
	for b in barks:
		var req = _tts.say(b)
		await req.finished
		if req.from_cache:
			instant += 1
	print("  precache() made %d/%d barks instant" % [instant, barks.size()])
	if instant != barks.size():
		_fail("precache() did not warm every line")
	await _tts.went_idle


# -------------------------------------------------------------------- timings

## Subtitle highlighting and lip sync both need to know when each word lands.
func _timings() -> void:
	_section("timings")
	_tts.compute_timings = true
	_tts.clear_cache(true)

	var text := "The old bridge collapsed under the weight of the caravan."
	var req = _tts.say(text)
	await req.finished
	if req.stream == null:
		_fail("timed request failed: %s" % req.error)
		_tts.compute_timings = false
		return

	var words: Array = req.timings.get("words", [])
	var phones: Array = req.timings.get("phonemes", [])
	var dur := _secs(req.stream)
	print("  %.2f s, %d words, %d phonemes" % [dur, words.size(), phones.size()])
	if words.is_empty():
		_fail("no word timings were produced")
		_tts.compute_timings = false
		return

	# One entry per spoken word, in order.
	var spoken := text.replace(".", "").split(" ", false).size()
	if absi(words.size() - spoken) > 1:
		_fail("%d word timings for %d words" % [words.size(), spoken])

	var prev := -1.0
	for w in words:
		if w["start"] < prev - 0.001:
			_fail("'%s' starts before the word before it" % w["text"])
			break
		if w["end"] < w["start"]:
			_fail("'%s' ends before it starts" % w["text"])
			break
		if w["start"] < -0.001 or w["end"] > dur + 0.05:
			_fail("'%s' at %.2f-%.2f falls outside a %.2f s clip"
					% [w["text"], w["start"], w["end"], dur])
			break
		prev = w["end"]
	print("  first '%s' at %.2f s, last '%s' ends %.2f s (clip %.2f s)"
			% [words[0]["text"], words[0]["start"], words[-1]["text"],
			words[-1]["end"], dur])

	# The predicted durations must add up to the audio actually produced. If
	# they drift, a subtitle highlight walks off the end of a long line.
	if absf(float(words[-1]["end"]) - dur) > 0.35:
		_fail("timings cover %.2f s of a %.2f s clip"
				% [words[-1]["end"], dur])

	# The words have to be the ones that were said, not the phonemes.
	if not str(words[0]["text"]).to_lower().begins_with("the"):
		_fail("first word came back as '%s'" % words[0]["text"])

	# Plausible ordering is not alignment. Measure the audio: the stretch the
	# timings call speech has to be loud, and the stretch outside it quiet. A
	# constant offset would sail through every check above and put every
	# subtitle a syllable early.
	var inside := _rms(req.stream, words[0]["start"], words[-1]["end"])
	var before := _rms(req.stream, 0.0, words[0]["start"])
	var after := _rms(req.stream, words[-1]["end"], dur)
	var lead_db := 20.0 * log(inside / maxf(before, 1e-9)) / log(10.0)
	print("  speech %.0f dB above the lead-in, %.0f dB above the tail"
			% [lead_db, 20.0 * log(inside / maxf(after, 1e-9)) / log(10.0)])
	if before > 0.0 and lead_db < 12.0:
		_fail("the span before the first word is only %.0f dB down — the "
				% lead_db + "timings are offset from the audio")

	# Shift everything by 150 ms and the same measurement must get worse, or
	# the check above is not sensitive enough to catch a real drift.
	var shifted := _rms(req.stream, 0.0, maxf(0.0, words[0]["start"] + 0.15))
	if before > 0.0 and shifted <= before * 1.5:
		_fail("a 150 ms shift does not change the lead-in level, so this "
				+ "check cannot detect a misaligned timing")

	# Long text is chunked; each chunk's times must be offset by the audio
	# before it rather than restarting at zero.
	var joined := ""
	for i in 4:
		joined += ("Sentence number %d in a passage long enough to be split "
				+ "into several chunks and stitched back together. ") % (i + 1)
	var lreq = _tts.say(joined)
	await lreq.finished
	var lw: Array = lreq.timings.get("words", [])
	var ldur := _secs(lreq.stream)
	var monotonic := true
	for i in range(1, lw.size()):
		if lw[i]["start"] < lw[i - 1]["start"] - 0.001:
			monotonic = false
			break
	print("  chunked: %d words over %.2f s, last ends %.2f s"
			% [lw.size(), ldur, lw[-1]["end"] if not lw.is_empty() else 0.0])
	if not monotonic:
		_fail("word times restart at each chunk boundary")
	if lw.is_empty() or absf(float(lw[-1]["end"]) - ldur) > 0.6:
		_fail("chunked timings do not reach the end of the clip")

	# Cached clips must bring their timings with them.
	var again = _tts.say(text)
	await again.finished
	if not again.from_cache:
		_fail("expected a cache hit")
	elif again.timings.get("words", []).size() != words.size():
		_fail("a cached clip came back without its timings")
	else:
		print("  cached clips keep their timings")

	_tts.compute_timings = false
	await _tts.went_idle


# ------------------------------------------------------------ main-thread cost

## Whatever `say()` does before handing off to the worker happens on the frame
## the caller is on. That is the part that shows up as a hitch.
func _main_thread() -> void:
	_section("main-thread cost")
	_tts.clear_cache(true)
	var cases := [
		["one bark", "Look out!"],
		["a line of dialogue", "I have not seen anything like this since the "
				+ "siege, and I do not care to see it again."],
		["a full page", "The quick brown fox jumps over the lazy dog. "
				.repeat(120)],
	]
	for c in cases:
		var t0 := Time.get_ticks_usec()
		var req = _tts.say(c[1])
		var ms := (Time.get_ticks_usec() - t0) / 1000.0
		print("  %-22s %6.3f ms on the calling thread" % [c[0], ms])
		if ms > MAIN_THREAD_BUDGET_MS:
			_fail("%s costs %.2f ms of frame time (budget %.1f)"
					% [c[0], ms, MAIN_THREAD_BUDGET_MS])
		await req.finished

	# And a cache hit, which is the common case once a game is warm.
	var t1 := Time.get_ticks_usec()
	var hit = _tts.say(cases[0][1])
	var hit_ms := (Time.get_ticks_usec() - t1) / 1000.0
	await hit.finished
	print("  %-22s %6.3f ms on the calling thread" % ["a cache hit", hit_ms])
	if hit_ms > MAIN_THREAD_BUDGET_MS:
		_fail("a cache hit costs %.2f ms of frame time" % hit_ms)


# ----------------------------------------------------------------------- soak

## An hour of dialogue, compressed. Watches for the three things that only show
## up after a while: memory creep, slowdown, and output that quietly changes.
func _soak() -> void:
	_section("soak (%d utterances)" % SOAK_RUNS)
	# Caching would turn this into a dictionary benchmark. The point is the GPU
	# path, so keep it off.
	_tts.cache_enabled = false
	var lines := [
		"Watch your step.", "Did you hear that?", "I have wares if you have coin.",
		"Not now, I'm busy.", "Halt! Who goes there?", "It's quiet. Too quiet.",
		"Twelve gold, take it or leave it.", "Another day, another arrow.",
	]
	var voices := ["af_heart", "am_michael", "bf_emma", "bm_george"]

	var first_ms := 0.0
	var first_size := 0
	var slowest := 0.0
	var start := Time.get_ticks_msec()
	# Measured after a warm-up run so the pool is at its steady size; comparing
	# against a cold device just measures the pool filling up.
	var warm = _tts.say(lines[0])
	await warm.finished
	await _tts.went_idle
	var device_start := _device_mb()

	for i in SOAK_RUNS:
		var t0 := Time.get_ticks_usec()
		var req = _tts.say(lines[i % lines.size()],
				voices[(i / lines.size()) % voices.size()])
		await req.finished
		var ms := (Time.get_ticks_usec() - t0) / 1000.0
		if req.stream == null:
			_fail("run %d failed: %s" % [i, req.error])
			_tts.cache_enabled = true
			return
		slowest = maxf(slowest, ms)
		if i == 0:
			first_ms = ms
			first_size = req.stream.data.size()
		# The same line in the same voice must give the same bytes every time.
		# Drift here is state leaking between runs, and it is the kind of bug
		# that only ever reproduces on a player's machine.
		if i == lines.size() * voices.size() \
				and req.stream.data.size() != first_size:
			_fail("run %d of the first line changed length (%d -> %d bytes)"
					% [i, first_size, req.stream.data.size()])
		if i % 25 == 24:
			print("  %3d runs   %6.1f ms last   device %6.1f MB"
					% [i + 1, ms, _device_mb()])

	var total := (Time.get_ticks_msec() - start) / 1000.0
	await _tts.went_idle
	var device_end := _device_mb()
	print("  %d utterances in %.1f s (%.0f ms each)"
			% [SOAK_RUNS, total, total * 1000.0 / SOAK_RUNS])
	print("  first %.0f ms, slowest %.0f ms" % [first_ms, slowest])
	print("  device %.1f MB -> %.1f MB (%+.2f MB)"
			% [device_start, device_end, device_end - device_start])

	# 100 runs at the old 33 KB-per-run leak was 3.3 MB. Nothing should move at
	# all now that the pool is warm.
	if device_end > device_start + 1.0:
		_fail("device grew %.2f MB over %d warm runs"
				% [device_end - device_start, SOAK_RUNS])
	if slowest > first_ms * 3.0 and slowest > 200.0:
		_fail("slowest run %.0f ms vs first %.0f ms" % [slowest, first_ms])
	_tts.cache_enabled = true


func _device_mb() -> float:
	return _tts.gpu_memory_bytes() / 1048576.0


# ------------------------------------------------------------------ lifecycle

## Scene changes, quit-during-speech, and a second engine after the first is
## gone. These are the paths that crash on shutdown rather than in play.
func _lifecycle() -> void:
	_section("lifecycle")

	var tts2 = TTSScript.new()
	tts2.preload_on_ready = false
	root.add_child(tts2)
	tts2.start()
	await tts2.engine_ready
	print("  second engine started alongside the first")

	# Freeing the node while the GPU is mid-utterance. The engine thread has to
	# be joined before the RenderingDevice goes, the caller awaiting a clip must
	# be released rather than left hanging, and none of it may cost a frame.
	var req = tts2.say("The quick brown fox jumps over the lazy dog. ".repeat(8))
	await create_timer(0.4).timeout
	# Boxed: GDScript lambdas capture locals by value.
	var released := [false]
	req.finished.connect(func(_s): released[0] = true)

	var t0 := Time.get_ticks_msec()
	tts2.queue_free()
	# queue_free defers to the end of the frame, which is where _exit_tree runs.
	await process_frame
	await process_frame
	var teardown := Time.get_ticks_msec() - t0
	print("  freed mid-utterance in %d ms" % teardown)
	if not released[0]:
		_fail("freeing the node left a caller awaiting a clip forever")
	if not req.is_done():
		_fail("the in-flight request never resolved")
	# Aborting the GPU submission in flight, then tearing the device down. Both
	# are real work; what matters is that it is not the several seconds it took
	# when teardown waited for the whole utterance.
	if teardown > 800:
		_fail("teardown blocked the main thread for %d ms" % teardown)

	# Idle teardown is the common case — a scene change while nothing is being
	# said — and has to be much cheaper than the busy one.
	var tts3 = TTSScript.new()
	tts3.preload_on_ready = false
	root.add_child(tts3)
	tts3.start()
	await tts3.engine_ready
	var t1 := Time.get_ticks_msec()
	tts3.queue_free()
	await process_frame
	await process_frame
	var idle_teardown := Time.get_ticks_msec() - t1
	print("  freed while idle in %d ms" % idle_teardown)
	if idle_teardown > 250:
		_fail("idle teardown cost %d ms" % idle_teardown)

	# The first engine must be untouched by the second one's teardown.
	var clip: AudioStreamWAV = await _tts.speak("The first engine is fine.")
	if clip == null:
		_fail("first engine broke when the second was freed: %s"
				% _tts.get_error())
	else:
		print("  first engine unaffected")

	# speak() before start(), and start() twice.
	_tts.start()
	var early = TTSScript.new()
	early.preload_on_ready = false
	root.add_child(early)
	var early_clip: AudioStreamWAV = await early.speak("Before loading.")
	if early_clip == null:
		_fail("speak() before start() did not wait for the engine: %s"
				% early.get_error())
	else:
		print("  speak() before start() waits for the engine")

	# Explicit shutdown, then a request afterwards: refused, not crashed.
	early.shutdown()
	var after = early.say("After shutdown.")
	await after.finished
	if after.stream != null:
		_fail("a request after shutdown() produced audio")
	elif after.error.strip_edges().is_empty():
		_fail("a request after shutdown() failed with no reason")
	else:
		print("  requests after shutdown() are refused: %s" % after.error)
	early.shutdown()  # twice must be harmless
	early.queue_free()
	await process_frame
