@tool
class_name KokoroTTS
extends Node

## Neural text-to-speech, running entirely inside Godot.
##
##     @onready var tts: KokoroTTS = $KokoroTTS
##
##     func _ready() -> void:
##         var clip := await tts.speak("Hello from Godot.", "af_heart")
##         $AudioStreamPlayer.stream = clip
##         $AudioStreamPlayer.play()
##
## `await` is the shape to reach for in a cutscene. Everywhere else — barks,
## reactions, anything a player can interrupt — use `say()`, which returns a
## [Request] straight away:
##
##     var r := tts.say("Did you hear that?", "am_michael")
##     r.finished.connect(func(clip): $Voice.stream = clip; $Voice.play())
##     ...
##     r.cancel()   # player walked away
##
## The model, the GPU device, text-to-phoneme conversion and every synthesis
## run live on a worker thread, so nothing here costs measurable frame time.
## Repeated lines come back from a cache instead of being synthesized again.
##
## Requires the Forward+ or Mobile renderer — Compatibility has no
## RenderingDevice and therefore cannot run compute shaders.

const GPUScript := preload("res://addons/best-tts/runtime/gpu.gd")
const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")
const ExecutorScript := preload("res://addons/best-tts/runtime/executor.gd")
const VoiceLoaderScript := preload("res://addons/best-tts/runtime/voice_loader.gd")
const TokenizerScript := preload("res://addons/best-tts/runtime/tokenizer.gd")
const AudioScript := preload("res://addons/best-tts/runtime/audio.gd")
const CacheScript := preload("res://addons/best-tts/runtime/clip_cache.gd")
const G2PScript := preload("res://addons/best-tts/g2p/g2p.gd")

const MODEL_PATH := "res://addons/best-tts/assets/model.kpk"
const LEVELS_PATH := "res://addons/best-tts/assets/voice_levels.json"
const SAMPLE_RATE := 24000
## Loudest sample a normalized clip may reach. `pcm16.glsl` hard-clamps, so a
## gain that overshoots turns a peak into a click; staying under 1.0 costs a
## tenth of a decibel and cannot.
const PEAK_CEILING := 0.99

## Queue position. A HIGH line jumps everything still waiting, which is what
## you want when the player triggers something while ambient chatter is backed
## up. LOW is for work you would rather have than not — see `precache()`.
enum Priority { LOW = 0, NORMAL = 1, HIGH = 2 }

## Emitted once the model is loaded and the first `speak()` can run.
signal engine_ready()
## Emitted if the engine could not start; `reason` is safe to show a user.
signal engine_failed(reason: String)
## Emitted per chunk while a long utterance is being synthesized. With several
## requests in flight this says nothing about *which* one — connect to a
## [Request]'s own `chunk_ready` when that matters.
signal synthesis_progress(chunk: int, total: int)
## Emitted once the queue has drained and idle GPU memory has been released.
## This lands after the last request resolves, not with it.
signal went_idle()

@export var default_voice := "af_heart"
@export_range(0.5, 2.0, 0.05) var default_speed := 1.0
## Start loading the model as soon as the node enters the tree.
@export var preload_on_ready := true

## Longest phoneme run per synthesis pass. Text above this is split at
## punctuation and the clips are joined.
##
## This is also what sets peak GPU memory, since activations scale with the
## length of a single pass. Measured on a 33 s passage: 510 needs ~1.1 GB,
## 160 needs ~0.7 GB, 96 needs ~0.55 GB, and throughput only slips from 8.0x
## to 7.0x realtime. 160 is a long clause — around 11 s of speech — so the
## prosody cost of splitting there is small.
@export_range(64, 510, 1) var chunk_phonemes := 160

## How much GPU memory to keep parked for reuse once synthesis goes idle.
## Anything above this is handed back to the driver between utterances.
@export_range(0, 1024, 16) var gpu_memory_budget_mb := 64

## Longest backlog to hold. Past this the lowest-priority waiting requests are
## dropped, newest first, and fail with a reason. A queue is a latency budget:
## a game that submits faster than the GPU can synthesize would otherwise build
## a backlog that is minutes stale by the time it plays.
@export_range(1, 256, 1) var max_queued := 32

## Notch out the tone Kokoro's iSTFT leaves at 4800 Hz and 9600 Hz.
##
## The model overlap-adds with hop 5 and its window does not sum flat, so both
## frequencies sit about 25 dB above the surrounding noise floor and are heard
## as a ring behind the speech. The numpy reference has the identical tone, so
## this is the model's, not the runtime's. Turn it off to hear exactly what the
## graph produces; leave it on to hear something cleaner.
@export var remove_vocoder_tone := true

## Level-match the voice packs against each other.
##
## Kokoro's packs are not mastered to a common level, so switching speaker
## mid-scene steps the volume. This applies one constant gain per voice, from
## the table `tests/test_calibrate_voices.gd` measures — constant so that the
## variation *within* a voice survives: a short exclamation should still be
## louder than a long calm line.
##
## Only the gain differs; nothing is compressed and nothing can clip.
@export var normalize_loudness := true

## Loudness to match, in LUFS. 0 means "whatever the table says", which is the
## median bundled voice — chosen so corrections are small in both directions
## and none of them run out of peak headroom.
@export_range(-40.0, 0.0, 0.5) var loudness_target_lufs := 0.0

@export_group("Cache")
## Serve repeated lines from memory instead of synthesizing them again. A cache
## hit resolves on the calling thread in the same frame.
@export var cache_enabled := true
@export_range(0, 512, 4) var cache_memory_mb := 32
## Also keep clips under `cache_dir`, so a line survives a restart and is only
## ever synthesized on the player's first playthrough.
@export var cache_on_disk := false
@export_range(0, 4096, 16) var cache_disk_mb := 128
@export var cache_dir := "user://kokoro_cache"

@export_group("Advanced")
## How much of the graph is submitted to the GPU at a time, in nodes.
##
## Buys cancellation responsiveness with throughput. Recording the graph is
## fast and running it is not, so with one big submission a cancel cannot take
## effect until the current chunk finishes on the GPU. Measured on a 25 s
## passage: 0 (submit once) is 3650 ms with a 1149 ms worst-case cancel; 1024
## is 3650 ms / 532 ms; **512 is 3734 ms / 348 ms**; 256 is 3817 ms / 248 ms.
## Below 256 the sync bubbles cost more than they save.
@export_range(0, 2048, 64) var gpu_submit_nodes := 512

@export_group("Timings")
## Report where each word and phoneme lands in the clip, for subtitle
## highlighting and viseme-driven lip sync. The duration predictor computes
## this anyway, so the only cost is assembling the arrays.
@export var compute_timings := false

var _thread: Thread
var _sem: Semaphore
var _mutex: Mutex
var _queue: Array = []
## Requests submitted and not yet resolved, so shutdown can release every
## caller instead of leaving an `await` that never returns.
var _outstanding: Array = []
var _active = null
## Set when anything is submitted, cleared when `went_idle` fires, so the
## signal marks a busy-to-idle transition rather than repeating on every drain.
var _idle_pending := false
## True once a request has actually reached the worker. When every request in a
## batch came from the cache the worker never wakes, so nothing would run the
## trim that normally emits `went_idle` — and `await went_idle` after a fully
## warm `precache()` would hang forever.
var _worker_used := false
var _quit := false
var _trim_now := false

var _gpu
var _model
var _executor
var _voices
var _tokenizer
var _g2p
var _cache
## Notch coefficients, uploaded once; see `remove_vocoder_tone`.
var _fir_rid := RID()
var _fir_len := 0
## Output samples per predicted duration frame; read from the packed model.
var _samples_per_frame := 600
## Measured per-voice loudness and peak; see `normalize_loudness`.
var _levels: Dictionary = {}

var _state := State.IDLE
var _error := ""

enum State { IDLE, LOADING, READY, FAILED, SHUTDOWN }


## A single piece of speech, in flight or finished.
##
## Returned by `say()` and friends. `finished` always fires exactly once —
## on success, on failure, on cancellation, and on engine shutdown — so a
## coroutine awaiting it can never be stranded.
class Request extends RefCounted:
	## The clip, or null if this did not produce audio. Check `error`.
	signal finished(clip: AudioStreamWAV)
	## Fires per chunk of a long utterance, before `finished`. Lets playback
	## start on the first clause instead of waiting for the last.
	signal chunk_ready(clip: AudioStreamWAV, index: int, total: int)

	var text := ""
	var phonemes := ""
	var voice := ""
	var speed := 1.0
	var priority := Priority.NORMAL

	var stream: AudioStreamWAV
	var error := ""
	## Characters the G2P had to guess at. Useful in a localization pass.
	var oov := PackedStringArray()
	## `{words: [...], phonemes: [...], duration: float}` when `compute_timings`
	## is on, otherwise empty. See `KokoroTTS.compute_timings`.
	var timings := {}
	## True when this came back from the cache rather than the GPU.
	var from_cache := false

	var _chunk_limit := 510
	var _timed := false
	var _cache_key := ""
	var _mutex := Mutex.new()
	var _cancelled := false
	var _done := false

	## Stops this request. Anything still queued is dropped; a synthesis
	## already running on the GPU is abandoned within a few milliseconds.
	## `finished` still fires, with a null clip and `error` set.
	func cancel() -> void:
		_mutex.lock()
		_cancelled = true
		_mutex.unlock()

	func is_cancelled() -> bool:
		_mutex.lock()
		var v := _cancelled
		_mutex.unlock()
		return v

	## True once `finished` has fired.
	func is_done() -> bool:
		_mutex.lock()
		var v := _done
		_mutex.unlock()
		return v

	## True if this produced audio.
	func is_ok() -> bool:
		return stream != null

	func _mark_done() -> bool:
		_mutex.lock()
		var already := _done
		_done = true
		_mutex.unlock()
		return not already


## An AudioStreamPlayer that plays clips in the order they arrive, so playback
## can start on the first chunk while the rest are still being synthesized.
class ChunkPlayer extends AudioStreamPlayer:
	signal utterance_finished()

	var _pending: Array = []
	var _closed := false

	func _ready() -> void:
		finished.connect(_advance)

	## Queues a clip, starting playback if nothing is playing.
	func enqueue(clip: AudioStreamWAV) -> void:
		if clip == null:
			return
		if playing:
			_pending.append(clip)
		else:
			stream = clip
			play()

	## No more clips are coming.
	func close() -> void:
		_closed = true
		if not playing and _pending.is_empty():
			utterance_finished.emit()

	func _advance() -> void:
		if _pending.is_empty():
			# Either we are done, or synthesis has not caught up yet; the next
			# enqueue() will restart playback.
			if _closed:
				utterance_finished.emit()
			return
		stream = _pending.pop_front()
		play()


## Feeds chunks into a player the caller already owns and positioned — an
## AudioStreamPlayer3D on an NPC, say. Holds no reference the caller has to
## clean up: it lives exactly as long as the request that made it.
class StreamFeeder extends RefCounted:
	var _player: Node
	var _pending: Array = []
	var _closed := false

	func _init(player: Node) -> void:
		_player = player
		if player.has_signal("finished"):
			player.finished.connect(_advance)

	func enqueue(clip: AudioStreamWAV) -> void:
		if clip == null or not is_instance_valid(_player):
			return
		if _player.playing:
			_pending.append(clip)
		else:
			_player.stream = clip
			_player.play()

	func close() -> void:
		_closed = true

	func _advance() -> void:
		if _pending.is_empty() or not is_instance_valid(_player):
			return
		_player.stream = _pending.pop_front()
		_player.play()


# ------------------------------------------------------------------ lifecycle

func _ready() -> void:
	if Engine.is_editor_hint():
		return
	if preload_on_ready:
		start()


## Begins loading the model. Safe to call more than once.
func start() -> void:
	if _state != State.IDLE:
		return
	_state = State.LOADING
	_sem = Semaphore.new()
	_mutex = Mutex.new()
	# Before the thread, not after: the worker reads it, and a request
	# submitted during loading would otherwise skip the cache entirely.
	_cache = CacheScript.new()
	_cache.memory_budget = cache_memory_mb * 1024 * 1024
	_cache.disk_budget = cache_disk_mb * 1024 * 1024
	_cache.disk_dir = cache_dir if cache_on_disk else ""
	_thread = Thread.new()
	_thread.start(_worker)


func is_ready() -> bool:
	return _state == State.READY


## The reason the most recent failure failed. With several requests in flight
## this is whichever failed last — read `Request.error` instead when it matters.
func get_error() -> String:
	return _error


## The bundled voice names.
static func list_voices() -> PackedStringArray:
	return VoiceLoaderScript.list_voices()


## True if this is a bundled voice. Case-sensitive on purpose: Windows and
## macOS would happily open `AF_HEART.pt` from `af_heart.pt` and Linux would
## not, so the wrong case has to fail everywhere or it fails only on the port.
static func has_voice(voice: String) -> bool:
	return VoiceLoaderScript.exists(voice)


## Language and gender for a voice name, e.g. "American English, female".
static func describe_voice(voice: String) -> String:
	return VoiceLoaderScript.describe(voice)


## True if `speak()` can drive this voice from text. French, Hindi, Japanese
## and Chinese voices have no G2P here and need `speak_phonemes()`; filter the
## voice list with this rather than discovering it as a runtime failure.
func has_g2p(voice: String) -> bool:
	if _g2p == null:
		_g2p = G2PScript.new()
	return _g2p.has_g2p(voice if voice != "" else default_voice)


func _exit_tree() -> void:
	shutdown()


## Stops the engine and releases the GPU device.
##
## Called for you when the node leaves the tree. Everything in flight is
## cancelled first, so this returns in a few milliseconds rather than blocking
## the frame until the current utterance finishes — a scene change during a
## long line used to stall for seconds.
func shutdown() -> void:
	if _thread == null:
		return
	cancel_all()
	_mutex.lock()
	_quit = true
	_mutex.unlock()
	_sem.post()
	# The worker frees the device itself: a RenderingDevice may only be torn
	# down on the thread that created it.
	_thread.wait_to_finish()
	_thread = null
	_state = State.SHUTDOWN
	# Whatever the worker was mid-delivery on will never arrive: its deferred
	# call is dropped along with this node. Release those callers by hand.
	for req in _outstanding.duplicate():
		_strand(req, "engine shut down")
	_outstanding.clear()


## Fails a request that shutdown caught in flight.
##
## Deferred on the Request rather than resolved here, because `shutdown()` runs
## from `_exit_tree`: emitting now resumes the caller's coroutine in the middle
## of node teardown, where even `add_child` fails outright. The Request is a
## RefCounted the caller holds, so it outlives this node and the emit lands on
## the next idle frame like any other.
func _strand(req: Request, reason: String) -> void:
	if not req._mark_done():
		return
	req.error = reason
	_error = reason
	req.finished.emit.call_deferred(null)


# ------------------------------------------------------------------ public API

## Synthesizes `text` and returns a ready-to-play clip. Await it:
##
##     var clip := await tts.speak("Hello.")
##
## Returns null on failure; check `get_error()`, or use `say()` and read the
## request's own `error`.
func speak(text: String, voice := "", speed := 0.0) -> AudioStreamWAV:
	var req := say(text, voice, speed)
	if not req.is_done():
		await req.finished
	return req.stream


## Same, but takes IPA phonemes directly and skips the G2P frontend. Use this
## for languages the built-in G2P does not cover.
func speak_phonemes(ipa: String, voice := "", speed := 0.0) -> AudioStreamWAV:
	var req := say_phonemes(ipa, voice, speed)
	if not req.is_done():
		await req.finished
	return req.stream


## Queues `text` and returns immediately. The [Request] carries the result when
## it lands, and can be cancelled before then.
func say(text: String, voice := "", speed := 0.0,
		priority := Priority.NORMAL) -> Request:
	return _submit(text, "", voice, speed, priority)


## `say()` for IPA, bypassing the G2P frontend.
func say_phonemes(ipa: String, voice := "", speed := 0.0,
		priority := Priority.NORMAL) -> Request:
	return _submit("", ipa, voice, speed, priority)


## Speaks into a player you already own — an AudioStreamPlayer2D or 3D on the
## character, usually — and starts it as soon as the clip is ready.
##
##     tts.speak_into($NPC/Voice, "Halt! Who goes there?", "bm_george")
func speak_into(player: Node, text: String, voice := "", speed := 0.0,
		priority := Priority.NORMAL) -> Request:
	var req := _submit(text, "", voice, speed, priority)
	req.finished.connect(func(clip: AudioStreamWAV):
		if clip != null and is_instance_valid(player):
			player.stream = clip
			player.play())
	return req


## The same, but starts on the first clause instead of the last. Worth it for
## anything over a sentence or two: the first chunk of a long line is ready in
## a few hundred milliseconds whatever the total length.
func stream_into(player: Node, text: String, voice := "", speed := 0.0,
		priority := Priority.NORMAL) -> Request:
	var req := _submit(text, "", voice, speed, priority)
	var feeder := StreamFeeder.new(player)
	req.chunk_ready.connect(func(clip: AudioStreamWAV, _i: int, _n: int):
		feeder.enqueue(clip))
	req.finished.connect(func(_clip): feeder.close())
	# The feeder must outlive this call; the request owns it.
	req.set_meta("feeder", feeder)
	return req


## Starts speaking as soon as the first chunk is ready, into a player this node
## creates and owns. Prefer `stream_into()` when you have a player of your own.
##
##     var p := tts.speak_streaming(long_text)
##     await p.utterance_finished
##     p.queue_free()
func speak_streaming(text: String, voice := "", speed := 0.0) -> ChunkPlayer:
	var player := ChunkPlayer.new()
	player.name = "KokoroChunkPlayer"
	add_child(player)

	var req := _submit(text, "", voice, speed, Priority.NORMAL)
	req.chunk_ready.connect(func(clip: AudioStreamWAV, _i: int, _n: int):
		player.enqueue(clip))
	# Closing has to wait for the request, not for the last chunk callback.
	req.finished.connect(func(_s): player.close())
	return player


## Synthesizes lines now so they are instant later. Queued behind everything
## else, so warming a hundred barks during a loading screen does not delay the
## line the player is waiting on.
##
##     await tts.precache(BARKS, "am_michael")
##
## Awaiting the returned array is not possible directly; await the last entry,
## or connect to `went_idle`.
func precache(lines: PackedStringArray, voice := "") -> Array:
	var out := []
	for line in lines:
		out.append(_submit(line, "", voice, 0.0, Priority.LOW))
	return out


## Drops everything queued and abandons whatever is on the GPU. Returns how
## many requests were affected. Every one of them still fires `finished`.
func cancel_all() -> int:
	if _mutex == null:
		return 0
	_mutex.lock()
	var dropped: Array = _queue
	_queue = []
	var active = _active
	_mutex.unlock()

	for req in dropped:
		req.cancel()
	if active != null:
		active.cancel()
	# The queued ones will never reach the worker, so release them here.
	for req in dropped:
		_resolve(req, null, "cancelled")
	return dropped.size() + (1 if active != null else 0)


## True while anything is queued or running.
func is_busy() -> bool:
	if _mutex == null:
		return false
	_mutex.lock()
	var busy := _active != null or not _queue.is_empty()
	_mutex.unlock()
	return busy


## How many requests are waiting, not counting the one being synthesized.
func queue_size() -> int:
	if _mutex == null:
		return 0
	_mutex.lock()
	var n := _queue.size()
	_mutex.unlock()
	return n


## Hands idle GPU memory back to the driver. This happens on its own once a
## queue drains, so call it only when something else needs the VRAM right now —
## before loading a level, say. The work is queued onto the engine thread,
## because a RenderingDevice may only be touched by the thread that made it.
func release_memory() -> void:
	if _thread == null:
		return
	_mutex.lock()
	_trim_now = true
	_mutex.unlock()
	_sem.post()


## Bytes the engine's RenderingDevice is holding right now — weights, the
## recycling pool and whatever a run in progress has allocated. Safe to poll
## from a debug overlay every frame; it is a counter, not a walk.
func gpu_memory_bytes() -> int:
	return _gpu.total_buffer_bytes() if _gpu != null else 0


## Hits, misses, bytes and entry counts for both cache tiers.
func cache_stats() -> Dictionary:
	return _cache.stats() if _cache != null else {}


## Forgets cached clips. Pass true to delete the on-disk copies too — do that
## when the voice or the model changes, not routinely.
func clear_cache(also_disk := false) -> void:
	if _cache != null:
		_cache.clear(also_disk)


## Text to IPA, without synthesizing. Handy for inspecting the frontend.
##
## Unlike `speak()`, this runs on the calling thread, and the first call for a
## language parses a 90 k-entry lexicon — about 90 ms. Call it off a frame that
## matters, or let the engine warm up first.
func phonemize(text: String, voice := "") -> Dictionary:
	if _g2p == null:
		_g2p = G2PScript.new()
	return _g2p.phonemize(text, voice if voice != "" else default_voice)


# ------------------------------------------------------------------- internals

func _submit(text: String, ipa: String, voice: String, speed: float,
		priority: int) -> Request:
	var req := Request.new()
	_idle_pending = true
	req.text = text
	req.phonemes = ipa
	req.voice = voice if voice != "" else default_voice
	req.speed = speed if speed > 0.0 else default_speed
	req.priority = priority
	req._chunk_limit = chunk_phonemes
	req._timed = compute_timings

	# Speed is a divisor deep in the duration predictor and the model was never
	# trained outside this range; clamping beats emitting a clip that is either
	# unintelligible or thirty seconds long.
	req.speed = clampf(req.speed, 0.5, 2.0)

	if _state == State.IDLE:
		start()
	if _state == State.FAILED:
		return _reject(req, _error)
	if _state == State.SHUTDOWN:
		return _reject(req, "engine shut down")

	if ipa.is_empty() and text.strip_edges().is_empty():
		return _reject(req, "nothing to speak")
	if not ipa.is_empty() and ipa.strip_edges().is_empty():
		return _reject(req, "nothing to speak")
	if not VoiceLoaderScript.exists(req.voice):
		return _reject(req, "unknown voice '%s' — see list_voices()"
				% req.voice)

	# The memory tier is checked here rather than on the worker so a repeated
	# line costs a dictionary lookup and resolves this frame. Keyed on the text
	# because G2P now runs on the worker — which is also why a cache hit skips
	# the lexicon entirely.
	if cache_enabled and _cache != null:
		# The gain is baked into the samples, so it belongs in the key: a
		# clip rendered before normalization was switched on must not be
		# served after it.
		req._cache_key = CacheScript.key_for(
				text if ipa.is_empty() else "ipa:" + ipa,
				req.voice, req.speed, remove_vocoder_tone,
				_gain_for(req.voice))
		var hit: Dictionary = _cache.peek(req._cache_key)
		if not hit.is_empty():
			req.from_cache = true
			req.timings = hit["timings"]
			# Deferred, so a caller of say() can still connect to `finished`.
			_resolve.call_deferred(req, hit["clip"], "")
			return req

	_outstanding.append(req)
	var evicted := _enqueue(req)
	_worker_used = true
	for e in evicted:
		_resolve(e, null, "dropped: more than %d requests queued" % max_queued)
	return req


## Inserts by priority, newest last within a priority. Returns any requests
## pushed out of a full queue.
func _enqueue(req: Request) -> Array:
	_mutex.lock()
	var i := _queue.size()
	while i > 0 and _queue[i - 1].priority < req.priority:
		i -= 1
	_queue.insert(i, req)

	var evicted := []
	while _queue.size() > max_queued:
		# The tail is the lowest priority and, within that, the most recently
		# added. Dropping the newest keeps the backlog in the order the game
		# asked for rather than silently reordering it.
		evicted.append(_queue.pop_back())
	_mutex.unlock()
	_sem.post()
	return evicted


## Fails a request before it ever reaches the worker.
func _reject(req: Request, reason: String) -> Request:
	_error = reason
	_resolve.call_deferred(req, null, reason)
	return req


## The one place a request finishes. Idempotent, so a cancelled request that
## the worker also reported on does not emit twice.
func _resolve(req: Request, clip: AudioStreamWAV, err: String) -> void:
	if not req._mark_done():
		return
	_outstanding.erase(req)
	req.stream = clip
	req.error = err
	if err != "":
		_error = err
	req.finished.emit(clip)
	# Nothing ever reached the worker, so nothing else is going to say we are
	# idle. Say it here.
	if not _worker_used:
		_emit_idle()


## Fires `went_idle` once per busy-to-idle transition, and only when there is
## genuinely nothing left.
func _emit_idle() -> void:
	if not _idle_pending or is_busy() or not _outstanding.is_empty():
		return
	_idle_pending = false
	_worker_used = false
	went_idle.emit()


# ------------------------------------------------------------- worker thread

func _worker() -> void:
	var err := _start_engine()
	_finish_load.call_deferred(err)

	while true:
		_sem.wait()
		_mutex.lock()
		var quit := _quit
		var trim := _trim_now
		_trim_now = false
		var req = _queue.pop_front() if not _queue.is_empty() else null
		_active = req
		_mutex.unlock()

		if req != null:
			if err != "":
				# Engine never started; release the caller rather than
				# leaving its await hanging forever.
				_deliver.call_deferred(req, null, err)
			elif req.is_cancelled():
				_deliver.call_deferred(req, null, "cancelled")
			else:
				_run_request(req)
			_mutex.lock()
			_active = null
			# Only give the memory back once nothing else is waiting: between
			# the chunks of one passage the pool is exactly what we want.
			trim = trim or _queue.is_empty()
			_mutex.unlock()

		if trim and _executor != null:
			_executor.release_activations()
			_executor.trim_memory(gpu_memory_budget_mb * 1024 * 1024)
			_emit_idle.call_deferred()
		if quit:
			# Anything still queued gets released too, so no caller is left
			# awaiting a signal that will never come.
			_mutex.lock()
			var stranded := _queue
			_queue = []
			_mutex.unlock()
			for r in stranded:
				_deliver.call_deferred(r, null, "shutting down")
			break

	_shutdown_engine()


## Runs on the worker thread, which is the only thread allowed to free the
## RenderingDevice it created.
func _shutdown_engine() -> void:
	_executor = null
	if _gpu != null:
		_gpu.cleanup()
		_gpu = null


func _start_engine() -> String:
	_gpu = GPUScript.new()
	var err: String = _gpu.initialize()
	if err != "":
		_gpu = null
		return err

	_model = KpkModelScript.new()
	if _model.load_from(MODEL_PATH) != OK:
		return "cannot load %s — run tools/pack_model.py" % MODEL_PATH
	# The F0 branch upsamples frames by two before the 300x resize, so one
	# predicted duration frame is 600 output samples.
	_samples_per_frame = _model.samples_per_frame * 2

	_levels = _load_levels()
	_voices = VoiceLoaderScript.new()
	_tokenizer = TokenizerScript.new()
	if not _tokenizer.load_vocab():
		return "cannot load the phoneme vocabulary"

	# Both of these used to happen on the calling thread the first time
	# somebody spoke: 90 ms of lexicon parsing, five dropped frames.
	_g2p = G2PScript.new()
	_g2p.warm_up(default_voice)

	_executor = ExecutorScript.new()
	var setup_err: String = _executor.setup(_gpu, _model)
	if setup_err != "":
		return setup_err

	# One kilobyte of coefficients, computed once. Cheap enough to build even
	# when the filter is off, so the flag can be flipped at runtime.
	var taps: PackedFloat32Array = AudioScript.notch_taps(SAMPLE_RATE)
	_fir_rid = _gpu.create_buffer_f32(taps)
	_fir_len = taps.size()
	return ""


## The measured level table, or `{}` if it has not been generated. Missing is
## not an error: every gain falls back to 1.0 and the addon behaves exactly as
## it did before normalization existed.
func _load_levels() -> Dictionary:
	if not FileAccess.file_exists(LEVELS_PATH):
		return {}
	var f := FileAccess.open(LEVELS_PATH, FileAccess.READ)
	if f == null:
		return {}
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(parsed) != TYPE_DICTIONARY or not parsed.has("voices"):
		push_warning("KokoroTTS: %s is malformed; loudness is unnormalized"
				% LEVELS_PATH)
		return {}
	return parsed


## Constant gain that brings `voice` to the target loudness, never above what
## its peak can take.
func _gain_for(voice: String) -> float:
	if not normalize_loudness or _levels.is_empty():
		return 1.0
	var entry = (_levels["voices"] as Dictionary).get(voice)
	if entry == null:
		return 1.0
	var target: float = loudness_target_lufs if loudness_target_lufs < 0.0 \
			else float(_levels.get("target_lufs", -20.0))
	var gain: float = pow(10.0, (target - float(entry["lufs"])) / 20.0)
	return minf(gain, PEAK_CEILING / maxf(0.001, float(entry["peak"])))


func _run_request(req: Request) -> void:
	# The disk tier is read here rather than in `_submit`: it is file I/O, and
	# a game that caches to disk should not pay for it on the frame it asks.
	if cache_enabled and cache_on_disk and _cache != null \
			and req._cache_key != "":
		var hit: Dictionary = _cache.load_from_disk(req._cache_key)
		if not hit.is_empty():
			req.from_cache = true
			req.timings = hit["timings"]
			_deliver.call_deferred(req, hit["clip"], "")
			return

	var words := []
	if req.phonemes.is_empty():
		var r: Dictionary = _g2p.phonemize(req.text, req.voice)
		if r.has("error"):
			_deliver.call_deferred(req, null, r["error"])
			return
		req.phonemes = r["phonemes"]
		req.oov = r["oov"]
		words = r.get("words", [])
	if req.phonemes.strip_edges().is_empty():
		_deliver.call_deferred(req, null, "no phonemes produced")
		return

	var spans: Array = _tokenizer.chunk_spans(req.phonemes,
			mini(req._chunk_limit, _tokenizer.max_phonemes))
	if spans.is_empty():
		_deliver.call_deferred(req, null, "no phonemes produced")
		return

	_executor.cancel_token = req
	_executor.cancel_flush_nodes = gpu_submit_nodes
	var clips: Array = []
	## Absolute [start, end] seconds for each character of `req.phonemes`.
	var char_times := {}
	var elapsed := 0.0

	for i in spans.size():
		if req.is_cancelled():
			_executor.cancel_token = null
			_deliver.call_deferred(req, null, "cancelled")
			return
		var span: Dictionary = spans[i]
		var out: Dictionary = _synthesize(span["text"], req.voice, req.speed,
				req._timed)
		if out.is_empty():
			_executor.cancel_token = null
			var why: String = _executor.last_error
			_deliver.call_deferred(req, null,
					why if why != "" else "synthesis failed")
			return
		var clip: AudioStreamWAV = out["clip"]
		clips.append(clip)
		if req._timed:
			_accumulate_times(char_times, out, int(span["start"]), elapsed)
		elapsed += clip.data.size() / 2.0 / SAMPLE_RATE
		_emit_chunk.call_deferred(req, clip, i, spans.size())

	_executor.cancel_token = null
	var final: AudioStreamWAV = clips[0] if clips.size() == 1 \
			else AudioScript.concatenate(clips)
	if req._timed:
		req.timings = _build_timings(char_times, words, req.phonemes, elapsed)
	if cache_enabled and _cache != null and req._cache_key != "":
		_cache.store(req._cache_key, final, req.timings,
				cache_on_disk)
	_deliver.call_deferred(req, final, "")


## Returns `{clip, durations, source}` or `{}`.
func _synthesize(phonemes: String, voice: String, speed: float,
		timed: bool) -> Dictionary:
	var enc: Dictionary = _tokenizer.encode(phonemes)
	if int(enc["phoneme_count"]) == 0:
		_executor.last_error = "chunk has no phonemes the model knows"
		return {}
	var style: PackedFloat32Array = _voices.style_for(voice,
			int(enc["phoneme_count"]))
	if style.is_empty():
		_executor.last_error = "unknown voice '%s'" % voice
		return {}
	_executor.want_durations = timed
	var res: Dictionary = _executor.run(enc["ids"], style, speed)
	if res.is_empty():
		return {}
	var clip := AudioScript.stream_from_gpu(_gpu, res["buffer"],
			int(res["samples"]), SAMPLE_RATE, _gain_for(voice),
			_fir_rid if remove_vocoder_tone else RID(),
			_fir_len if remove_vocoder_tone else 0)
	if clip == null:
		_executor.last_error = "could not read the waveform back"
		return {}
	return {
		"clip": clip,
		"durations": _executor.last_durations if timed \
				else PackedFloat32Array(),
		"source": enc["source"],
	}


# ---------------------------------------------------------------- timings

## Walks one chunk's predicted durations and records when each character of the
## phoneme string is spoken.
##
## `durations` has one entry per input token — the two boundary markers
## included — so token k of the body is `durations[k + 1]`. `source[k]` says
## which character of the chunk that token came from, and `offset` places the
## chunk inside the whole utterance.
func _accumulate_times(char_times: Dictionary, out: Dictionary, offset: int,
		start_time: float) -> void:
	var durations: PackedFloat32Array = out["durations"]
	var source: PackedInt32Array = out["source"]
	if durations.size() < source.size() + 2:
		return
	var per_frame := float(_samples_per_frame) / float(SAMPLE_RATE)
	# The leading boundary marker is spoken too, and skipping it would shift
	# every word earlier by its length.
	var cursor := durations[0]
	for k in source.size():
		var d: float = durations[k + 1]
		char_times[offset + source[k]] = [
			start_time + cursor * per_frame,
			start_time + (cursor + d) * per_frame,
		]
		cursor += d


## Turns per-character times into the arrays a game actually consumes.
func _build_timings(char_times: Dictionary, words: Array, phonemes: String,
		total: float) -> Dictionary:
	var out_words := []
	for w in words:
		var lo := INF
		var hi := -INF
		for c in range(int(w["start"]), int(w["end"])):
			if not char_times.has(c):
				continue
			var t: Array = char_times[c]
			lo = minf(lo, t[0])
			hi = maxf(hi, t[1])
		if lo == INF:
			# Every phoneme of this word fell outside the vocabulary, or the
			# chunk that held it was truncated. Skip rather than emit a span
			# that would highlight the wrong subtitle.
			continue
		out_words.append({"text": w["text"], "start": lo, "end": hi})

	var out_phonemes := []
	var indices := char_times.keys()
	indices.sort()
	for c in indices:
		var t: Array = char_times[c]
		out_phonemes.append({
			"symbol": phonemes[c] if c < phonemes.length() else "",
			"start": t[0], "end": t[1],
		})

	return {"words": out_words, "phonemes": out_phonemes, "duration": total}


# ----------------------------------------------- main-thread signal delivery

func _finish_load(err: String) -> void:
	if err != "":
		_state = State.FAILED
		_error = err
		engine_failed.emit(err)
		# Nothing queued before the failure can ever run.
		for req in _outstanding.duplicate():
			_resolve(req, null, err)
	else:
		_state = State.READY
		engine_ready.emit()


func _deliver(req: Request, stream: AudioStreamWAV, err: String) -> void:
	_resolve(req, stream, err)


func _emit_chunk(req: Request, clip: AudioStreamWAV, index: int,
		total: int) -> void:
	if not req.is_done():
		req.chunk_ready.emit(clip, index, total)
	synthesis_progress.emit(index + 1, total)
