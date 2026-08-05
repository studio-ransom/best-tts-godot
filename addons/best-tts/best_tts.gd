@tool
class_name BestTTS
extends Object

## The shared voice engine, reachable from anywhere without wiring anything up.
##
##     BestTTS.speak_into($Voice, "I have wares, if you have coin.", "bm_george")
##
## There is one engine per process — it owns 190 MB of GPU memory and a worker
## thread, so a second one is almost never what you want. The first call creates
## it and parents it to the scene tree root; it shuts down with the tree.
##
## This is a convenience layer, not a replacement. Add a [KokoroTTS] node
## yourself when you want per-scene settings, a separate cache, or control over
## when the model loads. Everything here forwards to exactly that node, which
## you can reach with [method engine].

## Emitted once the model is loaded and the first line can be synthesized.
## Connect via `BestTTS.engine().engine_ready` — a static class cannot carry
## signals of its own.

static var _engine: KokoroTTS = null
static var _configure: Callable = Callable()


## The shared engine, creating it if this is the first call.
##
## Returns null in the editor and in a headless run with no scene tree; every
## other method here tolerates that, so a tool script does not need to guard.
static func engine() -> KokoroTTS:
	if is_instance_valid(_engine):
		return _engine
	if Engine.is_editor_hint():
		return null
	var loop := Engine.get_main_loop()
	if loop == null or not loop is SceneTree:
		push_warning("BestTTS: no SceneTree — the shared engine needs one.")
		return null

	_engine = KokoroTTS.new()
	_engine.name = "BestTTSEngine"
	# _ready() would start it, but that is a frame away and a line submitted
	# before then would find no queue to go in. start() only needs the thread.
	_engine.preload_on_ready = false
	if _configure.is_valid():
		_configure.call(_engine)
	_engine.start()
	# Deferred because the first call usually lands inside someone's _ready(),
	# and a node cannot gain children while its parent is still building.
	(loop as SceneTree).root.add_child.call_deferred(_engine)
	return _engine


## Applies settings to the shared engine before it is created.
##
##     func _init() -> void:
##         BestTTS.configure(func(tts):
##             tts.default_voice = "bm_george"
##             tts.cache_on_disk = true)
##
## Call this from an autoload's `_init`, or anywhere before the first line. If
## the engine already exists the callable runs against it immediately, which is
## fine for most settings but too late for the cache ones.
static func configure(fn: Callable) -> void:
	_configure = fn
	if is_instance_valid(_engine) and fn.is_valid():
		fn.call(_engine)


## True once the model is loaded. Speaking before this is fine — the line waits.
static func is_ready() -> bool:
	var e := engine()
	return e != null and e.is_ready()


## Starts loading the model without speaking anything. Worth calling on a menu
## or loading screen: the first line then plays without the one-off load.
static func warm_up() -> void:
	engine()


## Awaitable. Returns null if there is no engine or synthesis failed.
static func speak(text: String, voice := "", speed := 0.0) -> AudioStreamWAV:
	var e := engine()
	if e == null:
		return null
	return await e.speak(text, voice, speed)


## Queues a line and returns its handle immediately. Never blocks.
static func say(text: String, voice := "", speed := 0.0,
		priority := KokoroTTS.Priority.NORMAL) -> KokoroTTS.Request:
	var e := engine()
	return e.say(text, voice, speed, priority) if e != null else null


## Speaks into a player you own — usually an AudioStreamPlayer3D on a character.
static func speak_into(player: Node, text: String, voice := "", speed := 0.0,
		priority := KokoroTTS.Priority.NORMAL) -> KokoroTTS.Request:
	var e := engine()
	return e.speak_into(player, text, voice, speed, priority) if e != null else null


## The same, starting on the first clause rather than the last. Use it for
## anything longer than a sentence or two.
static func stream_into(player: Node, text: String, voice := "", speed := 0.0,
		priority := KokoroTTS.Priority.NORMAL) -> KokoroTTS.Request:
	var e := engine()
	return e.stream_into(player, text, voice, speed, priority) if e != null else null


## Synthesizes lines now so they cost nothing later. Queued behind everything
## else, so a loading screen full of barks cannot delay a line in front of it.
static func precache(lines: PackedStringArray, voice := "") -> Array:
	var e := engine()
	return e.precache(lines, voice) if e != null else []


## Drops the queue and abandons whatever is on the GPU. Returns how many lines
## were affected; each still fires `finished`.
static func cancel_all() -> int:
	var e := engine()
	return e.cancel_all() if e != null else 0


## True while anything is queued or synthesizing.
static func is_busy() -> bool:
	var e := engine()
	return e != null and e.is_busy()


## The bundled voice names. Safe in the editor and with no engine running.
static func list_voices() -> PackedStringArray:
	return KokoroTTS.list_voices()


## True if this is a bundled pack. Case-sensitive on purpose.
static func has_voice(voice: String) -> bool:
	return KokoroTTS.has_voice(voice)


## "American English, female" and so on.
static func describe_voice(voice: String) -> String:
	return KokoroTTS.describe_voice(voice)


## Stops the shared engine and releases its GPU memory. The next call to
## anything here builds a fresh one, so this is a way to reclaim ~190 MB during
## a long stretch with no dialogue rather than a teardown step you must run.
static func shutdown() -> void:
	if not is_instance_valid(_engine):
		_engine = null
		return
	var e := _engine
	_engine = null
	if e.is_inside_tree():
		e.get_parent().remove_child(e)
	e.shutdown()
	e.queue_free()
