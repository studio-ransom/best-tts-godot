@tool
class_name BestVoicePlayer
extends AudioStreamPlayer

## An AudioStreamPlayer that speaks. Type a line, pick a voice, done.
##
## This is the shortest path to speech in the addon. Nothing to add to the scene
## besides this node, nothing to enable, no engine to wire up — the first line
## creates the shared [BestTTS] engine and every player after that shares it.
##
##     $Voice.speak("Did you hear that?")
##
## Or set `text` and `voice` in the inspector and leave `prepare_on_ready` on:
## the line is synthesized while the scene loads, `stream` fills with ordinary
## audio, and from then on this is an ordinary player — `play()`, `stop()`,
## `seek()`, `finished` and the rest all behave exactly as they always do.
##
##     $Voice.play()          # instant, if it has had time to prepare
##     await $Voice.prepared  # if you need to be sure
##
## [b]`speak()` is the verb, not `play()`.[/b] GDScript cannot override a native
## method, so a `play()` on an unprepared line has nothing to play and will
## simply do nothing. `speak()` always works: it synthesizes if it has to and
## starts as soon as there is audio.
##
## For positional dialogue use [BestVoicePlayer2D] or [BestVoicePlayer3D],
## which are the same node on a different base.

const Impl := preload("res://addons/best-tts/nodes/voice_player_impl.gd")

## Emitted when `stream` has been filled with synthesized audio and `play()`
## will work. Fires once per prepare, not per playback.
signal prepared()

## The line to speak. A [BestVoiceStream] in `stream` takes priority over this.
@export_multiline var text := ""

## The model's own delivery rate, not a resample — the performance changes with
## it. 1.0 is the voice's natural pace.
@export_range(0.5, 2.0, 0.05) var speed := 1.0

## Synthesize on entering the tree so `play()` is instant later. Costs nothing
## at runtime: it happens on the worker thread, and a line already in the cache
## comes back without touching the GPU.
@export var prepare_on_ready := true

## Start speaking immediately on entering the tree.
@export var play_on_ready := false

## Begin playback on the first clause instead of waiting for the whole line.
## Worth keeping on for anything longer than a sentence; for a bark it makes no
## difference. Only applies to `speak()`.
@export var stream_long_lines := true

## Which voice pack speaks. A dropdown in the inspector.
var voice := "af_heart"

var _request: KokoroTTS.Request = null


func _ready() -> void:
	if Engine.is_editor_hint():
		return
	await Impl.on_ready(self)


## Speaks `line`, or the current `text` when called with no argument. Replaces
## anything this player was already saying, and works whether or not the line
## has been prepared. Returns the request so it can be cancelled or awaited.
func speak(line := "") -> KokoroTTS.Request:
	return Impl.speak(self, line)


## Synthesizes into `stream` without playing. Awaitable; true if there is now
## audio. After this the ordinary `play()` works.
func prepare() -> bool:
	return await Impl.prepare(self)


## True from the moment a line is queued until its audio finishes — including
## the synthesis, before there is anything to hear.
func is_speaking() -> bool:
	return Impl.is_speaking(self)


## Abandons a line that has not been delivered yet, without stopping audio that
## is already playing.
func cancel() -> void:
	Impl.cancel(self)


## Cancel plus stop: nothing playing, nothing on the way.
func stop_speaking() -> void:
	Impl.stop_speaking(self)


func _get_property_list() -> Array[Dictionary]:
	return Impl.property_list()
