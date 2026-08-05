@tool
class_name BestVoicePlayer3D
extends AudioStreamPlayer3D

## [BestVoicePlayer] with 3D positional audio — the one most NPCs want. See that
## class for how it works and for why `speak()` is the verb rather than
## `play()`; everything here behaves identically.
##
##     $Merchant/Voice.speak("I have wares, if you have coin.")

const Impl := preload("res://addons/best-tts/nodes/voice_player_impl.gd")

## Emitted when `stream` holds synthesized audio and `play()` will work.
signal prepared()

## The line to speak. A [BestVoiceStream] in `stream` takes priority over this.
@export_multiline var text := ""

## The model's own delivery rate, not a resample.
@export_range(0.5, 2.0, 0.05) var speed := 1.0

## Synthesize on entering the tree so `play()` is instant later.
@export var prepare_on_ready := true

## Start speaking immediately on entering the tree.
@export var play_on_ready := false

## Begin playback on the first clause instead of waiting for the whole line.
@export var stream_long_lines := true

## Which voice pack speaks. A dropdown in the inspector.
var voice := "af_heart"

var _request: KokoroTTS.Request = null


func _ready() -> void:
	if Engine.is_editor_hint():
		return
	await Impl.on_ready(self)


## Speaks `line`, or the current `text`. Returns the request, or null.
func speak(line := "") -> KokoroTTS.Request:
	return Impl.speak(self, line)


## Synthesizes into `stream` without playing. Awaitable; true on success.
func prepare() -> bool:
	return await Impl.prepare(self)


## True from the moment a line is queued until its audio finishes.
func is_speaking() -> bool:
	return Impl.is_speaking(self)


## Abandons an undelivered line without stopping audio already playing.
func cancel() -> void:
	Impl.cancel(self)


## Cancel plus stop.
func stop_speaking() -> void:
	Impl.stop_speaking(self)


func _get_property_list() -> Array[Dictionary]:
	return Impl.property_list()
