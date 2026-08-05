@tool
class_name BestVoiceStream
extends AudioStreamWAV

## A spoken line, authored as a resource: type the text, pick the voice, done.
##
## It is a real [AudioStreamWAV], so once it has been rendered any
## AudioStreamPlayer, 2D or 3D, plays it with ordinary playback — correct
## length, working `finished` signal, seeking, the lot. What it adds is knowing
## how to fill itself in:
##
##     var line := BestVoiceStream.of("I have wares, if you have coin.",
##             "bm_george")
##     await line.render()
##     $Voice.stream = line
##     $Voice.play()
##
## [BestVoicePlayer] does that for you, which is the shorter path. Reach for
## this directly when you want lines as `.tres` files a writer can edit, or when
## you want to render during a loading screen and play later.
##
## Editing `text`, `voice` or `speed` discards the audio, so a stale render is
## never played by accident.

## What to say. The G2P frontend handles numbers, currency, times and
## abbreviations, so "$14.50 at 3pm" is fine as written.
@export_multiline var text := "":
	set(v):
		if v == text:
			return
		text = v
		_invalidate()

## Playback rate the model was asked for, not a resample — the delivery changes
## with it. 1.0 is the voice's natural pace.
@export_range(0.5, 2.0, 0.05) var speed := 1.0:
	set(v):
		if is_equal_approx(v, speed):
			return
		speed = v
		_invalidate()

## Which pack speaks it. Shown as a dropdown in the inspector; see
## [method KokoroTTS.describe_voice] for what each name means.
var voice := "af_heart":
	set(v):
		if v == voice:
			return
		voice = v
		_invalidate()

var _rendering := false


func _init() -> void:
	format = AudioStreamWAV.FORMAT_16_BITS
	mix_rate = KokoroTTS.SAMPLE_RATE
	stereo = false


## A line built in code.
static func of(line: String, pack := "af_heart", rate := 1.0) -> BestVoiceStream:
	var s := BestVoiceStream.new()
	s.text = line
	s.voice = pack
	s.speed = rate
	return s


## True once there is audio to play.
func is_rendered() -> bool:
	return data.size() > 0


## Length in seconds, or 0 before it has been rendered.
func duration() -> float:
	return data.size() / 2.0 / maxf(1.0, float(mix_rate))


## Synthesizes the line into this resource. Awaitable; returns true on success.
##
## Cheap to call again — a second call with the audio already present returns
## immediately, and the engine's own cache means even a re-render after an edit
## is usually free.
func render(engine: KokoroTTS = null) -> bool:
	if is_rendered():
		return true
	if text.strip_edges() == "":
		return false
	# Two players sharing one resource would otherwise both queue the same line.
	if _rendering:
		while _rendering:
			await Engine.get_main_loop().process_frame
		return is_rendered()

	var tts := engine if engine != null else BestTTS.engine()
	if tts == null:
		return false

	_rendering = true
	var clip: AudioStreamWAV = await tts.speak(text, voice, speed)
	_rendering = false
	if clip == null:
		# Cancellation is what shutdown looks like from here, and warning about
		# it turns every clean exit mid-line into a stack trace.
		var why := tts.get_error()
		if why != "" and why != "cancelled":
			push_warning("BestVoiceStream: %s" % why)
		return false

	_adopt(clip)
	return true


## Throws away the rendered audio. The next `render()` produces it again.
func clear() -> void:
	if data.size() > 0:
		data = PackedByteArray()
		emit_changed()


func _adopt(clip: AudioStreamWAV) -> void:
	format = clip.format
	mix_rate = clip.mix_rate
	stereo = clip.stereo
	data = clip.data
	emit_changed()


func _invalidate() -> void:
	if data.size() > 0:
		data = PackedByteArray()
	emit_changed()


# The voice list is only known at runtime, so the dropdown is built here rather
# than with @export_enum. Everything else is a plain export.
func _get_property_list() -> Array[Dictionary]:
	return [{
		"name": "voice",
		"type": TYPE_STRING,
		"usage": PROPERTY_USAGE_DEFAULT,
		"hint": PROPERTY_HINT_ENUM,
		"hint_string": ",".join(KokoroTTS.list_voices()),
	}]
