extends RefCounted

## Shared behaviour for [BestVoicePlayer], [BestVoicePlayer2D] and
## [BestVoicePlayer3D].
##
## The three differ only in which AudioStreamPlayer they extend, and GDScript
## has no multiple inheritance, so the logic lives here as static functions and
## each node forwards to it. Every function takes the player as its first
## argument and reaches its exports by name; the nodes are the contract.
##
## Note for anyone extending these: GDScript cannot override a native method.
## A `func play()` on an AudioStreamPlayer subclass parses, and is then never
## called — not by the engine, and not by a script calling `player.play()`
## either. That is why speaking is a `speak()` of its own, and why `prepare()`
## exists to put real audio in `stream` so the native `play()` works normally.


## Synthesizes the line into `stream` without playing it.
##
## Awaitable, returns true if there is now audio to play. This is what makes
## the ordinary `play()` work on these nodes: after it, `stream` holds a plain
## AudioStreamWAV and the player is an ordinary player again.
static func prepare(p) -> bool:
	var s = p.stream
	if s is BestVoiceStream:
		var ok: bool = await s.render()
		if ok:
			p.emit_signal("prepared")
		return ok

	if p.text.strip_edges() == "":
		return false
	var tts := BestTTS.engine()
	if tts == null:
		return false

	var req := tts.say(p.text, p.voice, p.speed)
	p._request = req
	var clip: AudioStreamWAV = await req.finished
	if not is_instance_valid(p):
		return false
	p._request = null
	if clip == null:
		if req.error != "" and req.error != "cancelled":
			push_warning("%s: %s" % [p.name, req.error])
		return false
	p.stream = clip
	p.emit_signal("prepared")
	return true


## Says a line now, replacing whatever this player was saying.
##
## Returns the request so it can be cancelled or awaited; null if there is no
## engine — in the editor, or headless with no scene tree.
static func speak(p, line: String) -> KokoroTTS.Request:
	if line != "":
		p.text = line
	cancel(p)

	# A prepared resource is already audio: play it rather than making the GPU
	# produce it a second time.
	var s = p.stream
	if line == "" and s is BestVoiceStream and s.is_rendered():
		p.play()
		return null

	var say_text: String = p.text
	if line == "" and s is BestVoiceStream and s.text.strip_edges() != "":
		say_text = s.text
	if say_text.strip_edges() == "":
		return null

	var tts := BestTTS.engine()
	if tts == null:
		return null

	# Streaming starts on the first clause, which for a paragraph is the
	# difference between a few hundred milliseconds and a few seconds. A single
	# bark is one chunk either way, so it costs nothing to leave on.
	var req: KokoroTTS.Request
	if p.stream_long_lines:
		req = tts.stream_into(p, say_text, p.voice, p.speed)
	else:
		req = tts.speak_into(p, say_text, p.voice, p.speed)
	p._request = req
	req.finished.connect(func(clip: AudioStreamWAV):
		if not is_instance_valid(p):
			return
		p._request = null
		if clip == null and req.error != "" and req.error != "cancelled":
			push_warning("%s: %s" % [p.name, req.error]))
	return req


## Drops a line that has not been delivered yet, leaving audio already playing
## alone. Safe at any time.
static func cancel(p) -> void:
	if p._request != null:
		p._request.cancel()
		p._request = null


## Cancel plus stop: nothing is playing and nothing is on its way.
static func stop_speaking(p) -> void:
	cancel(p)
	p.stop()


## True from the moment a line is queued until its audio finishes — including
## the synthesis, when there is nothing to hear yet.
static func is_speaking(p) -> bool:
	return p._request != null or p.playing


## Runs the on-ready behaviour the three nodes share.
static func on_ready(p) -> void:
	if p.play_on_ready:
		speak(p, "")
	elif p.prepare_on_ready:
		await prepare(p)


## The inspector dropdown of voice names, shared by all three nodes. The list
## is only known at runtime, which is why this is not an @export_enum.
static func property_list() -> Array[Dictionary]:
	return [{
		"name": "voice",
		"type": TYPE_STRING,
		"usage": PROPERTY_USAGE_DEFAULT,
		"hint": PROPERTY_HINT_ENUM,
		"hint_string": ",".join(KokoroTTS.list_voices()),
	}]
