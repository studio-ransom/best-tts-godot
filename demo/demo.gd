extends Control

## The best-tts demo: the addon's API in the order you would meet it.
##
## Four panels, each one thing, each showing the code that does it. Nothing here
## is scaffolding for the demo — every snippet on screen is the whole of what
## that panel runs.
##
## The nodes in `demo.tscn` are the point of panels one and two: a
## [BestVoicePlayer] with its `text` and `voice` typed into the inspector, and
## another whose `stream` is a [BestVoiceStream] resource. Neither needed an
## engine, an autoload, or a plugin to be enabled.
##
## Requires the Forward+ or Mobile renderer — Compatibility has no compute.

const CROWD := [
	["Did you hear that?", "am_michael"],
	["Over by the west gate.", "af_heart"],
	["Keep your voice down.", "bm_george"],
	["I told you we should have waited.", "af_bella"],
	["There's nothing out there.", "am_adam"],
	["Something moved. I saw it.", "bf_emma"],
	["Get behind me.", "am_onyx"],
	["It's just the wind again.", "af_nicole"],
	["Light the lantern, quickly.", "bm_lewis"],
	["Too late for that now.", "am_fenrir"],
]

const LONG_LINE := """The road out of the valley was never meant to carry this
much traffic, and after the third night of rain it stopped pretending to be a
road at all. We walked the last two miles, which gave everyone plenty of time to
explain whose idea this had been."""

@onready var _narrator: BestVoicePlayer = $Narrator
@onready var _merchant: BestVoicePlayer = $Merchant
@onready var _free_form: BestVoicePlayer = $FreeForm

var _log: RichTextLabel
var _status: Label
var _meters: Label
var _input: TextEdit
var _voices: OptionButton
var _speed: HSlider
var _speed_label: Label
var _prepared_dot: Label
var _crowd_players: Array[BestVoicePlayer] = []
var _last_ms := 0


func _ready() -> void:
	_build_ui()

	# This is the only "setup" in the whole demo, and it is optional: it starts
	# the model loading now rather than on the first line.
	BestTTS.warm_up()
	var tts := BestTTS.engine()
	if tts == null:
		_status.text = "No engine — this needs the Forward+ or Mobile renderer."
		return
	tts.compute_timings = true
	tts.engine_ready.connect(func():
		_status.text = "Ready. 50 voices, all of it on your GPU."
		_say("[color=#7fd1b9]engine ready[/color] — model loaded, %d voices"
				% BestTTS.list_voices().size()))
	tts.engine_failed.connect(func(reason: String):
		_status.text = "Engine failed: " + reason
		_say("[color=#e08b7a]%s[/color]" % reason))

	# Children are ready before their parent, so check as well as connect —
	# a cached line could in principle already be sitting there.
	_merchant.prepared.connect(_on_merchant_prepared)
	if _merchant.stream is BestVoiceStream and _merchant.stream.is_rendered():
		_on_merchant_prepared()

	_status.text = "Loading the model..."


func _process(_dt: float) -> void:
	if _meters == null:
		return
	var tts := BestTTS.engine()
	if tts == null:
		return
	var cache: Dictionary = tts.cache_stats()
	_meters.text = "queue %d   ·   gpu %d MB   ·   cache %d clips / %.1f MB   ·   last %d ms" % [
		tts.queue_size(),
		tts.gpu_memory_bytes() / 1048576,
		int(cache.get("entries", 0)),
		float(cache.get("bytes", 0)) / 1048576.0,
		_last_ms,
	]


# ------------------------------------------------------------------- panel one

func _panel_node(col: VBoxContainer) -> void:
	var box := _panel(col, "1 · A node that talks",
			"A BestVoicePlayer with its line and voice typed into the "
			+ "inspector. No engine node, no autoload, no plugin to enable.")

	_code(box, "$Narrator.speak()")

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)

	var speak := Button.new()
	speak.text = "Speak"
	speak.pressed.connect(func():
		_timed(_narrator.speak(), "Narrator"))
	row.add_child(speak)

	var interrupt := Button.new()
	interrupt.text = "Interrupt"
	interrupt.pressed.connect(func():
		_narrator.stop_speaking()
		_say("stopped the narrator mid-line"))
	row.add_child(interrupt)

	var detail := Label.new()
	detail.text = "voice: %s  ·  %s" % [_narrator.voice,
			BestTTS.describe_voice(_narrator.voice)]
	detail.add_theme_color_override("font_color", Color("#8b9bab"))
	row.add_child(detail)


# ------------------------------------------------------------------- panel two

func _panel_resource(col: VBoxContainer) -> void:
	var box := _panel(col, "2 · A line as a resource",
			"$Merchant's stream is a BestVoiceStream — text and voice saved "
			+ "in the scene. It renders itself while the scene loads, and is "
			+ "an ordinary AudioStreamWAV from then on.")

	_code(box, "$Merchant.play()   # the ordinary AudioStreamPlayer.play()")

	_prepared_dot = Label.new()
	_prepared_dot.text = "○ rendering..."
	_prepared_dot.add_theme_color_override("font_color", Color("#8b9bab"))
	box.add_child(_prepared_dot)

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)

	var play := Button.new()
	play.text = "Play"
	play.pressed.connect(func():
		var t0 := Time.get_ticks_usec()
		_merchant.play()
		_say("play() returned in %.2f ms — nothing to synthesize"
				% ((Time.get_ticks_usec() - t0) / 1000.0)))
	row.add_child(play)

	var again := Button.new()
	again.text = "Change the line and re-render"
	again.pressed.connect(_rerender_merchant)
	row.add_child(again)


func _on_merchant_prepared() -> void:
	_prepared_dot.text = "● prepared — play() is instant now"
	_prepared_dot.add_theme_color_override("font_color", Color("#7fd1b9"))


func _rerender_merchant() -> void:
	var line: BestVoiceStream = _merchant.stream
	if line == null:
		return
	line.text = "On second thoughts, the price has gone up."
	_prepared_dot.text = "○ rendering..."
	_prepared_dot.add_theme_color_override("font_color", Color("#8b9bab"))
	_say("edited the resource's text — the old audio was discarded")
	var t0 := Time.get_ticks_msec()
	await _merchant.prepare()
	_last_ms = Time.get_ticks_msec() - t0
	_say("re-rendered in %d ms" % _last_ms)
	_merchant.play()


# ----------------------------------------------------------------- panel three

func _panel_free(col: VBoxContainer) -> void:
	var box := _panel(col, "3 · Anything you type",
			"Type a line, pick any of the 50 voices, hear it. Numbers, "
			+ "currency, times and abbreviations are handled by the frontend.")

	_code(box, "$FreeForm.voice = \"af_heart\"\n$FreeForm.speak(text)")

	_input = TextEdit.new()
	_input.text = ("The 7:45 train to Bristol costs £24.50, which is about "
			+ "3 times what it cost in 2019.")
	_input.custom_minimum_size = Vector2(0, 66)
	_input.wrap_mode = TextEdit.LINE_WRAPPING_BOUNDARY
	box.add_child(_input)

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)

	_voices = OptionButton.new()
	var tts := BestTTS.engine()
	for v in BestTTS.list_voices():
		var speakable: bool = tts == null or tts.has_g2p(v)
		_voices.add_item(v if speakable else "%s  (IPA only)" % v)
		_voices.set_item_metadata(_voices.item_count - 1, v)
		_voices.set_item_disabled(_voices.item_count - 1, not speakable)
		if v == "af_heart":
			_voices.select(_voices.item_count - 1)
	row.add_child(_voices)

	_speed_label = Label.new()
	_speed_label.text = "1.00x"
	_speed = HSlider.new()
	_speed.min_value = 0.6
	_speed.max_value = 1.6
	_speed.step = 0.05
	_speed.value = 1.0
	_speed.custom_minimum_size = Vector2(130, 0)
	_speed.value_changed.connect(func(v: float):
		_speed_label.text = "%.2fx" % v)
	row.add_child(_speed)
	row.add_child(_speed_label)

	var speak := Button.new()
	speak.text = "Speak"
	speak.pressed.connect(_speak_typed)
	row.add_child(speak)

	var long_one := Button.new()
	long_one.text = "A long passage (streams)"
	long_one.pressed.connect(func():
		_input.text = LONG_LINE.replace("\n", " ")
		_speak_typed())
	row.add_child(long_one)


func _speak_typed() -> void:
	var text := _input.text.strip_edges()
	if text == "":
		return
	_free_form.voice = _voices.get_item_metadata(_voices.selected)
	_free_form.speed = _speed.value
	_timed(_free_form.speak(text), "FreeForm")


# ------------------------------------------------------------------ panel four

func _panel_load(col: VBoxContainer) -> void:
	var box := _panel(col, "4 · What a game actually does",
			"Ten characters speaking at once, an urgent line jumping the "
			+ "queue, and the same line a second time.")

	_code(box, "BestTTS.say(line, voice, 0.0, KokoroTTS.Priority.HIGH)\n"
			+ "BestTTS.cancel_all()")

	var row := HBoxContainer.new()
	row.add_theme_constant_override("separation", 8)
	box.add_child(row)

	var crowd := Button.new()
	crowd.text = "Ten NPCs at once"
	crowd.pressed.connect(_speak_crowd)
	row.add_child(crowd)

	var urgent := Button.new()
	urgent.text = "Urgent line (HIGH)"
	urgent.pressed.connect(func():
		var t0 := Time.get_ticks_msec()
		var req := BestTTS.say("Look out!", "am_michael", 0.0,
				KokoroTTS.Priority.HIGH)
		_say("queued HIGH behind %d others" % BestTTS.engine().queue_size())
		req.finished.connect(func(clip: AudioStreamWAV):
			if clip == null:
				return
			_crowd_players[0].stream = clip
			_crowd_players[0].play()
			_say("[color=#7fd1b9]HIGH line played after %d ms[/color]"
					% (Time.get_ticks_msec() - t0))))
	row.add_child(urgent)

	var repeat := Button.new()
	repeat.text = "Say the first one again (cache)"
	repeat.pressed.connect(func():
		var t0 := Time.get_ticks_usec()
		var req := BestTTS.speak_into(_crowd_players[0], CROWD[0][0], CROWD[0][1])
		var queued := (Time.get_ticks_usec() - t0) / 1000.0
		await req.finished
		_say("cache hit: %s — %.2f ms on this thread"
				% ["yes" if req.from_cache else "no", queued]))
	row.add_child(repeat)

	var stop := Button.new()
	stop.text = "Cancel everything"
	stop.pressed.connect(func():
		var n := BestTTS.cancel_all()
		_say("dropped %d requests, abandoned the GPU run" % n))
	row.add_child(stop)


func _speak_crowd() -> void:
	var t0 := Time.get_ticks_msec()
	var done := [0]
	for i in CROWD.size():
		var player := _crowd_players[i]
		var req := BestTTS.speak_into(player, CROWD[i][0], CROWD[i][1])
		req.finished.connect(func(_clip):
			done[0] += 1
			if done[0] == CROWD.size():
				_say("all %d lines delivered in %d ms"
						% [CROWD.size(), Time.get_ticks_msec() - t0]))
	_say("queued %d lines in %d ms on the main thread"
			% [CROWD.size(), Time.get_ticks_msec() - t0])


# ----------------------------------------------------------------------- shell

func _timed(req: KokoroTTS.Request, who: String) -> void:
	if req == null:
		return
	var t0 := Time.get_ticks_msec()
	req.finished.connect(func(clip: AudioStreamWAV):
		_last_ms = Time.get_ticks_msec() - t0
		if clip == null:
			_say("[color=#e08b7a]%s: %s[/color]" % [who, req.error])
			return
		var secs := clip.data.size() / 2.0 / clip.mix_rate
		var words: int = req.timings.get("words", []).size()
		_say("%s: %.2f s of audio in %d ms (%.1fx realtime)%s"
				% [who, secs, _last_ms, secs * 1000.0 / maxf(1.0, _last_ms),
				"" if words == 0 else ", %d word timings" % words]))


func _say(line: String) -> void:
	_log.append_text(line + "\n")


func _build_ui() -> void:
	set_anchors_preset(Control.PRESET_FULL_RECT)

	var margin := MarginContainer.new()
	margin.set_anchors_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "right", "top", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 18)
	add_child(margin)

	var page := VBoxContainer.new()
	page.add_theme_constant_override("separation", 10)
	margin.add_child(page)

	var title := Label.new()
	title.text = "best-tts"
	title.add_theme_font_size_override("font_size", 26)
	page.add_child(title)

	var sub := Label.new()
	sub.text = ("Kokoro-82M neural text-to-speech, running inside Godot in "
			+ "pure GDScript and GLSL. No C++, no GDExtension, no executables.")
	sub.add_theme_color_override("font_color", Color("#8b9bab"))
	page.add_child(sub)

	_status = Label.new()
	_status.add_theme_font_size_override("font_size", 15)
	page.add_child(_status)

	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	page.add_child(scroll)

	var col := VBoxContainer.new()
	col.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	col.add_theme_constant_override("separation", 14)
	scroll.add_child(col)

	_panel_node(col)
	_panel_resource(col)
	_panel_free(col)
	_panel_load(col)

	_meters = Label.new()
	_meters.add_theme_color_override("font_color", Color("#8b9bab"))
	page.add_child(_meters)

	_log = RichTextLabel.new()
	_log.bbcode_enabled = true
	_log.scroll_following = true
	_log.custom_minimum_size = Vector2(0, 150)
	page.add_child(_log)

	# One player per crowd line, so ten characters really can overlap.
	for i in CROWD.size():
		var p := BestVoicePlayer.new()
		p.name = "Crowd%d" % i
		p.prepare_on_ready = false
		add_child(p)
		_crowd_players.append(p)


## A titled section with a description, returning the box to fill.
func _panel(col: VBoxContainer, title: String, blurb: String) -> VBoxContainer:
	var panel := PanelContainer.new()
	col.add_child(panel)

	var inner := MarginContainer.new()
	for side in ["left", "right", "top", "bottom"]:
		inner.add_theme_constant_override("margin_" + side, 12)
	panel.add_child(inner)

	var box := VBoxContainer.new()
	box.add_theme_constant_override("separation", 8)
	inner.add_child(box)

	var head := Label.new()
	head.text = title
	head.add_theme_font_size_override("font_size", 18)
	box.add_child(head)

	var desc := Label.new()
	desc.text = blurb
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	desc.add_theme_color_override("font_color", Color("#8b9bab"))
	box.add_child(desc)
	return box


## The code that the panel's buttons actually run, shown verbatim.
func _code(box: VBoxContainer, snippet: String) -> void:
	var label := RichTextLabel.new()
	label.bbcode_enabled = true
	label.fit_content = true
	label.add_theme_font_override("normal_font", ThemeDB.fallback_font)
	label.append_text("[code][color=#9ecfff]%s[/color][/code]" % snippet)
	box.add_child(label)
