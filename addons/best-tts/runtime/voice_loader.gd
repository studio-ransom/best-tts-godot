@tool
class_name KokoroVoiceLoader
extends RefCounted

## Loads Kokoro voice packs from the original PyTorch `.pt` files.
##
## No unpickling is needed: a `.pt` is a zip archive, and these packs contain a
## single storage entry `<name>/data/0` holding 510 x 256 little-endian float32
## values. The style row for an utterance is picked by phoneme count, matching
## Kokoro's own KPipeline.

const VOICE_DIR := "res://addons/best-tts/assets/voices/"
const ROWS := 510
const DIM := 256

var _cache: Dictionary = {}
## Directory listing, done once. Also the allow-list — see `exists()`.
static var _names := PackedStringArray()
static var _scanned := false


## All bundled voice names, sorted.
static func list_voices() -> PackedStringArray:
	if _scanned:
		return _names
	_scanned = true
	var dir := DirAccess.open(VOICE_DIR)
	if dir == null:
		push_error("KokoroVoiceLoader: cannot open %s" % VOICE_DIR)
		return _names
	for f in dir.get_files():
		# Exported projects append .remap / .import to bundled files.
		var base := f.trim_suffix(".remap").trim_suffix(".import")
		if base.ends_with(".pt"):
			_names.append(base.trim_suffix(".pt"))
	_names.sort()
	return _names


## True if this is one of the bundled packs, matched exactly.
##
## Case matters and the check has to be an allow-list rather than a file probe.
## Windows and macOS open `AF_HEART.pt` from `af_heart.pt` quite happily, so a
## game written against the wrong case ships working and then fails on Linux —
## Godot warns about it at load time and the warning is easy to miss. Rejecting
## the name here makes that a same-day bug instead of a port-day one. It also
## keeps `../../` and other path fragments out of `load_pack`.
static func exists(voice: String) -> bool:
	return list_voices().has(voice)


## Human-readable origin, derived from the name prefix Kokoro uses.
static func describe(voice: String) -> String:
	if voice.length() < 2:
		return voice
	const LANG := {
		"a": "American English", "b": "British English", "e": "Spanish",
		"f": "French", "h": "Hindi", "i": "Italian", "j": "Japanese",
		"p": "Portuguese", "z": "Chinese",
	}
	var lang: String = LANG.get(voice[0], "Unknown")
	var gender := "female" if voice[1] == "f" else "male"
	return "%s, %s" % [lang, gender]


## The full [510 x 256] pack as a flat array, cached across calls.
func load_pack(voice: String) -> PackedFloat32Array:
	if _cache.has(voice):
		return _cache[voice]

	var path := VOICE_DIR + voice + ".pt"
	var zip := ZIPReader.new()
	if zip.open(path) != OK:
		push_error("KokoroVoiceLoader: cannot open voice %s" % path)
		return PackedFloat32Array()

	var entry := ""
	for f in zip.get_files():
		if f.ends_with("/data/0"):
			entry = f
			break
	if entry.is_empty():
		push_error("KokoroVoiceLoader: %s has no tensor storage entry" % path)
		zip.close()
		return PackedFloat32Array()

	var raw := zip.read_file(entry)
	zip.close()

	if raw.size() != ROWS * DIM * 4:
		push_error("KokoroVoiceLoader: %s is %d bytes, expected %d"
				% [path, raw.size(), ROWS * DIM * 4])
		return PackedFloat32Array()

	var pack := raw.to_float32_array()
	_cache[voice] = pack
	return pack


## Style vector for an utterance of `phoneme_count` phonemes.
##
## Kokoro indexes the pack by `len(phonemes) - 1`; `input_ids` additionally
## carries a leading and trailing 0, so this is `len(input_ids) - 3`.
func style_for(voice: String, phoneme_count: int) -> PackedFloat32Array:
	var pack := load_pack(voice)
	if pack.is_empty():
		return PackedFloat32Array()
	var row := clampi(phoneme_count - 1, 0, ROWS - 1)
	return pack.slice(row * DIM, (row + 1) * DIM)


func clear_cache() -> void:
	_cache.clear()
