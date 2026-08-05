@tool
class_name KokoroTokenizer
extends RefCounted

## Maps Kokoro phoneme strings to model token ids.
##
## The vocabulary is the 114-symbol IPA set from the official Kokoro config
## (ids up to 177). Input to the model is `[0, ...phonemes, 0]`; the leading and
## trailing zeros are boundary markers, not part of the phoneme count.

const VOCAB_PATH := "res://addons/best-tts/assets/vocab.json"

## Longest phoneme run the model accepts (the style pack has 510 rows).
const MAX_PHONEMES := 510

var vocab: Dictionary = {}
var max_phonemes := MAX_PHONEMES
var _loaded := false


func load_vocab() -> bool:
	if _loaded:
		return true
	var f := FileAccess.open(VOCAB_PATH, FileAccess.READ)
	if f == null:
		push_error("KokoroTokenizer: cannot open %s" % VOCAB_PATH)
		return false
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(parsed) != TYPE_DICTIONARY or not parsed.has("vocab"):
		push_error("KokoroTokenizer: malformed vocab.json")
		return false
	vocab = parsed["vocab"]
	max_phonemes = int(parsed.get("max_phonemes", MAX_PHONEMES))
	_loaded = true
	return true


func is_ready() -> bool:
	return _loaded


## True if the model knows this phoneme character.
func has_symbol(ch: String) -> bool:
	return vocab.has(ch)


## Encodes a phoneme string.
##
## Returns `{ids, phoneme_count, unknown, truncated, source}` — `ids` already
## carries the boundary zeros, `unknown` lists characters that were dropped,
## `truncated` reports whether the run hit `max_phonemes`, and `source[k]` is
## the index in `phonemes` that token `k` of the body came from. That last one
## is what lets a predicted duration be traced back to the word it belongs to.
func encode(phonemes: String) -> Dictionary:
	if not load_vocab():
		return {"ids": PackedInt32Array(), "phoneme_count": 0,
				"unknown": PackedStringArray(), "truncated": false,
				"source": PackedInt32Array()}

	var body := PackedInt32Array()
	var source := PackedInt32Array()
	var unknown := PackedStringArray()
	var truncated := false

	for i in phonemes.length():
		var ch := phonemes[i]
		if not vocab.has(ch):
			if not unknown.has(ch):
				unknown.append(ch)
			continue
		if body.size() >= max_phonemes:
			truncated = true
			break
		body.append(int(vocab[ch]))
		source.append(i)

	var ids := PackedInt32Array([0])
	ids.append_array(body)
	ids.append(0)
	return {
		"ids": ids,
		"phoneme_count": body.size(),
		"unknown": unknown,
		"truncated": truncated,
		"source": source,
	}


## Splits a long phoneme string into runs of at most `max_phonemes`, preferring
## to break after punctuation so chunk boundaries land on natural pauses.
func chunk(phonemes: String, limit := -1) -> PackedStringArray:
	var out := PackedStringArray()
	for span in chunk_spans(phonemes, limit):
		out.append(span["text"])
	return out


## The same split, with each run's starting offset in `phonemes`.
##
## Timings need the offset: a predicted duration belongs to a token, a token
## belongs to a character of the chunk, and only the offset says which
## character of the whole utterance that was.
func chunk_spans(phonemes: String, limit := -1) -> Array:
	if limit <= 0:
		limit = max_phonemes
	limit = mini(limit, max_phonemes)
	var out := []
	var current := ""
	var pending := ""
	## Where `current` starts in `phonemes`; `pending` follows it directly.
	var start := 0

	for ch in phonemes:
		pending += ch
		# ; : , . ! ? and the pause characters are safe split points
		if ch in ".!?;:," or ch == "…" or ch == "—":
			if current.length() + pending.length() > limit and current != "":
				_emit(out, current, start)
				start += current.length()
				current = pending
			else:
				current += pending
			pending = ""
		elif current.length() + pending.length() >= limit:
			# No punctuation in range; break at the last whitespace we can.
			var cut := pending.rfind(" ")
			if cut > 0:
				current += pending.substr(0, cut)
				pending = pending.substr(cut + 1)
			elif current == "":
				# A single run longer than the limit and nothing to break on —
				# a URL, a German compound, an initialism the lexicon spelled
				# out. This used to emit an empty chunk and leave `pending`
				# growing, so the whole utterance came back as "synthesis
				# failed". Cut mid-run instead: ugly, but it speaks.
				current = pending
				pending = ""
			_emit(out, current, start)
			start += current.length()
			current = ""

	current += pending
	_emit(out, current, start)
	return out


## Appends a run, skipping any that carries no phoneme at all — an empty chunk
## encodes to zero tokens and fails the whole utterance.
static func _emit(out: Array, text: String, start: int) -> void:
	if text.strip_edges() == "":
		return
	out.append({"text": text, "start": start})
