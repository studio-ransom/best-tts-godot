@tool
class_name KokoroG2P
extends RefCounted

## Text -> IPA phonemes for the Kokoro model.
##
## English (American and British) is done properly: misaki's gold lexicons,
## bundled as JSON, with a rule-based letter-to-sound fallback for words the
## lexicon does not know. Spanish, Italian and Portuguese are best-effort rule
## systems -- those orthographies are regular enough that plain rules produce
## intelligible speech. Other languages should use `speak_phonemes()` with
## externally produced IPA.

const LEXICON_DIR := "res://addons/best-tts/assets/lexicons/"

## Characters treated as vowel sounds when placing stress marks.
const VOWELS := "AIOWYaeiouæɐɑɒɔəɚɛɜɨɪʊʌᵻ"

var _lexicons: Dictionary = {}


## Kokoro voice names encode the language in their first letter.
static func lang_of(voice: String) -> String:
	match voice.substr(0, 1):
		"a": return "en-us"
		"b": return "en-gb"
		"e": return "es"
		"i": return "it"
		"p": return "pt"
		"f": return "fr"
		"h": return "hi"
		"j": return "ja"
		"z": return "zh"
	return "en-us"


static func supported(lang: String) -> bool:
	return lang in ["en-us", "en-gb", "es", "it", "pt"]


## Converts text to a phoneme string for the given voice.
##
## Returns `{phonemes, oov, lang, words}`, where `words` gives each spoken
## word's `[start, end)` character range in `phonemes` — the anchor timings are
## built from. An empty `phonemes` with a non-empty `error` means the language
## has no G2P.
func phonemize(text: String, voice: String) -> Dictionary:
	var lang := lang_of(voice)
	match lang:
		"en-us", "en-gb":
			var r := _phonemize_en(text, lang == "en-us")
			r["lang"] = lang
			return r
		"es", "it", "pt":
			var r := _phonemize_rules(text, lang)
			r["lang"] = lang
			return r
	return {
		"phonemes": "", "oov": PackedStringArray(), "lang": lang, "words": [],
		"error": "no G2P for '%s' — use speak_phonemes() with IPA" % lang,
	}


## True if `speak()` can handle this voice's language. Lets a game filter the
## voice list down to the ones it can actually drive from text.
func has_g2p(voice: String) -> bool:
	return lang_of(voice) in ["en-us", "en-gb", "es", "it", "pt"]


## Loads the lexicons this voice needs. 90 k entries take ~90 ms to parse, and
## doing that lazily on the first `speak()` drops five frames; the engine calls
## this on its worker thread at startup instead.
func warm_up(voice: String) -> void:
	match lang_of(voice):
		"en-us": _load_lexicon("us_gold")
		"en-gb": _load_lexicon("gb_gold")


# ------------------------------------------------------------------- English

func _phonemize_en(text: String, us: bool) -> Dictionary:
	var lex := _load_lexicon("us_gold" if us else "gb_gold")
	var oov := PackedStringArray()
	var tokens := _tokenize(_normalize_en(text))
	var out := ""
	# Where each spoken word landed in `out`. Recorded as it is written rather
	# than recovered afterwards: punctuation is folded into the preceding word
	# and a normalized token can expand to several words, so the string alone
	# no longer says which characters belong to which word.
	var words := []

	for i in tokens.size():
		var tk: String = tokens[i]
		if not _is_word(tk):
			# Punctuation binds to the previous word, then gets a space.
			var before := out.length()
			out = out.rstrip(" ") + tk + " "
			# Extend the previous word over the punctuation it now carries, so
			# a subtitle highlight does not blink off during the pause.
			if not words.is_empty() and before > 0:
				words[-1]["end"] = out.rstrip(" ").length()
			continue
		var nxt := ""
		for j in range(i + 1, tokens.size()):
			if _is_word(tokens[j]):
				nxt = tokens[j]
				break
		var ph := _word_en(tk, lex, nxt, oov)
		if ph != "":
			words.append({"text": tk, "start": out.length(),
					"end": out.length() + ph.length()})
			out += ph + " "

	return _trimmed(out, words, oov)


## `strip_edges()` on the phoneme string would silently shift every recorded
## offset, so the leading trim has to be applied to the spans as well.
static func _trimmed(out: String, words: Array,
		oov: PackedStringArray) -> Dictionary:
	var lead := out.length() - out.lstrip(" \t\n").length()
	if lead > 0:
		for w in words:
			w["start"] -= lead
			w["end"] -= lead
	var phonemes := out.strip_edges()
	for w in words:
		w["start"] = clampi(w["start"], 0, phonemes.length())
		w["end"] = clampi(w["end"], w["start"], phonemes.length())
	return {"phonemes": phonemes, "oov": oov, "words": words}


func _word_en(word: String, lex: Dictionary, next_word: String,
		oov: PackedStringArray) -> String:
	var lower := word.to_lower()

	# Function words whose sound depends on what follows.
	var next_vowel := next_word != "" and next_word[0].to_lower() in "aeiou"
	match lower:
		"a": return "ɐ"
		"the": return "ði" if next_vowel else "ðə"
		"to": return "tu" if next_vowel else "tə"

	var ph := _lex_lookup(word, lex)
	if ph != "":
		return ph
	ph = _suffix_lookup(lower, lex)
	if ph != "":
		return ph

	# An unknown all-caps word is an initialism, not a word: "GPU" is three
	# letter names, not a rhyme with "goo". The lexicon has every letter.
	if _is_acronym(word):
		ph = _spell_out(word, lex)
		if ph != "":
			return ph

	oov.append(word)
	return _lts_en(lower)


static func _is_acronym(word: String) -> bool:
	if word.length() < 2 or word != word.to_upper():
		return false
	for ch in word:
		if not (ch >= "A" and ch <= "Z"):
			return false
	return true


## Letter-by-letter pronunciation, with the stress kept only on the last
## letter — that is how initialisms are actually said.
func _spell_out(word: String, lex: Dictionary) -> String:
	var parts := PackedStringArray()
	for ch in word:
		var p := _lex_lookup(ch, lex)
		if p == "":
			return ""
		parts.append(p)
	var out := ""
	for i in parts.size():
		var p: String = parts[i]
		if i < parts.size() - 1:
			p = p.replace("ˈ", "ˌ")
		out += p
	return out


## Tries the word as written, then lowercase, then Capitalized (the lexicon
## keeps proper nouns and acronyms cased).
func _lex_lookup(word: String, lex: Dictionary) -> String:
	var lower := word.to_lower()
	var caps := lower
	if caps.length() > 0:
		caps = caps[0].to_upper() + caps.substr(1)
	for cand in [word, lower, caps, word.to_upper()]:
		if lex.has(cand):
			var v = lex[cand]
			if v is String:
				return v
			if v is Dictionary:
				var d = v.get("DEFAULT")
				if d is String:
					return d
				for k in v:
					if v[k] is String:
						return v[k]
	return ""


## Regular inflections of words the lexicon does know: plural/possessive -s,
## -ing, -ed, -ly.
func _suffix_lookup(w: String, lex: Dictionary) -> String:
	var base := ""
	if w.ends_with("'s"):
		base = _lex_lookup(w.substr(0, w.length() - 2), lex)
		return base + _s_sound(base) if base != "" else ""
	if w.ends_with("ies") and w.length() > 4:
		base = _lex_lookup(w.substr(0, w.length() - 3) + "y", lex)
		return base + "z" if base != "" else ""
	if w.ends_with("es"):
		base = _lex_lookup(w.substr(0, w.length() - 2), lex)
		if base != "":
			return base + _s_sound(base)
	if w.ends_with("s") and not w.ends_with("ss"):
		base = _lex_lookup(w.substr(0, w.length() - 1), lex)
		return base + _s_sound(base) if base != "" else ""
	if w.ends_with("ing") and w.length() > 4:
		var stem := w.substr(0, w.length() - 3)
		base = _lex_lookup(stem, lex)
		if base == "":
			base = _lex_lookup(stem + "e", lex)      # making -> make
		if base == "" and stem.length() > 2 \
				and stem[stem.length() - 1] == stem[stem.length() - 2]:
			base = _lex_lookup(stem.substr(0, stem.length() - 1), lex)
		return base + "ɪŋ" if base != "" else ""
	if w.ends_with("ed") and w.length() > 3:
		base = _lex_lookup(w.substr(0, w.length() - 2), lex)   # walked -> walk
		if base == "":
			base = _lex_lookup(w.substr(0, w.length() - 1), lex)  # loved -> love
		return base + _ed_sound(base) if base != "" else ""
	if w.ends_with("ly") and w.length() > 3:
		base = _lex_lookup(w.substr(0, w.length() - 2), lex)
		return base + "li" if base != "" else ""
	return ""


static func _s_sound(ph: String) -> String:
	if ph.is_empty():
		return "z"
	var last := ph[ph.length() - 1]
	if last in "szʃʒʧʤ":
		return "ɪz"
	if last in "ptkfθ":
		return "s"
	return "z"


static func _ed_sound(ph: String) -> String:
	if ph.is_empty():
		return "d"
	var last := ph[ph.length() - 1]
	if last in "td":
		return "ɪd"
	if last in "pkfθsʃʧ":
		return "t"
	return "d"


## Letter-to-sound fallback for out-of-vocabulary English words. Best effort:
## longest-match digraphs, soft c/g, magic-e, stress on the first vowel.
func _lts_en(word: String) -> String:
	var w := word
	var long_vowel_at := -1

	# magic e: "cake" -> long a, drop the e
	var n := w.length()
	if n >= 3 and w[n - 1] == "e" and not (w[n - 2] in "aeiouy") \
			and w[n - 3] in "aeiou":
		long_vowel_at = n - 3
		w = w.substr(0, n - 1)

	const DIGRAPHS := [
		["tch", "ʧ"], ["igh", "I"], ["dge", "ʤ"],
		["tion", "ʃən"], ["sion", "ʒən"], ["ough", "O"],
		["ch", "ʧ"], ["sh", "ʃ"], ["th", "θ"], ["ph", "f"], ["wh", "w"],
		["ck", "k"], ["qu", "kw"], ["ng", "ŋ"],
		["ee", "i"], ["ea", "i"], ["oo", "u"], ["ou", "W"], ["ow", "W"],
		["ai", "A"], ["ay", "A"], ["ei", "A"], ["ey", "A"], ["oa", "O"],
		["oi", "Y"], ["oy", "Y"], ["au", "ɔ"], ["aw", "ɔ"],
		["ar", "ɑɹ"], ["or", "ɔɹ"], ["er", "ɜɹ"], ["ir", "ɜɹ"], ["ur", "ɜɹ"],
	]
	const LONG := {"a": "A", "e": "i", "i": "I", "o": "O", "u": "u"}
	const SHORT := {"a": "æ", "e": "ɛ", "i": "ɪ", "o": "ɑ", "u": "ʌ"}

	var out := ""
	var i := 0
	while i < w.length():
		# initial silent letters
		if i == 0 and w.begins_with("kn"):
			out += "n"
			i += 2
			continue
		if i == 0 and w.begins_with("wr"):
			out += "ɹ"
			i += 2
			continue
		if i == long_vowel_at and w[i] in "aeiou":
			out += LONG[w[i]]
			i += 1
			continue
		var matched := false
		for pair in DIGRAPHS:
			var g: String = pair[0]
			if w.substr(i, g.length()) == g:
				out += pair[1]
				i += g.length()
				matched = true
				break
		if matched:
			continue
		var c := w[i]
		var nxt := w[i + 1] if i + 1 < w.length() else ""
		match c:
			"a", "e", "i", "o", "u":
				out += SHORT[c]
			"c":
				out += "s" if nxt in "eiy" else "k"
			"g":
				out += "ʤ" if nxt in "eiy" else "ɡ"
			"x":
				out += "ks"
			"y":
				out += "j" if i == 0 else ("i" if i == w.length() - 1 else "ɪ")
			"j":
				out += "ʤ"
			"r":
				out += "ɹ"
			"q":
				out += "k"
			_:
				if c in "bdfhklmnpstvwz":
					out += c
		i += 1

	return _stress_first_vowel(out)


static func _stress_first_vowel(ph: String) -> String:
	for i in ph.length():
		if ph[i] in VOWELS:
			return ph.substr(0, i) + "ˈ" + ph.substr(i)
	return ph


# --------------------------------------------------------- English normalizer

var _re_cache: Dictionary = {}

func _re(pattern: String) -> RegEx:
	if not _re_cache.has(pattern):
		var r := RegEx.new()
		r.compile(pattern)
		_re_cache[pattern] = r
	return _re_cache[pattern]


## Substitute every match of `pattern` using a callable on the RegExMatch.
func _sub(text: String, pattern: String, fn: Callable) -> String:
	var re := _re(pattern)
	var out := ""
	var pos := 0
	var m := re.search(text, pos)
	while m != null:
		out += text.substr(pos, m.get_start() - pos) + str(fn.call(m))
		pos = m.get_end()
		if pos >= text.length():
			break
		m = re.search(text, pos)
	return out + text.substr(pos)


func _normalize_en(text: String) -> String:
	var t := text
	t = t.replace("’", "'").replace("‘", "'")
	t = t.replace("“", "\"").replace("”", "\"")

	# abbreviations (before sentence handling eats the dots)
	const ABBREV := {
		"mr": "mister", "mrs": "missus", "ms": "miss", "dr": "doctor",
		"prof": "professor", "st": "saint", "vs": "versus",
		"etc": "etcetera", "no": "number", "dept": "department",
	}
	t = _sub(t, "\\b([A-Za-z]+)\\.(?=\\s+[a-z0-9])", func(m):
		var w: String = m.get_string(1)
		return ABBREV.get(w.to_lower(), w + ".") if w.to_lower() in ABBREV \
				else w + ".")
	t = _sub(t, "\\b(Mr|Mrs|Ms|Dr|Prof|St|vs)\\.?\\b", func(m):
		return ABBREV[m.get_string(1).to_lower()])
	t = t.replace("e.g.", "for example").replace("i.e.", "that is")

	t = _sub(t, "(\\d),(\\d)", func(m):
		return m.get_string(1) + m.get_string(2))

	# currency
	t = _sub(t, "\\$(\\d+)\\.(\\d\\d)", func(m):
		return "%s dollars and %s cents" % [_int_words(int(m.get_string(1))),
				_int_words(int(m.get_string(2)))])
	t = _sub(t, "\\$(\\d+)", func(m):
		var v := int(m.get_string(1))
		return _int_words(v) + (" dollar" if v == 1 else " dollars"))
	t = _sub(t, "£(\\d+)", func(m):
		var v := int(m.get_string(1))
		return _int_words(v) + (" pound" if v == 1 else " pounds"))
	t = _sub(t, "€(\\d+)", func(m):
		var v := int(m.get_string(1))
		return _int_words(v) + (" euro" if v == 1 else " euros"))

	# times: 3:05, 12:30
	t = _sub(t, "\\b(\\d{1,2}):(\\d{2})\\b", func(m):
		var h := int(m.get_string(1))
		var mn := int(m.get_string(2))
		if mn == 0:
			return _int_words(h) + " o'clock"
		if mn < 10:
			return "%s oh %s" % [_int_words(h), _int_words(mn)]
		return "%s %s" % [_int_words(h), _int_words(mn)])

	# ordinals and percentages
	t = _sub(t, "\\b(\\d+)(st|nd|rd|th)\\b", func(m):
		return _ordinal_words(int(m.get_string(1))))
	t = _sub(t, "(\\d+(?:\\.\\d+)?)\\s*%", func(m):
		return m.get_string(1) + " percent")

	# decimals: digits after the point are read out one by one
	t = _sub(t, "\\b(\\d+)\\.(\\d+)\\b", func(m):
		var frac := ""
		for ch in m.get_string(2):
			frac += " " + _int_words(int(ch))
		return _int_words(int(m.get_string(1))) + " point" + frac)

	# years read in pairs: 1984 -> nineteen eighty-four, 2024 -> twenty twenty-four
	t = _sub(t, "\\b(1[1-9]\\d\\d)\\b", func(m):
		var v := int(m.get_string(1))
		var lo := v % 100
		var hi := v / 100
		if lo == 0:
			return _int_words(hi) + " hundred"
		if lo < 10:
			return "%s oh %s" % [_int_words(hi), _int_words(lo)]
		return "%s %s" % [_int_words(hi), _int_words(lo)])
	t = _sub(t, "\\b20([1-9]\\d)\\b", func(m):
		return "twenty " + _int_words(int(m.get_string(1))))

	# whatever integers remain
	t = _sub(t, "\\b\\d+\\b", func(m):
		return _int_words(int(m.get_string())))

	t = t.replace("&", " and ").replace("+", " plus ").replace("@", " at ")
	t = t.replace("=", " equals ").replace("/", " slash ")
	return t


const _ONES := ["zero", "one", "two", "three", "four", "five", "six",
		"seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
		"fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
const _TENS := ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
		"seventy", "eighty", "ninety"]


static func _int_words(v: int) -> String:
	if v < 0:
		return "minus " + _int_words(-v)
	if v < 20:
		return _ONES[v]
	if v < 100:
		var s = _TENS[v / 10]
		return s if v % 10 == 0 else s + " " + _ONES[v % 10]
	if v < 1000:
		var s = _ONES[v / 100] + " hundred"
		return s if v % 100 == 0 else s + " and " + _int_words(v % 100)
	for pair in [[1000000000000, "trillion"], [1000000000, "billion"],
			[1000000, "million"], [1000, "thousand"]]:
		var unit: int = pair[0]
		if v >= unit:
			var s: String = _int_words(v / unit) + " " + pair[1]
			return s if v % unit == 0 else s + " " + _int_words(v % unit)
	return _ONES[0]


static func _ordinal_words(v: int) -> String:
	const IRREGULAR := {1: "first", 2: "second", 3: "third", 5: "fifth",
			8: "eighth", 9: "ninth", 12: "twelfth"}
	if v in IRREGULAR:
		return IRREGULAR[v]
	if v < 20:
		return _ONES[v] + "th"
	if v % 10 == 0 and v < 100:
		return _TENS[v / 10].trim_suffix("y") + "ieth"
	if v < 100:
		return _TENS[v / 10] + " " + _ordinal_words(v % 10)
	# "one hundredth", "two thousandth", "one hundred and twenty-first"
	if v % 100 == 0:
		return _int_words(v) + "th"
	var words := _int_words(v - v % 100)
	var rest := v % 100
	return words + " and " + _ordinal_words(rest)


func _tokenize(text: String) -> PackedStringArray:
	var out := PackedStringArray()
	for m in _re("[A-Za-zÀ-ÿ']+|[.,!?;:\"…—()]").search_all(text):
		out.append(m.get_string())
	return out


static func _is_word(tk: String) -> bool:
	return tk.length() > 0 and not (tk[0] in ".,!?;:\"…—()")


# --------------------------------------------- Spanish / Italian / Portuguese

## Rule-based G2P for phonetically regular orthographies. Best effort: enough
## for intelligible speech, not linguistic fidelity.
func _phonemize_rules(text: String, lang: String) -> Dictionary:
	var out := ""
	var words := []
	for tk in _tokenize(text):
		if not _is_word(tk):
			out = out.rstrip(" ") + tk + " "
			if not words.is_empty():
				words[-1]["end"] = out.rstrip(" ").length()
			continue
		var ph := ""
		match lang:
			"es": ph = _word_es(tk.to_lower())
			"it": ph = _word_it(tk.to_lower())
			"pt": ph = _word_pt(tk.to_lower())
		if ph != "":
			words.append({"text": tk, "start": out.length(),
					"end": out.length() + ph.length()})
			out += ph + " "
	return _trimmed(out, words, PackedStringArray())


func _word_es(w: String) -> String:
	var out := ""
	var stressed := false
	var i := 0
	while i < w.length():
		var c := w[i]
		var nxt := w[i + 1] if i + 1 < w.length() else ""
		var two := c + nxt
		match two:
			"ch": out += "ʧ"; i += 2; continue
			"ll": out += "ʝ"; i += 2; continue
			"rr": out += "r"; i += 2; continue
			"qu": out += "k"; i += 2; continue
			"gu":
				if i + 2 < w.length() and w[i + 2] in "ei":
					out += "ɡ"
					i += 2
					continue
		match c:
			"á": out += "ˈa"; stressed = true
			"é": out += "ˈe"; stressed = true
			"í": out += "ˈi"; stressed = true
			"ó": out += "ˈo"; stressed = true
			"ú": out += "ˈu"; stressed = true
			"ñ": out += "ɲ"
			"c": out += "s" if nxt in "ei" else "k"
			"g": out += "x" if nxt in "ei" else "ɡ"
			"h": pass
			"j": out += "x"
			"v": out += "b"
			"z": out += "s"
			"y": out += "i" if nxt == "" else "ʝ"
			"r": out += "r" if i == 0 else "ɾ"
			"x": out += "ks"
			_:
				if c in "abdefiklmnopstuw":
					out += c
		i += 1
	return out if stressed else _stress_default(out, w)


func _word_it(w: String) -> String:
	var out := ""
	var stressed := false
	var i := 0
	while i < w.length():
		var c := w[i]
		var nxt := w[i + 1] if i + 1 < w.length() else ""
		var three := w.substr(i, 3)
		var two := c + nxt
		if three == "gli":
			out += "ʎ"
			i += 3
			continue
		if two == "gn":
			out += "ɲ"
			i += 2
			continue
		if two == "ch":
			out += "k"
			i += 2
			continue
		if two == "gh":
			out += "ɡ"
			i += 2
			continue
		if two == "sc" and i + 2 < w.length() and w[i + 2] in "ei":
			out += "ʃ"
			i += 2
			continue
		if two == "qu":
			out += "kw"
			i += 2
			continue
		match c:
			"à": out += "ˈa"; stressed = true
			"è", "é": out += "ˈe"; stressed = true
			"ì": out += "ˈi"; stressed = true
			"ò", "ó": out += "ˈo"; stressed = true
			"ù": out += "ˈu"; stressed = true
			"c": out += "ʧ" if nxt in "ei" else "k"
			"g": out += "ʤ" if nxt in "ei" else "ɡ"
			"h": pass
			"z": out += "ts"
			"r": out += "r"
			_:
				if c in "abdefiklmnopstuvw":
					out += c
		i += 1
	return out if stressed else _stress_default(out, w)


func _word_pt(w: String) -> String:
	var out := ""
	var stressed := false
	var i := 0
	while i < w.length():
		var c := w[i]
		var nxt := w[i + 1] if i + 1 < w.length() else ""
		var two := c + nxt
		match two:
			"nh": out += "ɲ"; i += 2; continue
			"lh": out += "ʎ"; i += 2; continue
			"ch": out += "ʃ"; i += 2; continue
			"ss": out += "s"; i += 2; continue
			"rr": out += "ʁ"; i += 2; continue
			"ão": out += "ɐ̃w"; i += 2; continue
			"õe": out += "õj"; i += 2; continue
			"ãe": out += "ɐ̃j"; i += 2; continue
			"qu": out += "k"; i += 2; continue
		match c:
			"á", "â": out += "ˈa"; stressed = true
			"é", "ê": out += "ˈe"; stressed = true
			"í": out += "ˈi"; stressed = true
			"ó", "ô": out += "ˈo"; stressed = true
			"ú": out += "ˈu"; stressed = true
			"ã": out += "ɐ̃"
			"ç": out += "s"
			"x": out += "ʃ"
			"j": out += "ʒ"
			"g": out += "ʒ" if nxt in "ei" else "ɡ"
			"h": pass
			"s":
				var prev := w[i - 1] if i > 0 else ""
				out += "z" if (prev in "aeiouáéíóúâêô" and nxt in "aeiou") else "s"
			"r": out += "ʁ" if i == 0 else "ɾ"
			"o": out += "u" if nxt == "" else "o"
			"e": out += "i" if nxt == "" else "e"
			_:
				if c in "abdfiklmnptuvwz":
					out += c
		i += 1
	return out if stressed else _stress_default(out, w)


## Default Romance stress: penultimate vowel when the word ends in a vowel,
## n or s; final vowel otherwise.
static func _stress_default(ph: String, orth: String) -> String:
	var vowel_idx := PackedInt32Array()
	for i in ph.length():
		if ph[i] in "aeiou":
			vowel_idx.append(i)
	if vowel_idx.is_empty():
		return ph
	var last := orth[orth.length() - 1] if orth.length() > 0 else ""
	var target: int
	if last in "aeiouns" and vowel_idx.size() >= 2:
		target = vowel_idx[vowel_idx.size() - 2]
	else:
		target = vowel_idx[vowel_idx.size() - 1]
	return ph.substr(0, target) + "ˈ" + ph.substr(target)


# ------------------------------------------------------------------- lexicons

func _load_lexicon(name: String) -> Dictionary:
	if _lexicons.has(name):
		return _lexicons[name]
	var path := LEXICON_DIR + name + ".json"
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("KokoroG2P: cannot open %s" % path)
		_lexicons[name] = {}
		return {}
	var t0 := Time.get_ticks_msec()
	var parsed = JSON.parse_string(f.get_as_text())
	f.close()
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("KokoroG2P: %s is not a JSON object" % path)
		parsed = {}
	_lexicons[name] = parsed
	print("KokoroG2P: %s — %d entries in %d ms"
			% [name, parsed.size(), Time.get_ticks_msec() - t0])
	return parsed
