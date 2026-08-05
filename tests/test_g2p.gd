extends SceneTree

## Checks the text -> IPA frontend: normalization, lexicon lookup, the
## out-of-vocabulary fallback, and that every symbol produced is one the model
## actually knows.
##
##   godot --headless --script res://tests/test_g2p.gd

const G2PScript := preload("res://addons/best-tts/g2p/g2p.gd")
const TokenizerScript := preload("res://addons/best-tts/runtime/tokenizer.gd")

var _failures := 0


func _init() -> void:
	var g2p = G2PScript.new()
	var tok = TokenizerScript.new()
	if not tok.load_vocab():
		printerr("FAIL: vocab.json missing")
		quit(1)
		return

	print("— English —")
	for text in [
		"Hello from Godot.",
		"This is Kokoro speaking, running on your GPU.",
		"The quick brown fox jumps over the lazy dog.",
		"I paid $42.50 on the 3rd of May at 4:15.",
		"Dr. Smith owns 1,200 books and 15% of the shop.",
		"In 1984 they walked, talked and laughed loudly.",
		"Zorblax the quintaverse defenestrated my kerblonker.",
	]:
		_check(g2p, tok, text, "af_heart")

	print("\n— British —")
	_check(g2p, tok, "The water in the tower is rather bitter.", "bm_george")

	print("\n— rule-based Romance —")
	_check(g2p, tok, "Hola, me llamo Carlos y vivo en España.", "ef_dora")
	_check(g2p, tok, "Ciao, mi chiamo Giovanni e sono italiano.", "if_sara")
	_check(g2p, tok, "Olá, meu nome é João e falo português.", "pf_dora")

	print("\n— normalizer units —")
	# Kokoro's alphabet writes the diphthongs as single symbols: I = /aɪ/,
	# A = /eɪ/, O = /oʊ/, W = /aʊ/.
	_expect_contains(g2p, "I have 3 cats.", "θɹ", "three")
	_expect_contains(g2p, "It costs $5.", "fˈIv", "five")
	_expect_contains(g2p, "Meet me at 2:00.", "klˈɑk", "o'clock")
	_expect_contains(g2p, "the 1st time", "fˈɜɹst", "first")
	_expect_contains(g2p, "my GPU", "ʤˌipˌijˈu", "initialism spelled out")

	# Function words that change with what follows.
	_expect_contains(g2p, "the apple", "ði", "the + vowel")
	_expect_contains(g2p, "the dog", "ðə", "the + consonant")

	print("")
	if _failures == 0:
		print("OK: G2P produces model-legal phonemes for every case")
	quit(1 if _failures > 0 else 0)


func _check(g2p, tok, text: String, voice: String) -> void:
	var r: Dictionary = g2p.phonemize(text, voice)
	if r.has("error"):
		printerr("FAIL: %s" % r["error"])
		_failures += 1
		return
	var ph: String = r["phonemes"]
	if ph.strip_edges().is_empty():
		printerr("FAIL: no phonemes for \"%s\"" % text)
		_failures += 1
		return

	# Every character must be in the model's vocabulary, or the tokenizer
	# silently drops it and the utterance comes out mangled.
	var enc: Dictionary = tok.encode(ph)
	var unknown: PackedStringArray = enc["unknown"]
	print("  \"%s\"\n    -> %s\n    (%d phonemes)" % [text, ph,
			int(enc["phoneme_count"])])
	if unknown.size() > 0:
		printerr("FAIL: %d symbol(s) outside the vocabulary: %s"
				% [unknown.size(), " ".join(unknown)])
		_failures += 1
	var oov: PackedStringArray = r["oov"]
	if oov.size() > 0:
		print("    (fell back to letter-to-sound for: %s)" % " ".join(oov))


func _expect_contains(g2p, text: String, needle: String, label: String) -> void:
	var r: Dictionary = g2p.phonemize(text, "af_heart")
	var ph: String = r.get("phonemes", "")
	if ph.contains(needle):
		print("  ok: %s -> found %s in \"%s\"" % [label, needle, ph])
	else:
		printerr("FAIL: %s — expected %s in \"%s\" (from \"%s\")"
				% [label, needle, ph, text])
		_failures += 1
