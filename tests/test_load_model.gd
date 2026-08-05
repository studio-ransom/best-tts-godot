extends SceneTree

## Headless smoke test for the packed-model loader.
##
##   godot --headless --script res://tests/test_load_model.gd

const MODEL_PATH := "res://addons/best-tts/assets/model.kpk"

const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")


func _init() -> void:
	var failures := 0
	var t0 := Time.get_ticks_msec()

	var model = KpkModelScript.new()
	var err: int = model.load_from(MODEL_PATH)
	var load_ms := Time.get_ticks_msec() - t0

	if err != OK:
		printerr("FAIL: load_from returned %d" % err)
		quit(1)
		return

	print("loaded in %d ms" % load_ms)
	print("  nodes          %d" % model.nodes.size())
	print("  weights        %d tensors" % model.weights.size())
	print("  tensor names   %d" % model.tensor_names.size())
	print("  op types       %d" % model.ops.size())
	print("  blob           %.1f MB" % (model.blob.size() / 1e6))

	failures += _expect(model.nodes.size() == 3945, "node count", model.nodes.size())
	failures += _expect(model.weights.size() == 781, "weight count", model.weights.size())
	failures += _expect(model.ops.size() > 40, "op type count", model.ops.size())
	failures += _expect(model.sample_rate == 24000, "sample rate", model.sample_rate)
	failures += _expect(model.samples_per_frame == 300, "samples/frame",
			model.samples_per_frame)
	failures += _expect(model.input_ids_tid >= 0 and model.style_tid >= 0
			and model.speed_tid >= 0 and model.output_tid >= 0,
			"graph io ids", "ok")
	failures += _expect(model.name_of(model.output_tid) == "waveform",
			"output name", model.name_of(model.output_tid))

	# CPU/GPU split must match what the packer reported.
	var cpu := 0
	for n in model.nodes:
		if n.place == KpkModelScript.Place.CPU:
			cpu += 1
	failures += _expect(cpu == 409, "cpu node count", cpu)
	print("  placement      %d cpu / %d gpu" % [cpu, model.nodes.size() - cpu])

	# Spot-check a weight: the token embedding table is int8 [178, 128].
	var emb_tid := -1
	for tid in model.weights:
		var w = model.weights[tid]
		if w.shape.size() == 2 and w.shape[0] == 178 and w.shape[1] == 128:
			emb_tid = tid
			break
	failures += _expect(emb_tid != -1, "found [178,128] embedding", emb_tid)
	if emb_tid != -1:
		var w = model.weights[emb_tid]
		failures += _expect(w.dtype == KpkModelScript.DType.I8,
				"embedding dtype is int8", w.dtype)
		failures += _expect(w.nbytes == 178 * 128, "embedding byte count", w.nbytes)
		failures += _expect(model.weight_bytes(emb_tid).size() == 178 * 128,
				"embedding slice size", model.weight_bytes(emb_tid).size())

	# Every node input must resolve to a weight or be produced by an earlier
	# node -- this is what lets the executor run in a single forward pass.
	var produced := {}
	produced[model.input_ids_tid] = true
	produced[model.style_tid] = true
	produced[model.speed_tid] = true
	for tid in model.weights:
		produced[tid] = true
	var unresolved := 0
	for n in model.nodes:
		for tid in n.inputs:
			if tid >= 0 and not produced.has(tid):
				if unresolved < 5:
					printerr("  unresolved input %s for %s" % [model.name_of(tid), n])
				unresolved += 1
		for tid in n.outputs:
			if tid >= 0:
				produced[tid] = true
	failures += _expect(unresolved == 0, "unresolved inputs", unresolved)
	failures += _expect(produced.has(model.output_tid), "waveform is produced", true)

	if failures == 0:
		print("\nOK: all loader checks passed")
	else:
		printerr("\n%d check(s) FAILED" % failures)
	quit(1 if failures > 0 else 0)


func _expect(cond: bool, what: String, got) -> int:
	if cond:
		return 0
	printerr("FAIL: %s (got %s)" % [what, got])
	return 1
