extends SceneTree

## Finds the first tensor where the Godot executor disagrees with the numpy
## reference. Both run the same graph, so the earliest divergence is the bug.
##
##   python tools/export_taps.py --tokens 9
##   godot --rendering-driver vulkan --script res://tests/test_taps.gd

const GPUScript := preload("res://addons/best-tts/runtime/gpu.gd")
const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")
const ExecutorScript := preload("res://addons/best-tts/runtime/executor.gd")

const TAPS := "res://tests/_taps.bin"
const CASE := "res://tests/_case.bin"
const MODEL := "res://addons/best-tts/assets/model.kpk"

## Report at most this many divergences before stopping.
const MAX_REPORT := 40
## Loose enough to ignore accumulated quantization rounding (~0.5% in the
## prosody branches), tight enough to catch real bugs.
const TOL := 2e-1
## Relative RMS is the metric that catches a *systematic* error. A peak-relative
## max like TOL is blind to a whole tensor being uniformly 1.6x too large in a
## band that carries little of the peak — which is exactly how the vocoder's
## high-frequency excess hid here for a whole session.
const RMS_TOL := 5e-2


func _init() -> void:
	var cf := FileAccess.open(CASE, FileAccess.READ)
	if cf == null:
		printerr("run tools/export_test_case.py first")
		quit(1)
		return
	var n_ids := cf.get_32()
	var n_style := cf.get_32()
	cf.get_32()
	var ids := PackedInt32Array()
	for i in n_ids:
		ids.append(cf.get_32())
	var style := cf.get_buffer(n_style * 4).to_float32_array()
	cf.close()

	var tf := FileAccess.open(TAPS, FileAccess.READ)
	if tf == null:
		printerr("run tools/export_taps.py first")
		quit(1)
		return
	var count := tf.get_32()
	var names := PackedStringArray()
	var want := {}
	for i in count:
		var nlen := tf.get_32()
		var nm := tf.get_buffer(nlen).get_string_from_utf8()
		var n := tf.get_32()
		want[nm] = tf.get_buffer(n * 4).to_float32_array()
		names.append(nm)
	tf.close()
	print("loaded %d reference taps" % count)

	var gpu = GPUScript.new()
	var err: String = gpu.initialize()
	if err != "":
		printerr("FAIL: " + err)
		quit(1)
		return
	var model = KpkModelScript.new()
	model.load_from(MODEL)

	# Map names to tensor ids and ask the executor to keep those buffers.
	var tid_of := {}
	for i in model.tensor_names.size():
		tid_of[model.tensor_names[i]] = i

	var ex = ExecutorScript.new()
	for nm in names:
		if tid_of.has(nm):
			ex.trace_tids[tid_of[nm]] = true
	var setup_err: String = ex.setup(gpu, model)
	if setup_err != "":
		printerr("FAIL: " + setup_err)
		quit(1)
		return

	var res: Dictionary = ex.run(ids, style, 1.0)
	if res.is_empty():
		printerr("run failed: " + ex.last_error)

	# Walk in graph order and report the earliest mismatches.
	var reported := 0
	var checked := 0
	var order := []
	for node in model.nodes:
		for tid in node.outputs:
			if tid >= 0:
				order.append(tid)

	for tid in order:
		var nm: String = model.name_of(tid)
		if not want.has(nm) or not ex.has_value(tid):
			continue
		# Packed uint8 tensors are unpacked byte-by-byte and allowed one LSB of
		# slack: a rounding tie in quantization can land either way.
		var packed: bool = ex.is_packed(tid)
		var got: PackedFloat32Array = ex.value_of_packed(tid) if packed \
				else ex.value_of(tid)
		var exp: PackedFloat32Array = want[nm]
		checked += 1
		if got.size() != exp.size():
			printerr("SIZE  %s: got %d, want %d" % [nm, got.size(), exp.size()])
			reported += 1
		else:
			var worst := 0.0
			var scale := 1e-9
			var nans := 0
			var sq_err := 0.0
			var sq_ref := 0.0
			for i in exp.size():
				# NaN fails every comparison, so maxf() would silently ignore
				# it — count it explicitly or broken tensors sail through.
				if is_nan(got[i]) or is_inf(got[i]):
					nans += 1
					continue
				var d := got[i] - exp[i]
				worst = maxf(worst, absf(d))
				scale = maxf(scale, absf(exp[i]))
				sq_err += d * d
				sq_ref += float(exp[i]) * exp[i]
			var rel_rms := sqrt(sq_err / maxf(1e-30, sq_ref))
			if nans > 0:
				printerr("NAN   %s  shape %s  %d non-finite of %d"
						% [nm, ex.shape_of(tid), nans, exp.size()])
				printerr("        got  %s" % [_head(got)])
				printerr("        want %s" % [_head(exp)])
				reported += 1
			elif (worst > 1.01) if packed \
					else (worst / scale > TOL or rel_rms > RMS_TOL):
				printerr("DIFF  %s  shape %s  relRMS %s  max|err| %s (scale %s)"
						% [nm, ex.shape_of(tid), String.num_scientific(rel_rms),
						String.num_scientific(worst),
						String.num_scientific(scale)])
				printerr("        got  %s" % [_head(got)])
				printerr("        want %s" % [_head(exp)])
				reported += 1
		if reported >= MAX_REPORT:
			break

	print("checked %d tapped tensors" % checked)
	if reported == 0:
		print("OK: every tapped tensor matches the reference")
	gpu.cleanup()
	quit(1 if reported > 0 else 0)


func _head(a: PackedFloat32Array) -> String:
	var s := ""
	for i in mini(6, a.size()):
		s += "%.5f " % a[i]
	return s.strip_edges()
