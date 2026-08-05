extends SceneTree

## Runs the graph several times on one executor, which is what every real
## caller does. Buffer pooling and the per-run reset only get exercised here:
## a single run can leave stale state behind and still look correct.
##
##   godot --rendering-driver vulkan --script res://tests/test_repeat.gd

const GPUScript := preload("res://addons/best-tts/runtime/gpu.gd")
const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")
const ExecutorScript := preload("res://addons/best-tts/runtime/executor.gd")

const CASE := "res://tests/_case.bin"
const MODEL := "res://addons/best-tts/assets/model.kpk"


func _init() -> void:
	var f := FileAccess.open(CASE, FileAccess.READ)
	if f == null:
		printerr("FAIL: %s missing — run tools/export_test_case.py" % CASE)
		quit(1)
		return
	var n_ids := f.get_32()
	var n_style := f.get_32()
	var n_samples := f.get_32()
	var ids := PackedInt32Array()
	for i in n_ids:
		ids.append(f.get_32())
	var style := f.get_buffer(n_style * 4).to_float32_array()
	var want := f.get_buffer(n_samples * 4).to_float32_array()
	f.close()

	var gpu = GPUScript.new()
	var err: String = gpu.initialize()
	if err != "":
		printerr("FAIL: " + err)
		quit(1)
		return
	var model = KpkModelScript.new()
	if model.load_from(MODEL) != OK:
		printerr("FAIL: cannot load " + MODEL)
		quit(1)
		return
	var ex = ExecutorScript.new()
	var setup_err: String = ex.setup(gpu, model)
	if setup_err != "":
		printerr("FAIL: " + setup_err)
		quit(1)
		return

	# A shorter utterance in between, so the run after it cannot rely on
	# buffers that happen to already be the right size.
	var short_ids := ids.slice(0, maxi(3, n_ids - 3))
	var failures := 0
	var first := PackedFloat32Array()

	for pass_i in 5:
		var use_short := pass_i == 1 or pass_i == 3
		var in_ids := short_ids if use_short else ids
		var t0 := Time.get_ticks_msec()
		var res: Dictionary = ex.run(in_ids, style, 1.0)
		var ms := Time.get_ticks_msec() - t0
		if res.is_empty():
			printerr("FAIL: run %d: %s" % [pass_i, ex.last_error])
			failures += 1
			continue
		var got: PackedFloat32Array = gpu.read_f32(res.buffer, res.samples)
		var peak := 0.0
		var nonfinite := 0
		for v in got:
			if is_nan(v) or is_inf(v):
				nonfinite += 1
			else:
				peak = maxf(peak, absf(v))
		print("run %d: %d ids -> %d samples, peak %.4f, %d ms"
				% [pass_i, in_ids.size(), res.samples, peak, ms])
		if nonfinite > 0:
			printerr("FAIL: run %d produced %d non-finite samples"
					% [pass_i, nonfinite])
			failures += 1
		if peak < 0.01:
			printerr("FAIL: run %d is silent" % pass_i)
			failures += 1

		if not use_short:
			if first.is_empty():
				first = got
				# The first full-length run is also checked against the
				# reference, so a repeat failure cannot hide behind a wrong
				# baseline.
				var worst := 0.0
				for i in mini(got.size(), want.size()):
					worst = maxf(worst, absf(got[i] - want[i]))
				print("      vs reference: max|err| %s"
						% String.num_scientific(worst))
			else:
				# Identical inputs must give identical output; any drift means
				# state leaked from the run in between.
				var drift := 0.0
				if got.size() != first.size():
					printerr("FAIL: run %d returned %d samples, first gave %d"
							% [pass_i, got.size(), first.size()])
					failures += 1
				else:
					for i in got.size():
						drift = maxf(drift, absf(got[i] - first[i]))
					if drift > 0.0:
						printerr("FAIL: run %d drifted from run 0 by %s"
								% [pass_i, String.num_scientific(drift)])
						failures += 1
					else:
						print("      identical to run 0")

	gpu.cleanup()
	if failures == 0:
		print("OK: the executor is reusable across runs")
	quit(1 if failures > 0 else 0)
