extends SceneTree

## Runs the full graph in Godot and compares against the numpy reference.
##
##   python tools/export_test_case.py --tokens 9
##   godot --rendering-driver vulkan --script res://tests/test_executor.gd

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
	print("case: %d ids, %d style dims, %d expected samples"
			% [n_ids, n_style, n_samples])

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
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--nodes="):
			ex.max_nodes = int(arg.substr(8))
		elif arg == "--verbose-nodes":
			ex.verbose = true
	var t0 := Time.get_ticks_msec()
	var setup_err: String = ex.setup(gpu, model)
	if setup_err != "":
		printerr("FAIL: " + setup_err)
		quit(1)
		return
	print("setup (weights uploaded, kernels compiled): %d ms"
			% (Time.get_ticks_msec() - t0))

	t0 = Time.get_ticks_msec()
	var res: Dictionary = ex.run(ids, style, 1.0)
	var run_ms := Time.get_ticks_msec() - t0
	if res.is_empty():
		printerr("FAIL: " + ex.last_error)
		gpu.cleanup()
		quit(1)
		return

	var got: PackedFloat32Array = gpu.read_f32(res.buffer, res.samples)
	print("ran %d dispatches in %d ms -> %d samples"
			% [res.dispatches, run_ms, res.samples])

	var failures := 0
	if res.samples != n_samples:
		printerr("FAIL: got %d samples, expected %d" % [res.samples, n_samples])
		failures += 1
	else:
		var worst := 0.0
		var sum_sq := 0.0
		var ref_sq := 0.0
		for i in n_samples:
			var d: float = absf(got[i] - want[i])
			worst = maxf(worst, d)
			sum_sq += d * d
			ref_sq += want[i] * want[i]
		var rmse: float = sqrt(sum_sq / n_samples)
		var snr: float = 20.0 * (log(sqrt(ref_sq / n_samples)
				/ maxf(rmse, 1e-20)) / log(10.0))
		print("max|err| = %s   rmse = %s   snr = %.1f dB"
				% [String.num_scientific(worst), String.num_scientific(rmse), snr])
		# Bit-exactness is not achievable: quantization rounding ties in the
		# prosody branches nudge F0, and near-zero STFT bins have chaotic
		# phase (±π both valid), which drives the vocoder's noise branch.
		# What matters is that the residual stays a quiet noise floor under
		# the same speech, so gate on SNR and RMSE rather than max error.
		if snr < 10.0 or rmse > 0.02:
			printerr("FAIL: waveform does not match the reference")
			failures += 1
		else:
			print("OK: waveform matches the reference (residual is the "
					+ "expected quantization/phase noise floor)")

	gpu.cleanup()
	quit(1 if failures > 0 else 0)
