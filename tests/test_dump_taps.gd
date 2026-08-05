extends SceneTree

## Dumps named tensors from a Godot run so Python can inspect them properly.
##
##   godot --rendering-driver vulkan --script res://tests/test_dump_taps.gd \
##       --tensors=/decoder/decoder/generator/f0_upsamp/Resize_output_0,...
##
## Writes tests/out/dump.bin: int32 count, then per tensor int32 name_len,
## utf-8 name, int32 rank, int32 shape[rank], int32 n, float32[n]. Uses the
## same test case as tests/test_taps.gd, so values line up element for element
## with what tools/export_taps.py wrote.

const GPUScript := preload("res://addons/best-tts/runtime/gpu.gd")
const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")
const ExecutorScript := preload("res://addons/best-tts/runtime/executor.gd")

const CASE := "res://tests/_case.bin"
const MODEL := "res://addons/best-tts/assets/model.kpk"
const OUT := "res://tests/out/dump.bin"


func _init() -> void:
	var names := PackedStringArray()
	for arg in OS.get_cmdline_args():
		if arg.begins_with("--tensors="):
			names = arg.substr(10).split(",", false)
	if names.is_empty():
		printerr("pass --tensors=name1,name2")
		quit(1)
		return

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

	var gpu = GPUScript.new()
	var err: String = gpu.initialize()
	if err != "":
		printerr("FAIL: " + err)
		quit(1)
		return
	var model = KpkModelScript.new()
	model.load_from(MODEL)

	var tid_of := {}
	for i in model.tensor_names.size():
		tid_of[model.tensor_names[i]] = i

	var ex = ExecutorScript.new()
	var wanted := []
	for nm in names:
		if not tid_of.has(nm):
			printerr("unknown tensor: %s" % nm)
			continue
		ex.trace_tids[tid_of[nm]] = true
		wanted.append([nm, tid_of[nm]])
	var setup_err: String = ex.setup(gpu, model)
	if setup_err != "":
		printerr("FAIL: " + setup_err)
		quit(1)
		return

	var res: Dictionary = ex.run(ids, style, 1.0)
	if res.is_empty():
		printerr("run failed: " + ex.last_error)

	DirAccess.make_dir_recursive_absolute("res://tests/out/")
	var f := FileAccess.open(OUT, FileAccess.WRITE)
	f.store_32(wanted.size())
	for w in wanted:
		var nm: String = w[0]
		var tid: int = w[1]
		var vals: PackedFloat32Array = ex.value_of_packed(tid) \
				if ex.is_packed(tid) else ex.value_of(tid)
		var shape: PackedInt32Array = ex.shape_of(tid)
		var nb := nm.to_utf8_buffer()
		f.store_32(nb.size())
		f.store_buffer(nb)
		f.store_32(shape.size())
		for d in shape:
			f.store_32(d)
		f.store_32(vals.size())
		f.store_buffer(vals.to_byte_array())
		print("%s  %s  %d values" % [nm, shape, vals.size()])
	f.close()
	print("wrote %s" % OUT)

	# Taps must be read before the device goes away.
	gpu.cleanup()
	quit(0)
