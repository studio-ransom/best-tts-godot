@tool
class_name KokoroExecutor
extends RefCounted

## Runs the packed Kokoro graph: CPU nodes in GDScript, GPU nodes as compute
## dispatches, with the two unavoidable readbacks (predicted durations, and the
## iSTFT overlap-add index set) splitting the work into submissions.
##
## Semantics follow tools/ref_runtime.py, which is verified op-by-op against
## onnxruntime. fp16 is a storage format only; all compute is fp32.

const Ops := preload("res://addons/best-tts/runtime/ops.gd")
const Shapes := preload("res://addons/best-tts/runtime/shapes.gd")
const CpuOps := preload("res://addons/best-tts/runtime/cpu_ops.gd")
const KpkModelScript := preload("res://addons/best-tts/runtime/kpk_model.gd")

const KERNELS := [
	"elementwise", "copy", "gather", "reduce", "layernorm", "softmax",
	"matmul", "minmax", "quantize", "matmul_int8", "conv_int8", "conv",
	"lstm", "misc", "stft", "pcm16",
]

## misc.glsl modes
const M_CUMSUM := 0
const M_RESIZE_NEAREST := 1
const M_RESIZE_LINEAR := 2
const M_PAD_REFLECT := 3
const M_SCATTER_ND := 4
const M_UNPACK_F16 := 5
const M_DEQUANT_I8 := 6

var gpu
var model

## tid -> PackedInt32Array
var _shape: Dictionary = {}
## tid -> RID of a float32 buffer (or a packed byte buffer for quantized data)
var _buf: Dictionary = {}
## tid -> CpuOps.CpuTensor for values the CPU needs
var _cpu: Dictionary = {}
## tid -> true when the buffer holds packed 8-bit data rather than floats
var _packed: Dictionary = {}
## tid of the last node that reads each tensor, for buffer recycling
var _last_use: Dictionary = {}
## byte size -> Array[RID] of buffers available for reuse
var _pool: Dictionary = {}
var _live: Dictionary = {}
## RIDs currently sitting in `_pool`, so a double release cannot duplicate one
var _in_pool: Dictionary = {}
## Sum of `_pool`, maintained rather than computed: a memory readout can come
## from any thread while the engine thread is still recycling buffers, and
## walking a Dictionary another thread is erasing from throws on whichever key
## disappears mid-loop.
var _pooled_bytes := 0
## Per-run buffers that no tensor id owns; returned to the pool when a run ends
var _scratch: Array[RID] = []

## Bytes of pooled activation memory currently allocated, and the high-water
## mark since the last `reset_peak()`. Tracked incrementally because summing
## `_live` from another thread races with the worker.
var _live_bytes := 0
var peak_live_bytes := 0

var _recording := false
var _dispatches := 0
var last_error := ""
## Debug aids: stop after N nodes, and log each node as it runs.
var max_nodes := -1
var verbose := false
## Set to a tensor name to capture its value for debugging.
var trace_tids: Dictionary = {}
var traced: Dictionary = {}

## The duration predictor's rounded output, in frames, one per input token
## including the two boundary markers. Read back anyway to build the alignment,
## so exposing it costs nothing; `-1` means this graph does not have it.
const DURATION_TENSOR := "/encoder/Clip_output_0"
var duration_tid := -1
var last_durations := PackedFloat32Array()
## Ask for `last_durations` to be filled. Off by default: the tensor is only
## ever consumed on the GPU, so getting it costs one extra readback of a few
## dozen floats and holding its buffer past its last use.
var want_durations := false

## Set by the caller to abort a run in progress. Checked every
## `CANCEL_INTERVAL` nodes rather than every node: the check is a mutex lock
## and 3945 of them per utterance would cost more than the cancel saves.
var cancel_token = null
const CANCEL_INTERVAL := 64

## Nodes between forced submissions while a cancel token is set; 0 disables it.
##
## Recording is fast and execution is not: the whole graph is normally recorded
## in a few hundred milliseconds and then runs on the GPU in one submission of
## over a second. A cancel arriving during that submission cannot stop it —
## Vulkan has no way to un-queue work — so it lands at the chunk boundary
## instead. Submitting in segments bounds that wait to one segment, at the cost
## of the pipeline bubble each extra sync introduces.
var cancel_flush_nodes := 0


func setup(p_gpu, p_model) -> String:
	gpu = p_gpu
	model = p_model
	for k in KERNELS:
		if not gpu.load_kernel(k):
			return "kernel '%s' failed to load — run: godot --headless --import" % k
	_upload_weights()
	_compute_last_use()
	duration_tid = model.tensor_names.find(DURATION_TENSOR)
	return ""


# --------------------------------------------------------------- weight setup

func _upload_weights() -> void:
	var f16_jobs := []
	for tid in model.weights:
		var w = model.weights[tid]
		_shape[tid] = w.shape
		var bytes: PackedByteArray = model.weight_bytes(tid)

		match w.dtype:
			KpkModelScript.DType.F32:
				_buf[tid] = gpu.create_buffer(bytes.size(), bytes)
			KpkModelScript.DType.F16:
				var count: int = w.count()
				var raw: RID = gpu.create_buffer(bytes.size(), bytes)
				var out: RID = gpu.create_buffer(count * 4)
				f16_jobs.append([raw, out, count])
				_buf[tid] = out
			KpkModelScript.DType.I8, KpkModelScript.DType.U8:
				_buf[tid] = gpu.create_buffer(bytes.size(), bytes)
				_packed[tid] = true
			_:
				# int32 / int64 / bool constants: keep them as floats so any
				# GPU consumer sees the uniform representation.
				var vals := _weight_as_floats(w, bytes)
				_buf[tid] = gpu.create_buffer_f32(vals)

		# Anything small enough is mirrored on the CPU, because shape operands,
		# axes and slice bounds all arrive as weights.
		if w.count() <= 65536:
			_cpu[tid] = CpuOps.CpuTensor.new(w.shape,
					_weight_as_doubles(w, bytes))

	if f16_jobs.is_empty():
		return
	gpu.begin_compute()
	for job in f16_jobs:
		var push := _misc_push(M_UNPACK_F16, job[2], 1, 1, 1, job[2], 1.0, 0, 0)
		gpu.dispatch("misc", [job[1], job[1], job[0], job[1]] as Array[RID],
				push, (job[2] + 63) / 64)
	gpu.end_compute()
	gpu.submit_and_wait()
	for job in f16_jobs:
		gpu.free_buffer(job[0])


func _weight_as_floats(w, bytes: PackedByteArray) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	var n: int = w.count()
	out.resize(n)
	for i in n:
		out[i] = _decode(w.dtype, bytes, i)
	return out


func _weight_as_doubles(w, bytes: PackedByteArray) -> PackedFloat64Array:
	var out := PackedFloat64Array()
	var n: int = w.count()
	out.resize(n)
	for i in n:
		out[i] = _decode(w.dtype, bytes, i)
	return out


func _decode(dtype: int, bytes: PackedByteArray, i: int) -> float:
	match dtype:
		KpkModelScript.DType.F32: return bytes.decode_float(i * 4)
		KpkModelScript.DType.F16: return bytes.decode_half(i * 2)
		KpkModelScript.DType.I8: return float(bytes.decode_s8(i))
		KpkModelScript.DType.U8: return float(bytes.decode_u8(i))
		KpkModelScript.DType.I32: return float(bytes.decode_s32(i * 4))
		KpkModelScript.DType.I64: return float(bytes.decode_s64(i * 8))
		KpkModelScript.DType.BOOL: return float(bytes.decode_u8(i))
	return 0.0


func _compute_last_use() -> void:
	for i in model.nodes.size():
		for tid in model.nodes[i].inputs:
			if tid >= 0:
				_last_use[tid] = i


# ------------------------------------------------------------- buffer pooling

## Rounds an allocation up to a bucket, keeping four significant bits.
##
## The pool used to be keyed on the exact byte size, which meant a run of a
## different length shared nothing with the run before it — every new utterance
## allocated a complete fresh set of activations and memory grew without bound.
## Bucketing costs at most 6% slack and lets differently-sized runs reuse the
## same memory.
static func _bucket(nbytes: int) -> int:
	var size := maxi(256, (nbytes + 255) & ~255)
	if size <= 4096:
		return size
	var shift := 0
	var v := size
	while v > 15:
		v >>= 1
		shift += 1
	var gran := 1 << shift
	return (size + gran - 1) & ~(gran - 1)


func _alloc(nbytes: int) -> RID:
	var size := _bucket(nbytes)
	var free_list: Array = _pool.get(size, [])
	if not free_list.is_empty():
		var rid: RID = free_list.pop_back()
		_pool[size] = free_list
		_in_pool.erase(rid)
		_pooled_bytes -= size
		return rid
	var rid2: RID = gpu.create_buffer(size)
	_live[rid2] = size
	_live_bytes += size
	peak_live_bytes = maxi(peak_live_bytes, _live_bytes)
	return rid2


## A pooled buffer seeded with float data, replacing `gpu.create_buffer_f32`
## for anything allocated per run. Buffers made straight from the device were
## never entered into `_live`, so `_release` refused them and they leaked for
## the life of the engine — a few hundred per utterance.
func _alloc_f32(values: PackedFloat32Array) -> RID:
	var vals := values
	if vals.is_empty():
		vals = PackedFloat32Array([0.0])
	var rid := _alloc(vals.size() * 4)
	# A buffer cannot be updated while a compute list is open. Closing the list
	# is not a submission — the recorded dispatches stay queued and the next
	# `_record()` opens a fresh one — so this costs a barrier, not a GPU stall.
	if _recording:
		gpu.end_compute()
		_recording = false
	gpu.write_buffer_f32(rid, vals)
	return rid


func _release(rid: RID) -> void:
	if not rid.is_valid() or not _live.has(rid):
		return
	# Aliased views share one buffer under several tensor ids, so the same RID
	# can be released twice. Letting it land in the pool twice would hand the
	# same memory to two live tensors on a later run.
	if _in_pool.has(rid):
		return
	var size: int = _live[rid]
	var free_list: Array = _pool.get(size, [])
	free_list.append(rid)
	_pool[size] = free_list
	_in_pool[rid] = true
	_pooled_bytes += size


func _recycle(node_index: int, node) -> void:
	for tid in node.inputs:
		if tid < 0 or model.weights.has(tid):
			continue
		if _last_use.get(tid, -1) == node_index and _buf.has(tid):
			if trace_tids.has(tid):
				continue
			if want_durations and tid == duration_tid:
				continue
			_release(_buf[tid])
			_buf.erase(tid)


# --------------------------------------------------------------------- driver

## Runs the graph. Returns {"buffer": RID, "samples": int} or {} on failure.
func run(input_ids: PackedInt32Array, style: PackedFloat32Array,
		speed: float) -> Dictionary:
	last_error = ""
	traced.clear()
	_reset_activations()

	var ids_f := PackedFloat32Array()
	ids_f.resize(input_ids.size())
	for i in input_ids.size():
		ids_f[i] = float(input_ids[i])

	_set_input(model.input_ids_tid, PackedInt32Array([1, input_ids.size()]),
			ids_f)
	_set_input(model.style_tid, PackedInt32Array([1, 256]), style)
	_set_input(model.speed_tid, PackedInt32Array([1]),
			PackedFloat32Array([speed]))

	_dispatches = 0
	_recording = false

	var limit: int = model.nodes.size() if max_nodes < 0 \
			else mini(max_nodes, model.nodes.size())
	for i in limit:
		if cancel_token != null and i % CANCEL_INTERVAL == 0:
			if cancel_token.is_cancelled():
				var t := Time.get_ticks_msec()
				_flush()
				last_error = "cancelled at node %d of %d (%d ms draining)" \
						% [i, limit, Time.get_ticks_msec() - t]
				return {}
			if cancel_flush_nodes > 0 and i > 0 \
					and i % cancel_flush_nodes == 0:
				_flush()
		var node = model.nodes[i]
		var in_desc := ""
		if verbose:
			for tid in node.inputs:
				in_desc += ("-" if tid < 0 else str(_shape.get(tid, "?"))) + " "
		if not _run_node(i, node):
			_flush()
			last_error = last_error if last_error != "" else \
					"node %d (%s) failed" % [i, node.op_name]
			return {}
		if verbose:
			var oshape = _shape.get(node.outputs[0], PackedInt32Array()) \
					if node.outputs.size() > 0 else PackedInt32Array()
			print("[%4d] %-24s %s  in: %s -> %s" % [i, node.op_name,
					"cpu" if node.place == KpkModelScript.Place.CPU else "gpu",
					in_desc, oshape])
		_recycle(i, node)

	_flush()

	var out_tid: int = model.output_tid
	if not _buf.has(out_tid):
		last_error = "graph produced no waveform buffer"
		return {}

	# The graph only ever consumes the durations on the GPU, so this is a real
	# readback — a few dozen floats, after the work is already flushed.
	last_durations = PackedFloat32Array()
	if want_durations and duration_tid >= 0 and _ensure_cpu(duration_tid):
		var d: PackedFloat64Array = (_cpu[duration_tid] as CpuOps.CpuTensor).data
		last_durations.resize(d.size())
		for i in d.size():
			last_durations[i] = d[i]

	return {
		"buffer": _buf[out_tid],
		"samples": Ops.num_elements(_shape[out_tid]),
		"dispatches": _dispatches,
	}


## Clears everything produced by the previous run, keeping the weights and the
## scalar constants — those are owned outright rather than pooled, and the
## caches that hand out their tensor ids survive across runs.
func _reset_activations() -> void:
	for rid in _scratch:
		_release(rid)
	_scratch.clear()
	for tid in _buf.keys():
		if not model.weights.has(tid) and not _const_tids.has(tid):
			_release(_buf[tid])
			_buf.erase(tid)
	for tid in _cpu.keys():
		if not model.weights.has(tid) and not _const_tids.has(tid):
			_cpu.erase(tid)
	for tid in _shape.keys():
		if not model.weights.has(tid) and not _const_tids.has(tid):
			_shape.erase(tid)
	_alias_set.clear()
	# Weights keep their packed flag for the life of the executor; only the
	# runtime quantizations from the last run are stale.
	for tid in _packed.keys():
		if not model.weights.has(tid):
			_packed.erase(tid)


## Hands this run's activations back to the pool. Call it once the waveform has
## been read; `run()` does the same thing on entry, so it is only an
## optimization for the idle case, not a correctness requirement.
func release_activations() -> void:
	_flush()
	_reset_activations()


## Frees pooled buffers, largest first, until at most `budget_bytes` remain
## parked. Returns the number of bytes handed back to the driver.
##
## The pool exists so consecutive runs do not re-allocate, but a long utterance
## parks hundreds of megabytes that a short one will never touch again. Small
## buffers are kept: they are where the reuse actually pays, and they cost
## almost nothing.
func trim_memory(budget_bytes := 64 * 1024 * 1024) -> int:
	var sizes := _pool.keys()
	sizes.sort()
	sizes.reverse()
	var kept := 0
	var freed := 0
	for size in sizes:
		var list: Array = _pool[size]
		var keep_list := []
		for rid in list:
			if kept + size <= budget_bytes:
				kept += size
				keep_list.append(rid)
			else:
				_in_pool.erase(rid)
				_live.erase(rid)
				_live_bytes -= size
				_pooled_bytes -= size
				gpu.free_buffer(rid)
				freed += size
		if keep_list.is_empty():
			_pool.erase(size)
		else:
			_pool[size] = keep_list
	gpu.trim_uniform_sets()
	return freed


## Forgets the high-water mark, so the next stretch of work can be measured on
## its own.
func reset_peak() -> void:
	peak_live_bytes = _live_bytes


## Total bytes currently parked in the recycling pool.
func pooled_bytes() -> int:
	return _pooled_bytes


func _set_input(tid: int, shape: PackedInt32Array,
		data: PackedFloat32Array) -> void:
	_shape[tid] = shape
	_buf[tid] = _alloc_f32(data)
	var d := PackedFloat64Array()
	d.resize(data.size())
	for i in data.size():
		d[i] = data[i]
	_cpu[tid] = CpuOps.CpuTensor.new(shape, d)


func _record() -> void:
	if not _recording:
		gpu.begin_compute()
		_recording = true


func _flush() -> void:
	if _recording:
		gpu.end_compute()
		gpu.submit_and_wait()
		_recording = false


func _fail(msg: String) -> bool:
	last_error = msg
	push_error("KokoroExecutor: " + msg)
	return false


# ------------------------------------------------------------------ node exec

## Inputs whose *values* determine an output shape. If one of these is still
## GPU-only we must read it back before shapes can be inferred; that is what
## makes the predicted duration count a synchronization point.
const SHAPE_OPERANDS := {
	"Reshape": [1], "Expand": [1], "ConstantOfShape": [0],
	"Range": [0, 1, 2], "Slice": [1, 2, 3, 4], "Resize": [2, 3],
	"Pad": [1], "Tile": [1], "Split": [1], "Unsqueeze": [1], "Squeeze": [1],
	"ReduceMean": [1], "ReduceSum": [1], "CumSum": [1], "STFT": [1, 3],
	"ScatterND": [1],
}


func _run_node(index: int, node) -> bool:
	var op: String = node.op_name

	if SHAPE_OPERANDS.has(op):
		for k in SHAPE_OPERANDS[op]:
			if k < node.inputs.size():
				var t: int = node.inputs[k]
				if t >= 0 and not _cpu.has(t) and not _packed.has(t):
					_ensure_cpu(t)

	var in_shapes := []
	var consts := {}
	for k in node.inputs.size():
		var tid: int = node.inputs[k]
		if tid < 0:
			in_shapes.append(PackedInt32Array())
			continue
		if not _shape.has(tid):
			return _fail("input %s of %s is undefined"
					% [model.name_of(tid), op])
		in_shapes.append(_shape[tid])
		if _cpu.has(tid):
			consts[k] = (_cpu[tid] as CpuOps.CpuTensor).data

	if op == "NonZero":
		return _run_nonzero(node)

	var out_shapes := Shapes.infer(op, node.attrs, in_shapes, consts)
	for k in node.outputs.size():
		var tid: int = node.outputs[k]
		if tid >= 0 and k < out_shapes.size():
			_shape[tid] = out_shapes[k]

	if node.place == KpkModelScript.Place.CPU:
		return _run_cpu(node, out_shapes)
	return _run_gpu(node, in_shapes, out_shapes)


func _run_cpu(node, out_shapes: Array) -> bool:
	var op: String = node.op_name
	var ins := []
	for tid in node.inputs:
		if tid < 0:
			ins.append(null)
		elif _cpu.has(tid):
			ins.append(_cpu[tid])
		else:
			# Shape only needs the extents, which we always track.
			ins.append(CpuOps.CpuTensor.new(_shape.get(tid,
					PackedInt32Array()), PackedFloat64Array()))

	var outs := CpuOps.execute(op, node.attrs, ins, out_shapes)
	for k in node.outputs.size():
		var tid: int = node.outputs[k]
		if tid >= 0 and k < outs.size():
			_cpu[tid] = outs[k]
			_shape[tid] = (outs[k] as CpuOps.CpuTensor).shape
	return true


func _run_nonzero(node) -> bool:
	# Data-dependent shape: force everything recorded so far to complete and
	# pull the mask back to the CPU.
	var tid: int = node.inputs[0]
	if not _cpu.has(tid):
		_flush()
		if not _buf.has(tid):
			return _fail("NonZero input is neither on CPU nor GPU")
		var vals: PackedFloat32Array = gpu.read_f32(_buf[tid],
				Ops.num_elements(_shape[tid]))
		var d := PackedFloat64Array()
		d.resize(vals.size())
		for i in vals.size():
			d[i] = vals[i]
		_cpu[tid] = CpuOps.CpuTensor.new(_shape[tid], d)

	var outs := CpuOps.execute("NonZero", node.attrs, [_cpu[tid]], [])
	var out_tid: int = node.outputs[0]
	_cpu[out_tid] = outs[0]
	_shape[out_tid] = (outs[0] as CpuOps.CpuTensor).shape
	return true


## Ensures a tensor has a float32 GPU buffer, uploading from the CPU if needed.
func _gpu_of(tid: int) -> RID:
	if _buf.has(tid):
		return _buf[tid]
	if _cpu.has(tid):
		var t: CpuOps.CpuTensor = _cpu[tid]
		var f := PackedFloat32Array()
		f.resize(t.data.size())
		for i in t.data.size():
			f[i] = t.data[i]
		var rid: RID = _alloc_f32(f)
		_buf[tid] = rid
		return rid
	push_error("KokoroExecutor: tensor %s has no data" % model.name_of(tid))
	return RID()


## Debug accessors used by tests/test_taps.gd. Tensors listed in `trace_tids`
## are kept alive for the whole run so they can be read afterwards.
func is_packed(tid: int) -> bool:
	return _packed.has(tid)


func has_value(tid: int) -> bool:
	return _cpu.has(tid) or _buf.has(tid)


func shape_of(tid: int) -> PackedInt32Array:
	return _shape.get(tid, PackedInt32Array())


func value_of(tid: int) -> PackedFloat32Array:
	var n := maxi(1, Ops.num_elements(_shape.get(tid, PackedInt32Array())))
	if _cpu.has(tid):
		var d: PackedFloat64Array = (_cpu[tid] as CpuOps.CpuTensor).data
		var f := PackedFloat32Array()
		f.resize(d.size())
		for i in d.size():
			f[i] = d[i]
		return f
	if _buf.has(tid):
		_flush()
		return gpu.read_f32(_buf[tid], n)
	return PackedFloat32Array()


## Reads a packed uint8 tensor back as floats, one value per element. Only
## activation quantizations are packed at runtime and those are always uint8.
func value_of_packed(tid: int) -> PackedFloat32Array:
	var n := maxi(1, Ops.num_elements(_shape.get(tid, PackedInt32Array())))
	if not _buf.has(tid):
		return PackedFloat32Array()
	_flush()
	var bytes: PackedByteArray = gpu.read_bytes(_buf[tid])
	var out := PackedFloat32Array()
	out.resize(n)
	for i in n:
		out[i] = bytes.decode_u8(i)
	return out


## Pulls a tensor back to the CPU, flushing pending work first.
##
## This is a hard sync, so it is only used at the graph's two data-dependent
## points: the predicted durations and the iSTFT overlap-add index set.
func _ensure_cpu(tid: int) -> bool:
	if _cpu.has(tid):
		return true
	if not _buf.has(tid):
		return false
	_flush()
	var n := maxi(1, Ops.num_elements(_shape[tid]))
	var vals: PackedFloat32Array = gpu.read_f32(_buf[tid], n)
	var d := PackedFloat64Array()
	d.resize(vals.size())
	for i in vals.size():
		d[i] = vals[i]
	_cpu[tid] = CpuOps.CpuTensor.new(_shape[tid], d)
	return true


func _out_buf(tid: int) -> RID:
	var n := maxi(1, Ops.num_elements(_shape[tid]))
	var rid := _alloc(n * 4)
	_buf[tid] = rid
	return rid


func _scalar_buf(v: float) -> RID:
	return gpu.create_buffer_f32(PackedFloat32Array([v]))


# --------------------------------------------------------------- gpu dispatch

func _misc_push(mode: int, n: int, outer: int, dim: int, inner: int,
		out_dim: int, scale: float, pad_begin: int,
		update_count: int) -> PackedByteArray:
	var p := PackedByteArray()
	p.resize(48)
	p.encode_u32(0, mode)
	p.encode_u32(4, n)
	p.encode_u32(8, outer)
	p.encode_u32(12, dim)
	p.encode_u32(16, inner)
	p.encode_u32(20, out_dim)
	p.encode_float(24, scale)
	p.encode_s32(28, pad_begin)
	p.encode_u32(32, update_count)
	return p


## Records an elementwise dispatch; -1 for an unused operand.
func _ew(op: int, out_tid: int, a: int, b: int, c: int, alpha := 0.0) -> void:
	var oshape: PackedInt32Array = _shape[out_tid]
	var n := maxi(1, Ops.num_elements(oshape))
	var sa := Ops.broadcast_strides(_shape[a], oshape)
	var sb := Ops.broadcast_strides(
			_shape[b] if b >= 0 else PackedInt32Array(), oshape)
	var sc := Ops.broadcast_strides(
			_shape[c] if c >= 0 else PackedInt32Array(), oshape)
	var packed := Ops.collapse(oshape, [sa, sb, sc])

	var ra := _gpu_of(a)
	var rb := _gpu_of(b) if b >= 0 else ra
	var rc := _gpu_of(c) if c >= 0 else ra
	var ro := _out_buf(out_tid)

	var push: PackedByteArray = Ops.elementwise_push(op, n, packed.shape,
			packed.strides[0], packed.strides[1], packed.strides[2], alpha)
	_record()
	gpu.dispatch("elementwise", [ra, rb, rc, ro] as Array[RID], push,
			(n + 63) / 64)
	gpu.barrier()
	_dispatches += 1


## Records a strided copy from `src` into the buffer of `out_tid`.
func _copy(dst_rid: RID, src_rid: RID, region: PackedInt32Array,
		sstride: PackedInt32Array, src_base: int,
		dstride: PackedInt32Array, dst_base: int) -> void:
	var n := maxi(1, Ops.num_elements(region))
	var p := PackedByteArray()
	p.resize(64)
	p.encode_u32(0, n)
	p.encode_u32(4, region.size())
	p.encode_u32(8, src_base)
	p.encode_u32(12, dst_base)
	for i in 4:
		p.encode_u32(16 + i * 4, region[i] if i < region.size() else 1)
		p.encode_u32(32 + i * 4, sstride[i] if i < sstride.size() else 0)
		p.encode_u32(48 + i * 4, dstride[i] if i < dstride.size() else 0)
	_record()
	gpu.dispatch("copy", [src_rid, dst_rid] as Array[RID], p, (n + 63) / 64)
	gpu.barrier()
	_dispatches += 1


static func _row_strides(shape: PackedInt32Array) -> PackedInt32Array:
	var st := PackedInt32Array()
	st.resize(shape.size())
	var acc := 1
	for i in range(shape.size() - 1, -1, -1):
		st[i] = acc
		acc *= shape[i]
	return st


func _run_gpu(node, in_shapes: Array, out_shapes: Array) -> bool:
	var op: String = node.op_name
	var a: int = node.inputs[0] if node.inputs.size() > 0 else -1
	var b: int = node.inputs[1] if node.inputs.size() > 1 else -1
	var c: int = node.inputs[2] if node.inputs.size() > 2 else -1
	var out: int = node.outputs[0]

	if Ops.UNARY.has(op):
		_ew(Ops.UNARY[op], out, a, -1, -1)
		return true
	if Ops.BINARY.has(op):
		_ew(Ops.BINARY[op], out, a, b, -1)
		return true

	match op:
		"Cast":
			var to := int(node.attrs.get("to", 1))
			var mode := Ops.Op.IDENTITY
			if to in [3, 5, 6, 7, 12, 13]:
				mode = Ops.Op.CAST_INT
			elif to == 9:
				mode = Ops.Op.CAST_BOOL
			_ew(mode, out, a, -1, -1)
			return true
		"LeakyRelu":
			_ew(Ops.Op.LEAKYRELU, out, a, -1, -1,
					float(node.attrs.get("alpha", 0.01)))
			return true
		"Clip":
			var lo := b if b >= 0 else -1
			var hi := c if c >= 0 else -1
			if lo < 0 and hi < 0:
				_ew(Ops.Op.IDENTITY, out, a, -1, -1)
				return true
			var lo_t := lo if lo >= 0 else _const_tensor(-3.4e38)
			var hi_t := hi if hi >= 0 else _const_tensor(3.4e38)
			_ew(Ops.Op.CLIP, out, a, lo_t, hi_t)
			return true
		"Where":
			_ew(Ops.Op.WHERE, out, a, b, c)
			return true
		"FastGelu":
			_ew(Ops.Op.GELU, out, a, b if b >= 0 else _const_tensor(0.0), -1)
			return true
		"Softmax":
			return _run_softmax(node, out)
		"LayerNormalization", "SkipLayerNormalization":
			return _run_layernorm(node, out, op == "SkipLayerNormalization")
		"ReduceMean", "ReduceSum":
			return _run_reduce(node, in_shapes, out_shapes, out,
					op == "ReduceMean")
		"Reshape", "Squeeze", "Unsqueeze", "Identity":
			_alias(out, a)
			return true
		"Transpose":
			return _run_transpose(node, out, a)
		"Slice":
			return _run_slice(node, out, a)
		"Concat":
			return _run_concat(node, out)
		"Split":
			return _run_split(node, a)
		"Expand":
			var region: PackedInt32Array = _shape[out]
			_copy(_out_buf(out), _gpu_of(a), region,
					Ops.broadcast_strides(_shape[a], region), 0,
					_row_strides(region), 0)
			return true
		"ConstantOfShape":
			var region2: PackedInt32Array = _shape[out]
			var fill := 0.0
			var v = node.attrs.get("value")
			if v is Dictionary and v.has("v") and (v["v"] as Array).size() > 0:
				fill = float((v["v"] as Array)[0])
			var zeros := PackedInt32Array()
			zeros.resize(region2.size())
			_copy(_out_buf(out), _const_tensor_rid(fill), region2, zeros, 0,
					_row_strides(region2), 0)
			return true
		"Range":
			return _run_range(node, out)
		"Gather":
			return _run_gather(node, out, a, b)
		"MatMul", "FusedMatMul":
			return _run_matmul(node, out, a, b, op == "FusedMatMul")
		"MatMulInteger":
			return _run_matmul_int8(node, out)
		"DynamicQuantizeLinear":
			return _run_dynamic_quantize(node)
		"DequantizeLinear":
			# (x - zero_point) * scale, both scalars here
			var tmp := _temp_like(out)
			_ew(Ops.Op.SUB, tmp, a, c if c >= 0 else _const_tensor(0.0), -1)
			_ew(Ops.Op.MUL, out, tmp, b, -1)
			return true
		"Conv", "ConvInteger", "ConvTranspose":
			return _run_conv(node, out, op)
		"Resize":
			return _run_resize(node, out, a)
		"CumSum":
			return _run_cumsum(node, out, a)
		"Pad":
			return _run_pad(node, out, a)
		"STFT":
			return _run_stft(node, out)
		"ScatterND":
			return _run_scatternd(node, out)
		"DynamicQuantizeLSTM":
			return _run_lstm(node)

	return _fail("no GPU path for op '%s'" % op)


# ---------------------------------------------------------- per-op dispatchers

var _const_cache: Dictionary = {}
## Pseudo-tensor ids handed out by `_const_tensor`, exempt from the per-run
## reset because the cache that returns them outlives a single run.
var _const_tids: Dictionary = {}

func _const_tensor(v: float) -> int:
	## Registers a scalar as a pseudo-tensor so it can feed elementwise ops.
	if _const_cache.has(v):
		return _const_cache[v]
	var tid := -1000 - _const_cache.size()
	_shape[tid] = PackedInt32Array()
	_buf[tid] = _scalar_buf(v)
	_const_cache[v] = tid
	_const_tids[tid] = true
	return tid


func _const_tensor_rid(v: float) -> RID:
	return _gpu_of(_const_tensor(v))


var _temp_seq := 0

func _temp_like(tid: int) -> int:
	var t := -500000 - _temp_seq
	_temp_seq += 1
	_shape[t] = _shape[tid]
	return t


var _alias_set: Dictionary = {}

func _alias(dst: int, src: int) -> void:
	_buf[dst] = _gpu_of(src)
	_alias_set[dst] = true
	# Keep the owning buffer alive as long as any view of it is still read.
	_last_use[src] = maxi(_last_use.get(src, -1), _last_use.get(dst, -1))


func _run_softmax(node, out: int) -> bool:
	var shape: PackedInt32Array = _shape[out]
	var axis := Shapes.axis_of(int(node.attrs.get("axis", -1)), shape.size())
	if axis != shape.size() - 1:
		return _fail("Softmax over a non-final axis is not supported")
	var cols := shape[shape.size() - 1]
	var rows := maxi(1, Ops.num_elements(shape) / maxi(1, cols))
	var p := PackedByteArray()
	p.resize(16)
	p.encode_u32(0, rows)
	p.encode_u32(4, cols)
	_record()
	gpu.dispatch("softmax", [_gpu_of(node.inputs[0]), _out_buf(out)] as Array[RID],
			p, rows)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_layernorm(node, out: int, is_skip: bool) -> bool:
	var shape: PackedInt32Array = _shape[out]
	var cols := shape[shape.size() - 1]
	var rows := maxi(1, Ops.num_elements(shape) / maxi(1, cols))
	var x: int = node.inputs[0]
	var skip := -1
	var gamma := -1
	var beta := -1
	if is_skip:
		skip = node.inputs[1]
		gamma = node.inputs[2]
		beta = node.inputs[3] if node.inputs.size() > 3 else -1
	else:
		gamma = node.inputs[1]
		beta = node.inputs[2] if node.inputs.size() > 2 else -1

	var p := PackedByteArray()
	p.resize(32)
	p.encode_u32(0, rows)
	p.encode_u32(4, cols)
	p.encode_float(8, float(node.attrs.get("epsilon", 1e-5)))
	p.encode_u32(12, 1 if is_skip else 0)
	p.encode_u32(16, 1 if beta >= 0 else 0)

	var rx := _gpu_of(x)
	var rs := _gpu_of(skip) if skip >= 0 else rx
	var rg := _gpu_of(gamma)
	var rb := _gpu_of(beta) if beta >= 0 else rg
	_record()
	gpu.dispatch("layernorm", [rx, rs, rg, rb, _out_buf(out)] as Array[RID],
			p, rows)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_reduce(node, in_shapes: Array, out_shapes: Array, out: int,
		is_mean: bool) -> bool:
	var src: PackedInt32Array = in_shapes[0]
	var axes := PackedInt32Array()
	if _cpu.has(node.inputs[1] if node.inputs.size() > 1 else -1):
		for v in (_cpu[node.inputs[1]] as CpuOps.CpuTensor).data:
			axes.append(Shapes.axis_of(int(v), src.size()))
	elif node.attrs.has("axes"):
		for v in node.attrs["axes"]:
			axes.append(Shapes.axis_of(int(v), src.size()))
	if axes.size() != 1:
		return _fail("only single-axis reductions are supported (got %s)" % axes)

	var ax := axes[0]
	var outer := 1
	for i in ax:
		outer *= src[i]
	var inner := 1
	for i in range(ax + 1, src.size()):
		inner *= src[i]

	var p := PackedByteArray()
	p.resize(16)
	p.encode_u32(0, outer * inner)
	p.encode_u32(4, src[ax])
	p.encode_u32(8, inner)
	p.encode_u32(12, 1 if is_mean else 0)
	_record()
	gpu.dispatch("reduce", [_gpu_of(node.inputs[0]), _out_buf(out)] as Array[RID],
			p, outer * inner)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_transpose(node, out: int, a: int) -> bool:
	var src: PackedInt32Array = _shape[a]
	var region: PackedInt32Array = _shape[out]
	var perm := PackedInt32Array()
	if node.attrs.has("perm"):
		for v in node.attrs["perm"]:
			perm.append(int(v))
	else:
		for i in range(src.size() - 1, -1, -1):
			perm.append(i)
	var src_st := _row_strides(src)
	var ss := PackedInt32Array()
	ss.resize(region.size())
	for d in region.size():
		ss[d] = src_st[perm[d]]
	_copy(_out_buf(out), _gpu_of(a), region, ss, 0, _row_strides(region), 0)
	return true


func _run_slice(node, out: int, a: int) -> bool:
	var src: PackedInt32Array = _shape[a]
	var rank := src.size()
	var starts := _cpu_ints(node, 1)
	var axes := _cpu_ints(node, 3)
	var steps := _cpu_ints(node, 4)
	if axes.is_empty():
		axes.resize(starts.size())
		for i in starts.size():
			axes[i] = i
	if steps.is_empty():
		steps.resize(starts.size())
		steps.fill(1)

	var src_st := _row_strides(src)
	var ss := src_st.duplicate()
	var base := 0
	for i in starts.size():
		var ax := Shapes.axis_of(axes[i], rank)
		var s := starts[i]
		if s < 0:
			s += src[ax]
		s = clampi(s, 0, maxi(0, src[ax] - 1))
		base += s * src_st[ax]
		ss[ax] = src_st[ax] * steps[i]
	_copy(_out_buf(out), _gpu_of(a), _shape[out], ss, base,
			_row_strides(_shape[out]), 0)
	return true


func _run_concat(node, out: int) -> bool:
	var oshape: PackedInt32Array = _shape[out]
	var ax := Shapes.axis_of(int(node.attrs.get("axis", 0)), oshape.size())
	var dst_st := _row_strides(oshape)
	var dst_rid := _out_buf(out)
	var offset := 0
	for tid in node.inputs:
		if tid < 0:
			continue
		var s: PackedInt32Array = _shape[tid]
		if ax >= s.size():
			return _fail("Concat axis %d is outside input %s of shape %s "
					% [ax, model.name_of(tid), s]
					+ "(output %s)" % [oshape])
		_copy(dst_rid, _gpu_of(tid), s, _row_strides(s), 0, dst_st,
				offset * dst_st[ax])
		offset += s[ax]
	return true


func _run_split(node, a: int) -> bool:
	var src: PackedInt32Array = _shape[a]
	var ax := Shapes.axis_of(int(node.attrs.get("axis", 0)), src.size())
	var src_st := _row_strides(src)
	var offset := 0
	for tid in node.outputs:
		if tid < 0:
			continue
		var s: PackedInt32Array = _shape[tid]
		_copy(_out_buf(tid), _gpu_of(a), s, src_st, offset * src_st[ax],
				_row_strides(s), 0)
		offset += s[ax]
	return true


func _run_range(node, out: int) -> bool:
	var n := Ops.num_elements(_shape[out])
	var start := _cpu_scalar(node, 0)
	var delta := _cpu_scalar(node, 2)
	var vals := PackedFloat32Array()
	vals.resize(maxi(1, n))
	for i in n:
		vals[i] = start + delta * i
	_buf[out] = _alloc_f32(vals)
	return true


func _run_gather(node, out: int, a: int, b: int) -> bool:
	var data: PackedInt32Array = _shape[a]
	var ax := Shapes.axis_of(int(node.attrs.get("axis", 0)), data.size())
	var outer := 1
	for i in ax:
		outer *= data[i]
	var inner := 1
	for i in range(ax + 1, data.size()):
		inner *= data[i]
	var n_idx := maxi(1, Ops.num_elements(_shape[b]))

	var mode := 0
	if _packed.has(a):
		var w = model.weights.get(a)
		mode = 1 if (w != null and w.dtype == KpkModelScript.DType.I8) else 2

	var p := PackedByteArray()
	p.resize(32)
	p.encode_u32(0, maxi(1, Ops.num_elements(_shape[out])))
	p.encode_u32(4, outer)
	p.encode_u32(8, n_idx)
	p.encode_u32(12, inner)
	p.encode_u32(16, data[ax])
	p.encode_u32(20, mode)
	_record()
	gpu.dispatch("gather", [_gpu_of(a), _gpu_of(b), _out_buf(out)] as Array[RID],
			p, (Ops.num_elements(_shape[out]) + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	return true


func _matmul_dims(ashape: PackedInt32Array, bshape: PackedInt32Array,
		trans_a: bool, trans_b: bool) -> Dictionary:
	var av := ashape.duplicate()
	var bv := bshape.duplicate()
	if av.size() == 1:
		av.insert(0, 1)
	if bv.size() == 1:
		bv.append(1)
	var m := av[av.size() - 2]
	var k := av[av.size() - 1]
	if trans_a:
		var t := m
		m = k
		k = t
	var kb := bv[bv.size() - 2]
	var n := bv[bv.size() - 1]
	if trans_b:
		var t2 := kb
		kb = n
		n = t2
	var batch_a := 1
	for i in av.size() - 2:
		batch_a *= av[i]
	var batch_b := 1
	for i in bv.size() - 2:
		batch_b *= bv[i]
	var batch := maxi(batch_a, batch_b)
	return {
		"M": m, "N": n, "K": k, "batch": batch,
		"a_stride": (m * k) if batch_a > 1 else 0,
		"b_stride": (kb * n) if batch_b > 1 else 0,
	}


func _run_matmul(node, out: int, a: int, b: int, fused: bool) -> bool:
	var ta := fused and int(node.attrs.get("transA", 0)) != 0
	var tb := fused and int(node.attrs.get("transB", 0)) != 0
	var d := _matmul_dims(_shape[a], _shape[b], ta, tb)
	var p := PackedByteArray()
	p.resize(48)
	p.encode_u32(0, d.M)
	p.encode_u32(4, d.N)
	p.encode_u32(8, d.K)
	p.encode_u32(12, d.batch)
	p.encode_u32(16, d.a_stride)
	p.encode_u32(20, d.b_stride)
	p.encode_u32(24, 1 if ta else 0)
	p.encode_u32(28, 1 if tb else 0)
	p.encode_float(32, float(node.attrs.get("alpha", 1.0)))
	_record()
	gpu.dispatch("matmul", [_gpu_of(a), _gpu_of(b), _out_buf(out)] as Array[RID],
			p, (d.N + 15) / 16, (d.M + 15) / 16, d.batch)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_matmul_int8(node, out: int) -> bool:
	var a: int = node.inputs[0]
	var b: int = node.inputs[1]
	var azp: int = node.inputs[2] if node.inputs.size() > 2 else -1
	var bzp: int = node.inputs[3] if node.inputs.size() > 3 else -1
	var d := _matmul_dims(_shape[a], _shape[b], false, false)
	var per_col := bzp >= 0 and Ops.num_elements(_shape[bzp]) > 1

	var p := PackedByteArray()
	p.resize(32)
	p.encode_u32(0, d.M)
	p.encode_u32(4, d.N)
	p.encode_u32(8, d.K)
	p.encode_u32(12, d.batch)
	p.encode_u32(16, d.a_stride)
	p.encode_u32(20, d.b_stride)
	p.encode_u32(24, 1 if per_col else 0)
	p.encode_u32(28, 1 if _is_signed8(b) else 0)

	var zero := _const_tensor_rid(0.0)
	_record()
	gpu.dispatch("matmul_int8", [
		_gpu_of(a), _gpu_of(b),
		_zp_rid(azp, zero),
		_zp_rid(bzp, zero),
		_out_buf(out),
	] as Array[RID], p, (d.N + 15) / 16, (d.M + 15) / 16, d.batch)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_dynamic_quantize(node) -> bool:
	var a: int = node.inputs[0]
	var n := maxi(1, Ops.num_elements(_shape[a]))
	var q_tid: int = node.outputs[0]
	var s_tid: int = node.outputs[1] if node.outputs.size() > 1 else -1
	var z_tid: int = node.outputs[2] if node.outputs.size() > 2 else -1

	# The min/max pair is scratch for the two dispatches below and never lands
	# in `_buf`, so nothing else will ever release it. One 256-byte bucket per
	# node across 139 nodes is 33 KB an utterance — invisible over a test run
	# and a third of a gigabyte over a long session.
	var mm := _alloc(8)
	_scratch.append(mm)
	var words := (n + 3) / 4
	var q_rid := _alloc(words * 4)
	_buf[q_tid] = q_rid
	_packed[q_tid] = true
	var s_rid := _alloc(4)
	var z_rid := _alloc(4)
	if s_tid >= 0:
		_buf[s_tid] = s_rid
		_shape[s_tid] = PackedInt32Array()
	else:
		_scratch.append(s_rid)
	if z_tid >= 0:
		_buf[z_tid] = z_rid
		_shape[z_tid] = PackedInt32Array()
	else:
		_scratch.append(z_rid)

	var src := _gpu_of(a)
	var p1 := PackedByteArray()
	p1.resize(16)
	p1.encode_u32(0, n)
	_record()
	gpu.dispatch("minmax", [src, mm] as Array[RID], p1, 1)
	gpu.barrier()

	var p2 := PackedByteArray()
	p2.resize(16)
	p2.encode_u32(0, n)
	p2.encode_u32(4, words)
	gpu.dispatch("quantize", [src, mm, q_rid, s_rid, z_rid] as Array[RID], p2,
			(words + 63) / 64)
	gpu.barrier()
	_dispatches += 2
	return true


func _run_conv(node, out: int, op: String) -> bool:
	var x: int = node.inputs[0]
	var w: int = node.inputs[1]
	var xs: PackedInt32Array = _shape[x]
	var ws: PackedInt32Array = _shape[w]
	var os: PackedInt32Array = _shape[out]

	var pads := _attr_ints(node, "pads", [0, 0])
	var strides := _attr_ints(node, "strides", [1])
	var dils := _attr_ints(node, "dilations", [1])
	var group := int(node.attrs.get("group", 1))
	var transposed := op == "ConvTranspose"
	var cout := os[1]
	var cin_g := ws[1] if not transposed else (xs[1] / maxi(1, group))
	var cout_g := cout / maxi(1, group)
	if transposed:
		cin_g = xs[1] / maxi(1, group)
		cout_g = ws[1]

	var n_out := maxi(1, Ops.num_elements(os))

	if op == "ConvInteger":
		var xzp: int = node.inputs[2] if node.inputs.size() > 2 else -1
		var wzp: int = node.inputs[3] if node.inputs.size() > 3 else -1
		var p := PackedByteArray()
		p.resize(48)
		p.encode_u32(0, n_out)
		p.encode_u32(4, cin_g)
		p.encode_u32(8, cout)
		p.encode_u32(12, xs[2])
		p.encode_u32(16, os[2])
		p.encode_u32(20, ws[2])
		p.encode_u32(24, strides[0])
		p.encode_u32(28, dils[0])
		p.encode_s32(32, -pads[0])
		p.encode_u32(36, group)
		p.encode_u32(40, cout_g)
		p.encode_u32(44, 1 if _is_signed8(w) else 0)
		var zero := _const_tensor_rid(0.0)
		_record()
		gpu.dispatch("conv_int8", [
			_gpu_of(x), _gpu_of(w),
			_zp_rid(xzp, zero),
			_zp_rid(wzp, zero),
			_out_buf(out),
		] as Array[RID], p, (n_out + 63) / 64)
		gpu.barrier()
		_dispatches += 1
		return true

	var bias: int = node.inputs[2] if node.inputs.size() > 2 else -1
	var p2 := PackedByteArray()
	p2.resize(64)
	p2.encode_u32(0, n_out)
	p2.encode_u32(4, cin_g)
	p2.encode_u32(8, cout)
	p2.encode_u32(12, cout_g)
	p2.encode_u32(16, xs[2])
	p2.encode_u32(20, os[2])
	p2.encode_u32(24, ws[2])
	p2.encode_u32(28, strides[0])
	p2.encode_u32(32, dils[0])
	p2.encode_s32(36, -pads[0])
	p2.encode_u32(40, group)
	p2.encode_u32(44, 1 if transposed else 0)
	p2.encode_u32(48, 1 if bias >= 0 else 0)
	var rx := _gpu_of(x)
	_record()
	gpu.dispatch("conv", [rx, _gpu_of(w),
			_gpu_of(bias) if bias >= 0 else rx,
			_out_buf(out)] as Array[RID], p2, (n_out + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	return true


func _axis_view(shape: PackedInt32Array, axis: int) -> Dictionary:
	var outer := 1
	for i in axis:
		outer *= shape[i]
	var inner := 1
	for i in range(axis + 1, shape.size()):
		inner *= shape[i]
	return {"outer": outer, "inner": inner, "dim": shape[axis]}


func _run_resize(node, out: int, a: int) -> bool:
	var src: PackedInt32Array = _shape[a]
	var os: PackedInt32Array = _shape[out]
	var axis := src.size() - 1
	var v := _axis_view(src, axis)
	var mode_s := str(node.attrs.get("mode", "nearest"))
	var mode := M_RESIZE_NEAREST if mode_s == "nearest" else M_RESIZE_LINEAR
	var scale := float(os[axis]) / maxf(1.0, float(src[axis]))
	var n := maxi(1, Ops.num_elements(os))
	var p := _misc_push(mode, n, v.outer, v.dim, v.inner, os[axis], scale, 0, 0)
	var ra := _gpu_of(a)
	_record()
	gpu.dispatch("misc", [ra, ra, ra, _out_buf(out)] as Array[RID], p,
			(n + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_cumsum(node, out: int, a: int) -> bool:
	var src: PackedInt32Array = _shape[a]
	var axis := 0
	if _cpu.has(node.inputs[1]):
		axis = Shapes.axis_of(int((_cpu[node.inputs[1]] as CpuOps.CpuTensor)
				.data[0]), src.size())
	var v := _axis_view(src, axis)
	var n := maxi(1, Ops.num_elements(src))
	var p := _misc_push(M_CUMSUM, n, v.outer, v.dim, v.inner, v.dim, 1.0, 0, 0)
	var ra := _gpu_of(a)
	_record()
	gpu.dispatch("misc", [ra, ra, ra, _out_buf(out)] as Array[RID], p,
			(n + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_pad(node, out: int, a: int) -> bool:
	var src: PackedInt32Array = _shape[a]
	var os: PackedInt32Array = _shape[out]
	var pads := _cpu_ints(node, 1)
	var rank := src.size()
	var axis := -1
	for i in rank:
		if pads[i] != 0 or pads[i + rank] != 0:
			if axis != -1:
				return _fail("Pad on more than one axis is not supported")
			axis = i
	if axis == -1:
		_alias(out, a)
		return true
	if str(node.attrs.get("mode", "constant")) != "reflect":
		return _fail("only reflect Pad is implemented")
	var v := _axis_view(src, axis)
	var n := maxi(1, Ops.num_elements(os))
	var p := _misc_push(M_PAD_REFLECT, n, v.outer, v.dim, v.inner, os[axis],
			1.0, pads[axis], 0)
	var ra := _gpu_of(a)
	_record()
	gpu.dispatch("misc", [ra, ra, ra, _out_buf(out)] as Array[RID], p,
			(n + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_stft(node, out: int) -> bool:
	var sig: PackedInt32Array = _shape[node.inputs[0]]
	var os: PackedInt32Array = _shape[out]
	var hop := int(_cpu_scalar(node, 1))
	var n_fft := int(_cpu_scalar(node, 3))
	var p := PackedByteArray()
	p.resize(32)
	p.encode_u32(0, os[0])
	p.encode_u32(4, os[1])
	p.encode_u32(8, os[2])
	p.encode_u32(12, n_fft)
	p.encode_u32(16, hop)
	p.encode_u32(20, sig[1])
	var total := os[0] * os[1] * os[2]
	_record()
	gpu.dispatch("stft", [_gpu_of(node.inputs[0]), _gpu_of(node.inputs[2]),
			_out_buf(out)] as Array[RID], p, (total + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_scatternd(node, out: int) -> bool:
	# Copy the base tensor, then overwrite the indexed positions.
	var data: int = node.inputs[0]
	var idx: int = node.inputs[1]
	var upd: int = node.inputs[2]
	var shape: PackedInt32Array = _shape[out]
	var dst := _out_buf(out)
	_copy(dst, _gpu_of(data), shape, _row_strides(shape), 0,
			_row_strides(shape), 0)

	if not _ensure_cpu(idx):
		return _fail("ScatterND indices are unavailable")
	var it: CpuOps.CpuTensor = _cpu[idx]
	var k := it.shape[it.shape.size() - 1]
	var count := it.data.size() / maxi(1, k)
	var strides := _row_strides(shape)
	var flat := PackedFloat32Array()
	flat.resize(maxi(1, count))
	for i in count:
		var off := 0
		for d in k:
			off += int(it.data[i * k + d]) * strides[d]
		flat[i] = off
	# Scratch: no tensor id owns it, so it is returned to the pool when the run
	# ends rather than at a last-use boundary.
	var idx_rid: RID = _alloc_f32(flat)
	_scratch.append(idx_rid)
	var p := _misc_push(M_SCATTER_ND, count, 0, 1, 1, 1, 1.0, 0, count)
	var ru := _gpu_of(upd)
	_record()
	gpu.dispatch("misc", [ru, idx_rid, ru, dst] as Array[RID], p,
			(count + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	return true


func _run_lstm(node) -> bool:
	var x: int = node.inputs[0]
	var w: int = node.inputs[1]
	var r: int = node.inputs[2]
	var bias: int = node.inputs[3] if node.inputs.size() > 3 else -1
	var hidden := int(node.attrs.get("hidden_size", 256))
	var ndir := 2 if str(node.attrs.get("direction", "forward")) \
			== "bidirectional" else 1
	var xs: PackedInt32Array = _shape[x]
	var seq := xs[0]
	var batch := xs[1]
	var input_size := xs[2]
	var h4 := hidden * 4
	if batch != 1:
		return _fail("LSTM batch %d is not supported" % batch)

	var y_tid: int = node.outputs[0]
	var y_rid := _out_buf(y_tid)
	# Y_h and Y_c are consumed downstream, so they must be materialized even
	# though the sequence output carries most of the signal.
	var yh_tid: int = node.outputs[1] if node.outputs.size() > 1 else -1
	var yc_tid: int = node.outputs[2] if node.outputs.size() > 2 else -1
	# When the graph does not consume them the kernel still writes them, so they
	# have to exist — as scratch, not as a tensor nothing will ever free.
	var yh_rid := _out_buf(yh_tid) if yh_tid >= 0 else _alloc(ndir * hidden * 4)
	var yc_rid := _out_buf(yc_tid) if yc_tid >= 0 else _alloc(ndir * hidden * 4)
	if yh_tid < 0:
		_scratch.append(yh_rid)
	if yc_tid < 0:
		_scratch.append(yc_rid)

	for d in ndir:
		var wf := _dequant_lstm(w, d, input_size * h4,
				node.inputs[8], node.inputs[9], d)
		var rf := _dequant_lstm(r, d, hidden * h4,
				node.inputs[10], node.inputs[11], d)
		if not wf.is_valid() or not rf.is_valid():
			return _fail("LSTM weight dequantization failed")

		# XW = X · W(d), then fold in Wb + Rb.
		var xw := _alloc(seq * h4 * 4)
		var p := PackedByteArray()
		p.resize(48)
		p.encode_u32(0, seq)
		p.encode_u32(4, h4)
		p.encode_u32(8, input_size)
		p.encode_u32(12, 1)
		p.encode_float(32, 1.0)
		_record()
		gpu.dispatch("matmul", [_gpu_of(x), wf, xw] as Array[RID], p,
				(h4 + 15) / 16, (seq + 15) / 16, 1)
		gpu.barrier()
		_dispatches += 1

		if bias >= 0:
			var bsum := _lstm_bias(bias, d, hidden)
			var xw2 := _alloc(seq * h4 * 4)
			var shape := PackedInt32Array([seq, h4])
			var push2: PackedByteArray = Ops.elementwise_push(
					Ops.Op.ADD, seq * h4, shape,
					PackedInt32Array([h4, 1]), PackedInt32Array([0, 1]),
					PackedInt32Array([0, 0]))
			gpu.dispatch("elementwise", [xw, bsum, bsum, xw2] as Array[RID],
					push2, (seq * h4 + 63) / 64)
			gpu.barrier()
			_dispatches += 1
			_release(xw)
			xw = xw2

		var lp := PackedByteArray()
		lp.resize(32)
		lp.encode_u32(0, seq)
		lp.encode_u32(4, hidden)
		lp.encode_u32(8, 1 if d == 1 else 0)
		lp.encode_u32(12, 0)
		lp.encode_u32(16, ndir * hidden)
		lp.encode_u32(20, d * hidden)
		lp.encode_u32(24, d * hidden)
		gpu.dispatch("lstm", [xw, rf, xw, xw, y_rid, yh_rid, yc_rid]
				as Array[RID], lp, 1)
		gpu.barrier()
		_dispatches += 1
		_release(xw)

	# Y_h / Y_c are unused downstream in this graph.
	return true


var _deq_cache: Dictionary = {}

func _dequant_lstm(tid: int, dir_index: int, count: int, scale_tid: int,
		zp_tid: int, d: int) -> RID:
	var key := "%d:%d" % [tid, dir_index]
	if _deq_cache.has(key):
		return _deq_cache[key]
	var scale := 1.0
	var zp := 0
	if scale_tid >= 0 and _cpu.has(scale_tid):
		var sd: PackedFloat64Array = (_cpu[scale_tid] as CpuOps.CpuTensor).data
		scale = sd[d] if sd.size() > d else sd[0]
	if zp_tid >= 0 and _cpu.has(zp_tid):
		var zd: PackedFloat64Array = (_cpu[zp_tid] as CpuOps.CpuTensor).data
		zp = int(zd[d] if zd.size() > d else zd[0])

	var out: RID = gpu.create_buffer(count * 4)
	var p := _misc_push(M_DEQUANT_I8, count, dir_index * count, 1, 1, 1,
			scale, zp, 0)
	_record()
	gpu.dispatch("misc", [out, out, _gpu_of(tid), out] as Array[RID], p,
			(count + 63) / 64)
	gpu.barrier()
	_dispatches += 1
	_deq_cache[key] = out
	return out


var _bias_cache: Dictionary = {}

func _lstm_bias(tid: int, d: int, hidden: int) -> RID:
	var key := "%d:%d" % [tid, d]
	if _bias_cache.has(key):
		return _bias_cache[key]
	var h4 := hidden * 4
	var t: CpuOps.CpuTensor = _cpu.get(tid)
	var vals := PackedFloat32Array()
	vals.resize(h4)
	if t != null:
		for i in h4:
			var base := d * h4 * 2
			vals[i] = t.data[base + i] + t.data[base + h4 + i]
	var rid: RID = gpu.create_buffer_f32(vals)
	_bias_cache[key] = rid
	return rid


# ---------------------------------------------------------------- small utils

## True when a packed 8-bit tensor holds signed values.
func _is_signed8(tid: int) -> bool:
	var w = model.weights.get(tid)
	return w != null and w.dtype == KpkModelScript.DType.I8


var _zp_cache: Dictionary = {}

## Zero points arrive from two very different places: activation zero points
## are runtime scalars that DynamicQuantizeLinear already writes as float32,
## but *weight* zero points are int8/uint8 initializers that were uploaded as
## packed bytes. The int8 kernels read their zp bindings as float, so a packed
## buffer must be converted first — a raw byte like 114 reinterpreted as
## float32 is a denormal that rounds to zero, which silently erases the
## correction. The tiny CPU mirror already holds the decoded values.
func _zp_rid(tid: int, zero: RID) -> RID:
	if tid < 0:
		return zero
	if not _packed.has(tid):
		return _gpu_of(tid)
	if _zp_cache.has(tid):
		return _zp_cache[tid]
	var t: CpuOps.CpuTensor = _cpu.get(tid)
	var vals := PackedFloat32Array()
	if t != null:
		vals.resize(t.data.size())
		for i in t.data.size():
			vals[i] = t.data[i]
	if vals.is_empty():
		vals.append(0.0)
	var rid: RID = gpu.create_buffer_f32(vals)
	_zp_cache[tid] = rid
	return rid


func _cpu_ints(node, idx: int) -> PackedInt32Array:
	var out := PackedInt32Array()
	if idx >= node.inputs.size():
		return out
	var tid: int = node.inputs[idx]
	if tid < 0 or not _cpu.has(tid):
		return out
	for v in (_cpu[tid] as CpuOps.CpuTensor).data:
		out.append(Shapes.safe_int(v))
	return out


func _cpu_scalar(node, idx: int) -> float:
	var tid: int = node.inputs[idx] if idx < node.inputs.size() else -1
	if tid < 0 or not _cpu.has(tid):
		return 0.0
	var d: PackedFloat64Array = (_cpu[tid] as CpuOps.CpuTensor).data
	return d[0] if d.size() > 0 else 0.0


func _attr_ints(node, key: String, fallback: Array) -> PackedInt32Array:
	var out := PackedInt32Array()
	var src = node.attrs.get(key, fallback)
	for v in src:
		out.append(int(v))
	if out.is_empty():
		for v in fallback:
			out.append(int(v))
	return out
