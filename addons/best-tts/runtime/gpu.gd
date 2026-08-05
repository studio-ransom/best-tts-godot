@tool
class_name KokoroGPU
extends RefCounted

## Thin wrapper over a local RenderingDevice for the TTS compute pipeline.
##
## Owns the device, the compiled kernels, and the storage buffers. Everything
## here is renderer-agnostic apart from one hard requirement: a RenderingDevice
## exists only on Forward+ and Mobile, never on Compatibility/GL.

const KERNEL_DIR := "res://addons/best-tts/kernels/"

var rd: RenderingDevice

## kernel name -> RID of the compute pipeline
var _pipelines: Dictionary = {}
## kernel name -> RID of the shader (kept so uniform sets can be created)
var _shaders: Dictionary = {}
## buffer RIDs we allocated, so cleanup is exhaustive. A dictionary rather than
## an array because buffers are freed one at a time and `Array.erase` is linear.
var _buffers: Dictionary = {}
## Sum of `_buffers`, maintained rather than computed — see `total_buffer_bytes`.
var _total_bytes := 0
## cache of uniform sets keyed by "kernel|rid0,rid1,..." — creating these is
## expensive and the executor rebinds the same combinations every call
var _uniform_sets: Dictionary = {}
## buffer id -> Array of cache keys that bind it, so freeing a buffer can drop
## the sets that referenced it
var _set_users: Dictionary = {}

var _compute_list: int = -1


## Creates the device. Returns an error string, or "" on success.
func initialize() -> String:
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		return ("Kokoro TTS needs a RenderingDevice for compute shaders. "
				+ "Set Project Settings > Rendering > Renderer to 'Forward+' "
				+ "or 'Mobile'; the Compatibility backend cannot run it.")
	return ""


func is_ready() -> bool:
	return rd != null


## Compiles a kernel from `kernels/<name>.glsl`, caching the pipeline.
func load_kernel(name: String) -> bool:
	if _pipelines.has(name):
		return true
	var path := KERNEL_DIR + name + ".glsl"
	if not ResourceLoader.exists(path):
		push_error("KokoroGPU: missing kernel %s" % path)
		return false
	var file: RDShaderFile = load(path)
	if file == null:
		push_error("KokoroGPU: %s did not import as an RDShaderFile" % path)
		return false
	var spirv := file.get_spirv()
	if spirv == null:
		push_error("KokoroGPU: %s produced no SPIR-V" % path)
		return false
	var err := spirv.compile_error_compute
	if err != "":
		push_error("KokoroGPU: %s failed to compile:\n%s" % [path, err])
		return false
	var shader := rd.shader_create_from_spirv(spirv)
	if not shader.is_valid():
		push_error("KokoroGPU: could not create shader for %s" % path)
		return false
	var pipeline := rd.compute_pipeline_create(shader)
	if not pipeline.is_valid():
		push_error("KokoroGPU: could not create pipeline for %s" % path)
		return false
	_shaders[name] = shader
	_pipelines[name] = pipeline
	return true


func has_kernel(name: String) -> bool:
	return _pipelines.has(name)


# --- buffers ---------------------------------------------------------------

## Allocates a storage buffer, optionally seeded with `data`.
func create_buffer(size_bytes: int, data := PackedByteArray()) -> RID:
	# Vulkan rejects zero-sized buffers, and std430 wants 4-byte alignment.
	var size := maxi(4, (size_bytes + 3) & ~3)
	var rid: RID
	if data.is_empty():
		rid = rd.storage_buffer_create(size)
	else:
		var padded := data
		if padded.size() < size:
			padded = padded.duplicate()
			padded.resize(size)
		rid = rd.storage_buffer_create(size, padded)
	if rid.is_valid():
		_buffers[rid] = size
		_total_bytes += size
	return rid


## Every byte this device has allocated, weights included — the number the
## driver is actually holding.
##
## A running total rather than a walk over `_buffers`. This is read from
## whatever thread wants a memory reading while the engine thread is still
## allocating and freeing, and iterating a Dictionary that another thread is
## erasing from throws on whichever key vanishes mid-loop. An int can only ever
## be a few kilobytes stale, which for a diagnostic is no cost at all.
func total_buffer_bytes() -> int:
	return _total_bytes


func create_buffer_f32(values: PackedFloat32Array) -> RID:
	return create_buffer(values.size() * 4, values.to_byte_array())


## Overwrites the start of an existing buffer. This is what lets the executor
## recycle a pooled buffer for a new upload instead of allocating another one.
func write_buffer(buffer: RID, data: PackedByteArray) -> void:
	if data.is_empty():
		return
	# buffer_update wants a 4-byte multiple, and every buffer we hand out is
	# already padded to one.
	var n := (data.size() + 3) & ~3
	var padded := data
	if padded.size() != n:
		padded = padded.duplicate()
		padded.resize(n)
	rd.buffer_update(buffer, 0, n, padded)


func write_buffer_f32(buffer: RID, values: PackedFloat32Array) -> void:
	write_buffer(buffer, values.to_byte_array())


func create_buffer_i32(values: PackedInt32Array) -> RID:
	return create_buffer(values.size() * 4, values.to_byte_array())


func read_f32(buffer: RID, count := -1) -> PackedFloat32Array:
	var bytes := rd.buffer_get_data(buffer)
	var out := bytes.to_float32_array()
	if count >= 0 and out.size() > count:
		out = out.slice(0, count)
	return out


func read_bytes(buffer: RID) -> PackedByteArray:
	return rd.buffer_get_data(buffer)


func read_i32(buffer: RID, count := -1) -> PackedInt32Array:
	var bytes := rd.buffer_get_data(buffer)
	var out := bytes.to_int32_array()
	if count >= 0 and out.size() > count:
		out = out.slice(0, count)
	return out


func free_buffer(buffer: RID) -> void:
	if not buffer.is_valid():
		return
	_total_bytes -= int(_buffers.get(buffer, 0))
	_buffers.erase(buffer)
	# RenderingDevice frees a uniform set as soon as one of its buffers goes
	# away, so the cached RID is dead the moment we free this. Drop the entries
	# without freeing them again — handing out a stale set is a GPU crash, and
	# leaving them in place is a slow leak.
	var id := buffer.get_id()
	for key in _set_users.get(id, []):
		_uniform_sets.erase(key)
	_set_users.erase(id)
	rd.free_rid(buffer)


# --- dispatch --------------------------------------------------------------

## Builds (and caches) a uniform set binding `buffers` to bindings 0..n-1.
func uniform_set(kernel: String, buffers: Array[RID]) -> RID:
	var key := kernel
	for b in buffers:
		key += "|" + str(b.get_id())
	if _uniform_sets.has(key):
		return _uniform_sets[key]

	var uniforms: Array[RDUniform] = []
	for i in buffers.size():
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = i
		u.add_id(buffers[i])
		uniforms.append(u)
	var set := rd.uniform_set_create(uniforms, _shaders[kernel], 0)
	_uniform_sets[key] = set
	for b in buffers:
		var id := b.get_id()
		var users: Array = _set_users.get(id, [])
		users.append(key)
		_set_users[id] = users
	return set


## Drops the uniform-set cache once it grows past `cap`.
##
## Without a bound this climbed by a couple of thousand entries per utterance
## and never came down. Rebuilding the sets costs a little on the next run, so
## the cap is loose; it is a backstop, not a policy. Must not run while a
## compute list is being recorded.
func trim_uniform_sets(cap := 8192) -> int:
	if _compute_list != -1 or _uniform_sets.size() <= cap:
		return 0
	var n := _uniform_sets.size()
	for key in _uniform_sets:
		var set: RID = _uniform_sets[key]
		if set.is_valid():
			rd.free_rid(set)
	_uniform_sets.clear()
	_set_users.clear()
	return n


func begin_compute() -> void:
	_compute_list = rd.compute_list_begin()


## Records one dispatch. `groups` is the workgroup count, already divided by
## the kernel's local size. Must sit between begin_compute/end_compute.
func dispatch(kernel: String, buffers: Array[RID], push: PackedByteArray,
		groups_x: int, groups_y := 1, groups_z := 1) -> void:
	if _compute_list == -1:
		push_error("KokoroGPU.dispatch called outside begin_compute()")
		return
	rd.compute_list_bind_compute_pipeline(_compute_list, _pipelines[kernel])
	rd.compute_list_bind_uniform_set(_compute_list, uniform_set(kernel, buffers), 0)
	if not push.is_empty():
		# Push constants must be a multiple of 16 bytes.
		var pc := push
		var need := (pc.size() + 15) & ~15
		if pc.size() != need:
			pc = pc.duplicate()
			pc.resize(need)
		rd.compute_list_set_push_constant(_compute_list, pc, pc.size())
	rd.compute_list_dispatch(_compute_list, maxi(1, groups_x),
			maxi(1, groups_y), maxi(1, groups_z))


## Inserts a barrier so the next dispatch sees the previous one's writes.
func barrier() -> void:
	if _compute_list != -1:
		rd.compute_list_add_barrier(_compute_list)


func end_compute() -> void:
	if _compute_list == -1:
		return
	rd.compute_list_end()
	_compute_list = -1


## Submits recorded work and blocks until the GPU is done.
func submit_and_wait() -> void:
	rd.submit()
	rd.sync()


func cleanup() -> void:
	if rd == null:
		return
	_uniform_sets.clear()
	_set_users.clear()
	for b in _buffers:
		if b.is_valid():
			rd.free_rid(b)
	_buffers.clear()
	_total_bytes = 0
	for name in _pipelines:
		rd.free_rid(_pipelines[name])
	for name in _shaders:
		rd.free_rid(_shaders[name])
	_pipelines.clear()
	_shaders.clear()
	rd.free()
	rd = null
