extends SceneTree

## Exercises the whole GPU path: local RenderingDevice, kernel import, buffer
## upload, broadcast index math, dispatch and readback.
##
##   godot --headless --script res://tests/test_gpu_elementwise.gd
##
## Expected values are computed here in plain GDScript, independently of the
## stride machinery under test, so an indexing bug cannot hide.

const GPUScript := preload("res://addons/best-tts/runtime/gpu.gd")
const Ops := preload("res://addons/best-tts/runtime/ops.gd")

var gpu
var failures := 0


func _init() -> void:
	gpu = GPUScript.new()
	var err: String = gpu.initialize()
	if err != "":
		printerr("FAIL: " + err)
		quit(1)
		return
	if not gpu.load_kernel("elementwise"):
		printerr("FAIL: elementwise kernel did not compile")
		quit(1)
		return
	print("device + elementwise kernel ready")

	_test_binary("Mul add-broadcast", Ops.Op.MUL,
			[2, 3, 4], [3, 1])
	_test_binary("Add scalar rhs", Ops.Op.ADD, [5, 7], [])
	_test_binary("Sub equal shapes", Ops.Op.SUB, [4, 6], [4, 6])
	_test_binary("Div outer broadcast", Ops.Op.DIV, [2, 1, 5], [2, 4, 5])
	_test_unary("Sqrt", Ops.Op.SQRT, [3, 9])
	_test_unary("Sigmoid", Ops.Op.SIGMOID, [64])
	_test_unary("Floor", Ops.Op.FLOOR, [17])
	_test_leaky()
	_test_where()

	gpu.cleanup()
	if failures == 0:
		print("\nOK: all elementwise GPU checks passed")
	else:
		printerr("\n%d check(s) FAILED" % failures)
	quit(1 if failures > 0 else 0)


func _fill(shape: Array, seed: int) -> PackedFloat32Array:
	var n := 1
	for d in shape:
		n *= d
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var out := PackedFloat32Array()
	out.resize(maxi(n, 1))
	for i in out.size():
		out[i] = rng.randf_range(0.5, 3.5)
	return out


## Straightforward reference broadcast, written without the stride helpers.
func _expected(op: int, ashape: Array, a: PackedFloat32Array,
		bshape: Array, b: PackedFloat32Array,
		oshape: Array) -> PackedFloat32Array:
	var rank := oshape.size()
	var n := 1
	for d in oshape:
		n *= d
	var out := PackedFloat32Array()
	out.resize(n)
	for i in n:
		# unravel i into coordinates
		var coord := []
		coord.resize(rank)
		var rem := i
		for d in range(rank - 1, -1, -1):
			coord[d] = rem % oshape[d]
			rem /= oshape[d]
		out[i] = _apply(op, _pick(ashape, a, coord), _pick(bshape, b, coord))
	return out


func _pick(shape: Array, data: PackedFloat32Array, coord: Array) -> float:
	if shape.is_empty():
		return data[0]
	var pad: int = coord.size() - shape.size()
	var idx := 0
	for d in shape.size():
		var c: int = coord[pad + d]
		if shape[d] == 1:
			c = 0
		idx = idx * shape[d] + c
	return data[idx]


func _apply(op: int, x: float, y: float) -> float:
	match op:
		Ops.Op.ADD: return x + y
		Ops.Op.SUB: return x - y
		Ops.Op.MUL: return x * y
		Ops.Op.DIV: return x / y
		Ops.Op.SQRT: return sqrt(x)
		Ops.Op.SIGMOID: return 1.0 / (1.0 + exp(-x))
		Ops.Op.FLOOR: return floor(x)
	return x


func _run(op: int, ashape: Array, a: PackedFloat32Array,
		bshape: Array, b: PackedFloat32Array,
		cshape: Array, c: PackedFloat32Array,
		alpha := 0.0) -> PackedFloat32Array:
	var oshape := Ops.broadcast_shape(PackedInt32Array(ashape),
			PackedInt32Array(bshape))
	oshape = Ops.broadcast_shape(oshape, PackedInt32Array(cshape))
	var n := Ops.num_elements(oshape)

	var sa := Ops.broadcast_strides(PackedInt32Array(ashape), oshape)
	var sb := Ops.broadcast_strides(PackedInt32Array(bshape), oshape)
	var sc := Ops.broadcast_strides(PackedInt32Array(cshape), oshape)
	var packed := Ops.collapse(oshape, [sa, sb, sc])

	var buf_a: RID = gpu.create_buffer_f32(a)
	var buf_b: RID = gpu.create_buffer_f32(b)
	var buf_c: RID = gpu.create_buffer_f32(c)
	var buf_o: RID = gpu.create_buffer(n * 4)

	var push: PackedByteArray = Ops.elementwise_push(op, n, packed.shape,
			packed.strides[0], packed.strides[1], packed.strides[2], alpha)

	gpu.begin_compute()
	gpu.dispatch("elementwise", [buf_a, buf_b, buf_c, buf_o] as Array[RID],
			push, (n + 63) / 64)
	gpu.end_compute()
	gpu.submit_and_wait()

	var out: PackedFloat32Array = gpu.read_f32(buf_o, n)
	gpu.free_buffer(buf_a)
	gpu.free_buffer(buf_b)
	gpu.free_buffer(buf_c)
	gpu.free_buffer(buf_o)
	return out


func _test_binary(label: String, op: int, ashape: Array, bshape: Array) -> void:
	var a := _fill(ashape, 1)
	var b := _fill(bshape, 2)
	var oshape := Ops.broadcast_shape(PackedInt32Array(ashape),
			PackedInt32Array(bshape))
	var want := _expected(op, ashape, a, bshape, b, Array(oshape))
	var got := _run(op, ashape, a, bshape, b, [], PackedFloat32Array([0.0]))
	_check(label, want, got)


func _test_unary(label: String, op: int, shape: Array) -> void:
	var a := _fill(shape, 3)
	var want := _expected(op, shape, a, [], PackedFloat32Array([0.0]),
			Array(PackedInt32Array(shape)))
	var got := _run(op, shape, a, [], PackedFloat32Array([0.0]),
			[], PackedFloat32Array([0.0]))
	_check(label, want, got)


func _test_leaky() -> void:
	var a := PackedFloat32Array([-2.0, -0.5, 0.0, 0.5, 2.0])
	var want := PackedFloat32Array()
	for v in a:
		want.append(v * 0.2 if v < 0.0 else v)
	var got := _run(Ops.Op.LEAKYRELU, [5], a, [], PackedFloat32Array([0.0]),
			[], PackedFloat32Array([0.0]), 0.2)
	_check("LeakyRelu alpha=0.2", want, got)


func _test_where() -> void:
	var cond := PackedFloat32Array([1.0, 0.0, 1.0, 0.0])
	var x := PackedFloat32Array([10.0, 20.0, 30.0, 40.0])
	var y := PackedFloat32Array([-1.0])
	var want := PackedFloat32Array([10.0, -1.0, 30.0, -1.0])
	var got := _run(Ops.Op.WHERE, [4], cond, [4], x, [], y)
	_check("Where with scalar else-branch", want, got)


func _check(label: String, want: PackedFloat32Array,
		got: PackedFloat32Array) -> void:
	if got.size() != want.size():
		printerr("FAIL %s: size %d, expected %d" % [label, got.size(), want.size()])
		failures += 1
		return
	var worst := 0.0
	var at := -1
	for i in want.size():
		var d: float = absf(got[i] - want[i])
		if d > worst:
			worst = d
			at = i
	# GDScript's format has no %e, so scientific notation goes through
	# String.num_scientific.
	if worst > 1e-5:
		printerr("FAIL %s: max err %s at %d (got %f want %f)"
				% [label, String.num_scientific(worst), at, got[at], want[at]])
		failures += 1
	else:
		print("  ok  %s (n=%d) max err %s"
				% [label, want.size(), String.num_scientific(worst)])
