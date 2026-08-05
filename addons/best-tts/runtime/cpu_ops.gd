@tool
class_name KokoroCpuOps
extends RefCounted

## The 409 shape/index nodes of the graph, executed in GDScript.
##
## These never touch bulk tensor data — they compute shapes, axes, slice
## bounds and alignment indices, all tiny int64 arrays. Values are carried as
## PackedFloat64Array, which represents every integer this graph produces
## exactly (all well under 2^53) while keeping one uniform container.

const Shapes := preload("res://addons/best-tts/runtime/shapes.gd")


class CpuTensor extends RefCounted:
	var shape: PackedInt32Array
	var data: PackedFloat64Array

	func _init(p_shape := PackedInt32Array(),
			p_data := PackedFloat64Array()) -> void:
		shape = p_shape
		data = p_data

	static func scalar(v: float) -> CpuTensor:
		return CpuTensor.new(PackedInt32Array(), PackedFloat64Array([v]))

	func count() -> int:
		var n := 1
		for d in shape:
			n *= d
		return n

	func _to_string() -> String:
		var head := ""
		for i in mini(6, data.size()):
			head += ("%d" % data[i]) if data[i] == floor(data[i]) \
					else ("%.4f" % data[i])
			head += " "
		return "%s[%s]" % [shape, head.strip_edges()]


## Row-major strides for `shape`.
static func strides_of(shape: PackedInt32Array) -> PackedInt32Array:
	var st := PackedInt32Array()
	st.resize(shape.size())
	var acc := 1
	for i in range(shape.size() - 1, -1, -1):
		st[i] = acc
		acc *= shape[i]
	return st


## Unravels a flat index into per-axis coordinates.
static func unravel(idx: int, shape: PackedInt32Array) -> PackedInt32Array:
	var c := PackedInt32Array()
	c.resize(shape.size())
	var rem := idx
	for i in range(shape.size() - 1, -1, -1):
		c[i] = rem % shape[i]
		rem /= shape[i]
	return c


## Executes one CPU node. `ins` is an Array[CpuTensor] (null for omitted
## inputs). Returns Array[CpuTensor], one per output.
static func execute(op: String, attrs: Dictionary, ins: Array,
		out_shapes: Array) -> Array:
	match op:
		"Shape":
			# The value is the *input's* shape, supplied by the executor as a
			# zero-data tensor carrying that shape.
			var src: CpuTensor = ins[0]
			var start := int(attrs.get("start", 0))
			var end := int(attrs.get("end", src.shape.size()))
			var vals := PackedFloat64Array()
			for i in range(start, mini(end, src.shape.size())):
				vals.append(src.shape[i])
			return [CpuTensor.new(PackedInt32Array([vals.size()]), vals)]

		"Cast":
			var src: CpuTensor = ins[0]
			var to := int(attrs.get("to", 1))
			var vals := src.data.duplicate()
			# integer and bool targets truncate / normalize
			if to in [2, 3, 5, 6, 7, 12, 13]:
				# GDScript has no trunc(); round toward zero explicitly.
				for i in vals.size():
					vals[i] = floor(vals[i]) if vals[i] >= 0.0 \
							else ceil(vals[i])
			elif to == 9:
				for i in vals.size():
					vals[i] = 1.0 if vals[i] != 0.0 else 0.0
			return [CpuTensor.new(src.shape, vals)]

		"Concat":
			return [_concat(ins, int(attrs.get("axis", 0)), out_shapes[0])]

		"Unsqueeze", "Squeeze", "Reshape", "Identity":
			return [CpuTensor.new(out_shapes[0], (ins[0] as CpuTensor).data)]

		"Gather":
			return [_gather(ins[0], ins[1], int(attrs.get("axis", 0)),
					out_shapes[0])]

		"Slice":
			return [_slice(ins, out_shapes[0])]

		"Transpose":
			return [_transpose(ins[0], attrs.get("perm"), out_shapes[0])]

		"Range":
			var start: float = (ins[0] as CpuTensor).data[0]
			var delta: float = (ins[2] as CpuTensor).data[0]
			var n: int = Shapes._num(out_shapes[0])
			var vals := PackedFloat64Array()
			vals.resize(n)
			for i in n:
				vals[i] = start + delta * i
			return [CpuTensor.new(out_shapes[0], vals)]

		"NonZero":
			return [_nonzero(ins[0])]

		"Equal", "Greater", "Less", "GreaterOrEqual", "And", "Or", \
		"Add", "Sub", "Mul", "Div", "Where", "Min", "Max":
			return [_broadcast_op(op, ins, out_shapes[0])]

		"ConstantOfShape":
			var n: int = Shapes._num(out_shapes[0])
			var fill := 0.0
			var v = attrs.get("value")
			if v is Dictionary and v.has("v") and (v["v"] as Array).size() > 0:
				fill = float((v["v"] as Array)[0])
			var vals := PackedFloat64Array()
			vals.resize(n)
			vals.fill(fill)
			return [CpuTensor.new(out_shapes[0], vals)]

	push_error("KokoroCpuOps: no CPU implementation for '%s'" % op)
	return [CpuTensor.new(out_shapes[0])]


# -------------------------------------------------------------------- helpers

static func _concat(ins: Array, axis: int,
		out_shape: PackedInt32Array) -> CpuTensor:
	var rank := out_shape.size()
	var ax := Shapes.axis_of(axis, rank)
	var out := PackedFloat64Array()
	out.resize(Shapes._num(out_shape))

	# outer = product of dims before the axis, inner = after it
	var outer := 1
	for i in ax:
		outer *= out_shape[i]
	var inner := 1
	for i in range(ax + 1, rank):
		inner *= out_shape[i]

	var written := 0
	for o in outer:
		for t in ins:
			var src: CpuTensor = t
			if src == null:
				continue
			var seg: int = src.shape[ax] if ax < src.shape.size() else 1
			var chunk := seg * inner
			for j in chunk:
				out[o * out_shape[ax] * inner + written + j] = \
						src.data[o * chunk + j]
			written += chunk
		written = 0
	return CpuTensor.new(out_shape, out)


static func _gather(data: CpuTensor, indices: CpuTensor, axis: int,
		out_shape: PackedInt32Array) -> CpuTensor:
	var rank := data.shape.size()
	var ax := Shapes.axis_of(axis, rank)
	var dim := data.shape[ax] if ax < rank else 1

	var outer := 1
	for i in ax:
		outer *= data.shape[i]
	var inner := 1
	for i in range(ax + 1, rank):
		inner *= data.shape[i]

	var n_idx := indices.data.size()
	var out := PackedFloat64Array()
	out.resize(outer * n_idx * inner)
	for o in outer:
		for k in n_idx:
			var src_i := int(indices.data[k])
			if src_i < 0:
				src_i += dim
			src_i = clampi(src_i, 0, maxi(0, dim - 1))
			for j in inner:
				out[(o * n_idx + k) * inner + j] = \
						data.data[(o * dim + src_i) * inner + j]
	return CpuTensor.new(out_shape, out)


static func _slice(ins: Array, out_shape: PackedInt32Array) -> CpuTensor:
	var src: CpuTensor = ins[0]
	var rank := src.shape.size()
	var starts := _ints(ins, 1)
	var ends := _ints(ins, 2)
	var axes := _ints(ins, 3)
	var steps := _ints(ins, 4)
	if axes.is_empty():
		axes.resize(starts.size())
		for i in starts.size():
			axes[i] = i
	if steps.is_empty():
		steps.resize(starts.size())
		steps.fill(1)

	var begin := PackedInt32Array()
	var step := PackedInt32Array()
	begin.resize(rank)
	step.resize(rank)
	step.fill(1)
	for i in rank:
		begin[i] = 0
	for i in starts.size():
		var ax := Shapes.axis_of(axes[i], rank)
		var dim := src.shape[ax]
		var s := starts[i]
		if s < 0:
			s += dim
		var st := steps[i]
		begin[ax] = clampi(s, 0, maxi(0, dim - 1)) if st > 0 \
				else clampi(s, 0, maxi(0, dim - 1))
		step[ax] = st

	var src_st := strides_of(src.shape)
	var n := Shapes._num(out_shape)
	var out := PackedFloat64Array()
	out.resize(n)
	for i in n:
		var c := unravel(i, out_shape)
		var off := 0
		for d in rank:
			off += (begin[d] + c[d] * step[d]) * src_st[d]
		out[i] = src.data[off]
	return CpuTensor.new(out_shape, out)


static func _transpose(src: CpuTensor, perm,
		out_shape: PackedInt32Array) -> CpuTensor:
	var rank := src.shape.size()
	var order := PackedInt32Array()
	if perm == null:
		order.resize(rank)
		for i in rank:
			order[i] = rank - 1 - i
	else:
		for v in perm:
			order.append(int(v))

	var src_st := strides_of(src.shape)
	var n := src.data.size()
	var out := PackedFloat64Array()
	out.resize(n)
	for i in n:
		var c := unravel(i, out_shape)
		var off := 0
		for d in rank:
			off += c[d] * src_st[order[d]]
		out[i] = src.data[off]
	return CpuTensor.new(out_shape, out)


static func _nonzero(src: CpuTensor) -> CpuTensor:
	var rank := maxi(1, src.shape.size())
	var hits: Array[int] = []
	for i in src.data.size():
		if src.data[i] != 0.0:
			hits.append(i)
	var out := PackedFloat64Array()
	out.resize(rank * hits.size())
	# ONNX returns [rank, n]: one row of coordinates per axis.
	for k in hits.size():
		var c := unravel(hits[k], src.shape)
		for d in rank:
			out[d * hits.size() + k] = c[d] if d < c.size() else 0
	return CpuTensor.new(PackedInt32Array([rank, hits.size()]), out)


static func _broadcast_op(op: String, ins: Array,
		out_shape: PackedInt32Array) -> CpuTensor:
	var n := Shapes._num(out_shape)
	var out := PackedFloat64Array()
	out.resize(n)
	var a: CpuTensor = ins[0]
	var b: CpuTensor = ins[1] if ins.size() > 1 else null
	var c: CpuTensor = ins[2] if ins.size() > 2 else null

	var sa := _bstrides(a, out_shape)
	var sb := _bstrides(b, out_shape)
	var sc := _bstrides(c, out_shape)

	for i in n:
		var coord := unravel(i, out_shape)
		var x := _at(a, coord, sa)
		var y := _at(b, coord, sb)
		match op:
			"Add": out[i] = x + y
			"Sub": out[i] = x - y
			"Mul": out[i] = x * y
			"Div": out[i] = y if false else (x / y if y != 0.0 else 0.0)
			"Equal": out[i] = 1.0 if x == y else 0.0
			"Greater": out[i] = 1.0 if x > y else 0.0
			"Less": out[i] = 1.0 if x < y else 0.0
			"GreaterOrEqual": out[i] = 1.0 if x >= y else 0.0
			"And": out[i] = 1.0 if (x != 0.0 and y != 0.0) else 0.0
			"Or": out[i] = 1.0 if (x != 0.0 or y != 0.0) else 0.0
			"Min": out[i] = minf(x, y)
			"Max": out[i] = maxf(x, y)
			"Where": out[i] = y if x != 0.0 else _at(c, coord, sc)
	return CpuTensor.new(out_shape, out)


static func _bstrides(t: CpuTensor, out_shape: PackedInt32Array) -> PackedInt32Array:
	if t == null:
		return PackedInt32Array()
	const Ops := preload("res://addons/best-tts/runtime/ops.gd")
	return Ops.broadcast_strides(t.shape, out_shape)


static func _at(t: CpuTensor, coord: PackedInt32Array,
		strides: PackedInt32Array) -> float:
	if t == null or t.data.is_empty():
		return 0.0
	var off := 0
	for d in mini(coord.size(), strides.size()):
		off += coord[d] * strides[d]
	return t.data[off] if off < t.data.size() else 0.0


static func _ints(ins: Array, idx: int) -> PackedInt32Array:
	var out := PackedInt32Array()
	if idx >= ins.size() or ins[idx] == null:
		return out
	for v in (ins[idx] as CpuTensor).data:
		out.append(Shapes.safe_int(v))
	return out
