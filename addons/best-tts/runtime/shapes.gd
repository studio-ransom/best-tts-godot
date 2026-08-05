@tool
class_name KokoroShapes
extends RefCounted

## Output-shape rules for every op in the Kokoro graph.
##
## The executor needs each node's output shape before it can size a GPU buffer,
## and the graph has dynamic axes (phoneme count, then predicted frame count),
## so shapes are recomputed on every run. Semantics follow the ONNX spec as
## implemented in tools/ref_runtime.py, which is verified against onnxruntime.

const Ops := preload("res://addons/best-tts/runtime/ops.gd")


static func _num(shape: PackedInt32Array) -> int:
	return Ops.num_elements(shape)


## Normalizes a possibly-negative axis against `rank`.
static func axis_of(a: int, rank: int) -> int:
	return a if a >= 0 else a + rank


## Broadcast of several shapes.
static func broadcast_all(shapes: Array) -> PackedInt32Array:
	var out: PackedInt32Array = shapes[0]
	for i in range(1, shapes.size()):
		out = Ops.broadcast_shape(out, shapes[i])
	return out


## Resolves the output shape of `node`.
##
## `in_shapes` holds one shape per input (empty array for omitted inputs) and
## `const_inputs` maps input index -> PackedFloat64Array for inputs whose values
## are already known on the CPU (shape operands, axes, split sizes, ...).
## Returns an Array of shapes, one per output.
static func infer(op: String, attrs: Dictionary, in_shapes: Array,
		const_inputs: Dictionary) -> Array:
	match op:
		# --- elementwise, shape follows broadcasting -----------------------
		"Add", "Sub", "Mul", "Div", "Pow", "Equal", "Greater", "Less", \
		"GreaterOrEqual", "And", "Or", "Min", "Max", "Mod":
			return [broadcast_all([in_shapes[0], in_shapes[1]])]
		"Where", "Clip":
			var parts := []
			for s in in_shapes:
				if not (s as PackedInt32Array).is_empty() or in_shapes.size() == 1:
					parts.append(s)
			if parts.is_empty():
				parts = [in_shapes[0]]
			return [broadcast_all(parts)]
		"Sqrt", "Exp", "Sin", "Cos", "Atan", "Tanh", "Sigmoid", "Floor", \
		"Round", "LeakyRelu", "Cast", "Identity", "Neg", "Abs", "Log", \
		"Erf", "Relu", "Softmax", "Not":
			return [in_shapes[0]]
		"FastGelu", "SkipLayerNormalization", "LayerNormalization":
			return [in_shapes[0]]

		# --- shape manipulation --------------------------------------------
		"Shape":
			var rank: int = (in_shapes[0] as PackedInt32Array).size()
			var start: int = int(attrs.get("start", 0))
			var end: int = int(attrs.get("end", rank))
			return [PackedInt32Array([maxi(0, end - start)])]
		"Reshape":
			return [_reshape(in_shapes[0], const_inputs.get(1),
					int(attrs.get("allowzero", 0)))]
		"Transpose":
			return [_transpose(in_shapes[0], attrs.get("perm"))]
		"Unsqueeze":
			return [_unsqueeze(in_shapes[0], _axes(attrs, const_inputs, 1))]
		"Squeeze":
			return [_squeeze(in_shapes[0], _axes(attrs, const_inputs, 1))]
		"Concat":
			return [_concat(in_shapes, int(attrs.get("axis", 0)))]
		"Split":
			return _split(in_shapes[0], int(attrs.get("axis", 0)),
					const_inputs.get(1), attrs.get("num_outputs", 0))
		"Slice":
			return [_slice(in_shapes[0], const_inputs)]
		"Gather":
			return [_gather(in_shapes[0], in_shapes[1],
					int(attrs.get("axis", 0)))]
		"Expand":
			return [Ops.broadcast_shape(in_shapes[0],
					_as_shape(const_inputs.get(1)))]
		"ConstantOfShape":
			return [_as_shape(const_inputs.get(0))]
		"Range":
			return [PackedInt32Array([_range_len(const_inputs)])]
		"Pad":
			return [_pad(in_shapes[0], const_inputs.get(1))]
		"Tile":
			return [_tile(in_shapes[0], const_inputs.get(1))]

		# --- reductions ------------------------------------------------------
		"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd":
			return [_reduce(in_shapes[0], _axes(attrs, const_inputs, 1),
					bool(int(attrs.get("keepdims", 1))),
					bool(int(attrs.get("noop_with_empty_axes", 0))))]
		"CumSum":
			return [in_shapes[0]]

		# --- matmul ----------------------------------------------------------
		"MatMul", "MatMulInteger":
			return [_matmul(in_shapes[0], in_shapes[1], false, false)]
		"FusedMatMul":
			return [_matmul(in_shapes[0], in_shapes[1],
					bool(int(attrs.get("transA", 0))),
					bool(int(attrs.get("transB", 0))))]

		# --- quantization ------------------------------------------------------
		"DynamicQuantizeLinear":
			return [in_shapes[0], PackedInt32Array(), PackedInt32Array()]
		"DequantizeLinear", "QuantizeLinear":
			return [in_shapes[0]]

		# --- convolutions -------------------------------------------------------
		"Conv", "ConvInteger":
			return [_conv(in_shapes[0], in_shapes[1], attrs, false)]
		"ConvTranspose":
			return [_conv(in_shapes[0], in_shapes[1], attrs, true)]

		# --- misc ---------------------------------------------------------------
		"Resize":
			return [_resize(in_shapes[0], const_inputs)]
		"STFT":
			return [_stft(in_shapes[0], attrs, const_inputs)]
		"ScatterND":
			return [in_shapes[0]]
		"NonZero":
			push_error("KokoroShapes: NonZero output is data-dependent")
			return [PackedInt32Array()]
		"DynamicQuantizeLSTM", "LSTM":
			return _lstm(in_shapes[0], attrs)

	push_error("KokoroShapes: no rule for op '%s'" % op)
	return [PackedInt32Array()]


# -------------------------------------------------------------------- helpers

## Saturating conversion to int32.
##
## Slice bounds routinely carry INT64_MAX as an "to the end" sentinel; taken
## literally it overflows and the axis silently collapses to length zero.
static func safe_int(v: float) -> int:
	return int(clampf(v, -2147483000.0, 2147483000.0))


static func _as_shape(vals) -> PackedInt32Array:
	var out := PackedInt32Array()
	if vals == null:
		return out
	for v in vals:
		out.append(safe_int(v))
	return out


static func _axes(attrs: Dictionary, const_inputs: Dictionary,
		idx: int) -> Variant:
	if const_inputs.has(idx):
		return _as_shape(const_inputs[idx])
	if attrs.has("axes"):
		return _as_shape(attrs["axes"])
	return null


static func _reshape(src: PackedInt32Array, target,
		allowzero: int) -> PackedInt32Array:
	var want := _as_shape(target)
	var out := want.duplicate()
	var known := 1
	var wild := -1
	for i in out.size():
		if out[i] == 0 and allowzero == 0:
			out[i] = src[i] if i < src.size() else 1
		if out[i] == -1:
			wild = i
		else:
			known *= out[i]
	if wild >= 0:
		out[wild] = _num(src) / maxi(1, known)
	return out


static func _transpose(src: PackedInt32Array, perm) -> PackedInt32Array:
	var rank := src.size()
	var order := _as_shape(perm)
	if order.is_empty():
		order.resize(rank)
		for i in rank:
			order[i] = rank - 1 - i        # default is full reversal
	var out := PackedInt32Array()
	out.resize(rank)
	for i in rank:
		out[i] = src[order[i]]
	return out


static func _unsqueeze(src: PackedInt32Array, axes) -> PackedInt32Array:
	var list := _as_shape(axes)
	var rank := src.size() + list.size()
	var norm: Array[int] = []
	for a in list:
		norm.append(axis_of(a, rank))
	norm.sort()
	var out := src.duplicate()
	for a in norm:
		out.insert(a, 1)
	return out


static func _squeeze(src: PackedInt32Array, axes) -> PackedInt32Array:
	var out := PackedInt32Array()
	if axes == null:
		for d in src:
			if d != 1:
				out.append(d)
		return out
	var drop := {}
	for a in _as_shape(axes):
		drop[axis_of(a, src.size())] = true
	for i in src.size():
		if not drop.has(i):
			out.append(src[i])
	return out


static func _concat(shapes: Array, axis: int) -> PackedInt32Array:
	var first: PackedInt32Array = shapes[0]
	var ax := axis_of(axis, first.size())
	var out := first.duplicate()
	out[ax] = 0
	for s in shapes:
		var sh: PackedInt32Array = s
		if sh.is_empty():
			continue
		out[ax] += sh[ax]
	return out


static func _split(src: PackedInt32Array, axis: int, sizes,
		num_outputs: int) -> Array:
	var ax := axis_of(axis, src.size())
	var parts := _as_shape(sizes)
	if parts.is_empty():
		var n := maxi(1, num_outputs)
		var each := src[ax] / n
		parts.resize(n)
		for i in n:
			parts[i] = each
	var out := []
	for p in parts:
		var s := src.duplicate()
		s[ax] = p
		out.append(s)
	return out


static func _slice(src: PackedInt32Array, ci: Dictionary) -> PackedInt32Array:
	var starts := _as_shape(ci.get(1))
	var ends := _as_shape(ci.get(2))
	var axes := _as_shape(ci.get(3))
	var steps := _as_shape(ci.get(4))
	if axes.is_empty():
		axes.resize(starts.size())
		for i in starts.size():
			axes[i] = i
	if steps.is_empty():
		steps.resize(starts.size())
		steps.fill(1)

	var out := src.duplicate()
	for i in starts.size():
		var ax := axis_of(axes[i], src.size())
		var dim := src[ax]
		var st := steps[i]
		var s := starts[i]
		var e := ends[i]
		if s < 0:
			s += dim
		if e < 0:
			e += dim
		if st > 0:
			s = clampi(s, 0, dim)
			e = clampi(e, 0, dim)
			out[ax] = maxi(0, (e - s + st - 1) / st)
		else:
			s = clampi(s, -1, dim - 1)
			e = clampi(e, -1, dim - 1)
			out[ax] = maxi(0, (e - s + st + 1) / st)
	return out


static func _gather(data: PackedInt32Array, indices: PackedInt32Array,
		axis: int) -> PackedInt32Array:
	var ax := axis_of(axis, data.size())
	var out := PackedInt32Array()
	for i in ax:
		out.append(data[i])
	out.append_array(indices)
	for i in range(ax + 1, data.size()):
		out.append(data[i])
	return out


static func _range_len(ci: Dictionary) -> int:
	if not (ci.has(0) and ci.has(1) and ci.has(2)):
		push_error("KokoroShapes: Range needs constant bounds")
		return 0
	var start: float = (ci[0] as PackedFloat64Array)[0]
	var limit: float = (ci[1] as PackedFloat64Array)[0]
	var delta: float = (ci[2] as PackedFloat64Array)[0]
	if delta == 0.0:
		return 0
	return maxi(0, int(ceil((limit - start) / delta)))


static func _pad(src: PackedInt32Array, pads) -> PackedInt32Array:
	var p := _as_shape(pads)
	var rank := src.size()
	var out := src.duplicate()
	for i in rank:
		out[i] = src[i] + p[i] + p[i + rank]
	return out


static func _tile(src: PackedInt32Array, reps) -> PackedInt32Array:
	var r := _as_shape(reps)
	var out := src.duplicate()
	for i in mini(out.size(), r.size()):
		out[i] *= r[i]
	return out


static func _reduce(src: PackedInt32Array, axes, keepdims: bool,
		noop_empty: bool) -> PackedInt32Array:
	var list := _as_shape(axes)
	if axes == null or list.is_empty():
		if noop_empty:
			return src
		list.resize(src.size())
		for i in src.size():
			list[i] = i
	var drop := {}
	for a in list:
		drop[axis_of(a, src.size())] = true
	var out := PackedInt32Array()
	for i in src.size():
		if drop.has(i):
			if keepdims:
				out.append(1)
		else:
			out.append(src[i])
	return out


static func _matmul(a: PackedInt32Array, b: PackedInt32Array,
		trans_a: bool, trans_b: bool) -> PackedInt32Array:
	var av := a.duplicate()
	var bv := b.duplicate()
	if av.size() == 1:
		av.insert(0, 1)
	if bv.size() == 1:
		bv.append(1)
	if trans_a and av.size() >= 2:
		var t := av[av.size() - 1]
		av[av.size() - 1] = av[av.size() - 2]
		av[av.size() - 2] = t
	if trans_b and bv.size() >= 2:
		var t := bv[bv.size() - 1]
		bv[bv.size() - 1] = bv[bv.size() - 2]
		bv[bv.size() - 2] = t

	var m := av[av.size() - 2]
	var n := bv[bv.size() - 1]
	# broadcast the leading batch dims
	var batch_a := av.slice(0, av.size() - 2)
	var batch_b := bv.slice(0, bv.size() - 2)
	var out := Ops.broadcast_shape(batch_a, batch_b)
	out.append(m)
	out.append(n)
	return out


static func _conv(x: PackedInt32Array, w: PackedInt32Array,
		attrs: Dictionary, transposed: bool) -> PackedInt32Array:
	# Every convolution in this graph is 1-D.
	var pads := _as_shape(attrs.get("pads", [0, 0]))
	var strides := _as_shape(attrs.get("strides", [1]))
	var dil := _as_shape(attrs.get("dilations", [1]))
	var group := int(attrs.get("group", 1))
	var s := strides[0] if strides.size() > 0 else 1
	var d := dil[0] if dil.size() > 0 else 1
	var p0 := pads[0] if pads.size() > 0 else 0
	var p1 := pads[1] if pads.size() > 1 else 0
	var k := w[w.size() - 1]
	var lin := x[2]

	var out := PackedInt32Array([x[0], 0, 0])
	if transposed:
		var outpad := _as_shape(attrs.get("output_padding", [0]))
		var op := outpad[0] if outpad.size() > 0 else 0
		out[1] = w[1] * group                       # [cin, cout/group, k]
		out[2] = (lin - 1) * s + d * (k - 1) + 1 + op - p0 - p1
	else:
		out[1] = w[0]                               # [cout, cin/group, k]
		out[2] = (lin + p0 + p1 - d * (k - 1) - 1) / s + 1
	return out


static func _resize(src: PackedInt32Array, ci: Dictionary) -> PackedInt32Array:
	# scales is input 2, sizes input 3; this graph always supplies scales.
	var out := src.duplicate()
	if ci.has(3) and (ci[3] as PackedFloat64Array).size() == src.size():
		return _as_shape(ci[3])
	var scales = ci.get(2)
	if scales == null:
		return out
	var sc: PackedFloat64Array = scales
	for i in mini(out.size(), sc.size()):
		out[i] = int(floor(float(src[i]) * sc[i]))
	return out


static func _stft(sig: PackedInt32Array, attrs: Dictionary,
		ci: Dictionary) -> PackedInt32Array:
	# `signal` is a GDScript keyword, hence `sig`.
	var step := int((ci[1] as PackedFloat64Array)[0]) if ci.has(1) else 1
	var flen := int((ci[3] as PackedFloat64Array)[0]) if ci.has(3) else 0
	var slen := sig[1]
	var frames := (slen - flen) / step + 1
	var onesided := int(attrs.get("onesided", 1)) != 0
	var freq := (flen / 2 + 1) if onesided else flen
	return PackedInt32Array([sig[0], frames, freq, 2])


static func _lstm(x: PackedInt32Array, attrs: Dictionary) -> Array:
	var hidden := int(attrs.get("hidden_size", 0))
	var ndir := 2 if str(attrs.get("direction", "forward")) == "bidirectional" \
			else 1
	var seq := x[0]
	var batch := x[1]
	return [
		PackedInt32Array([seq, ndir, batch, hidden]),   # Y
		PackedInt32Array([ndir, batch, hidden]),        # Y_h
		PackedInt32Array([ndir, batch, hidden]),        # Y_c
	]
