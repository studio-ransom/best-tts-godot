@tool
class_name KokoroOps
extends RefCounted

## Op codes and shape helpers shared by the executor and the GLSL kernels.
##
## The Op values must stay in sync with the `#define OP_*` block in
## kernels/elementwise.glsl.

const MAX_RANK := 4

enum Op {
	ADD = 0, SUB = 1, MUL = 2, DIV = 3, POW = 4, SQRT = 5, EXP = 6,
	SIN = 7, COS = 8, ATAN = 9, TANH = 10, SIGMOID = 11, FLOOR = 12,
	ROUND = 13, LEAKYRELU = 14, CLIP = 15, EQUAL = 16, GREATER = 17,
	LESS = 18, GEQ = 19, AND = 20, WHERE = 21, CAST_INT = 22,
	CAST_BOOL = 23, IDENTITY = 24, NEG = 25, ABS = 26, LOG = 27,
	MIN = 28, MAX = 29, ERF = 30, GELU = 31,
}

## ONNX op_type -> elementwise Op, for ops that need no extra handling.
const UNARY := {
	"Sqrt": Op.SQRT, "Exp": Op.EXP, "Sin": Op.SIN, "Cos": Op.COS,
	"Atan": Op.ATAN, "Tanh": Op.TANH, "Sigmoid": Op.SIGMOID,
	"Floor": Op.FLOOR, "Round": Op.ROUND, "Neg": Op.NEG, "Abs": Op.ABS,
	"Log": Op.LOG, "Erf": Op.ERF, "Identity": Op.IDENTITY,
}

const BINARY := {
	"Add": Op.ADD, "Sub": Op.SUB, "Mul": Op.MUL, "Div": Op.DIV,
	"Pow": Op.POW, "Equal": Op.EQUAL, "Greater": Op.GREATER,
	"Less": Op.LESS, "GreaterOrEqual": Op.GEQ, "And": Op.AND,
	"Min": Op.MIN, "Max": Op.MAX,
}


static func num_elements(shape: PackedInt32Array) -> int:
	var n := 1
	for d in shape:
		n *= d
	return n


## NumPy broadcasting: right-align and take the max of each axis.
static func broadcast_shape(a: PackedInt32Array,
		b: PackedInt32Array) -> PackedInt32Array:
	var rank := maxi(a.size(), b.size())
	var out := PackedInt32Array()
	out.resize(rank)
	for i in rank:
		var da := 1 if i < rank - a.size() else a[i - (rank - a.size())]
		var db := 1 if i < rank - b.size() else b[i - (rank - b.size())]
		if da != db and da != 1 and db != 1:
			push_error("KokoroOps: cannot broadcast %s with %s" % [a, b])
			return PackedInt32Array()
		out[i] = maxi(da, db)
	return out


## Row-major strides for `shape` right-aligned into `out_shape`, with 0 on any
## axis that is broadcast so the kernel can index all operands uniformly.
static func broadcast_strides(shape: PackedInt32Array,
		out_shape: PackedInt32Array) -> PackedInt32Array:
	var rank := out_shape.size()
	var pad := rank - shape.size()
	var strides := PackedInt32Array()
	strides.resize(rank)
	var acc := 1
	for i in range(rank - 1, -1, -1):
		var d := 1 if i < pad else shape[i - pad]
		if d == 1 and out_shape[i] != 1:
			strides[i] = 0        # broadcast: never advance along this axis
		else:
			strides[i] = acc
		if i >= pad:
			acc *= d
	return strides


## Merges adjacent axes that every operand walks contiguously.
##
## Elementwise nodes here are mostly rank 3-4 but rarely broadcast on more than
## one axis, so this usually collapses them to rank 1 and removes the inner
## coordinate loop entirely.
static func collapse(out_shape: PackedInt32Array,
		strides: Array) -> Dictionary:
	var n_ops := strides.size()
	var dims: Array[int] = []
	var st: Array = []
	for _k in n_ops:
		st.append([] as Array[int])

	# Size-1 axes contribute nothing to the index walk.
	for i in out_shape.size():
		if out_shape[i] == 1:
			continue
		dims.append(out_shape[i])
		for k in n_ops:
			st[k].append((strides[k] as PackedInt32Array)[i])

	# Axes i-1 and i fuse when every operand steps through them contiguously,
	# i.e. the outer stride is exactly the inner stride times the inner extent.
	# Broadcast axes satisfy this too, since 0 == 0 * d.
	var i := dims.size() - 1
	while i > 0:
		var mergeable := true
		for k in n_ops:
			if st[k][i - 1] != st[k][i] * dims[i]:
				mergeable = false
				break
		if mergeable:
			dims[i - 1] *= dims[i]
			for k in n_ops:
				st[k][i - 1] = st[k][i]
			dims.remove_at(i)
			for k in n_ops:
				st[k].remove_at(i)
		i -= 1

	if dims.is_empty():
		dims.append(1)
		for k in n_ops:
			st[k].append(0)

	var shape := PackedInt32Array(dims)
	var merged: Array[PackedInt32Array] = []
	for k in n_ops:
		merged.append(PackedInt32Array(st[k]))
	return {"shape": shape, "strides": merged}


## Push-constant blob for elementwise.glsl.
static func elementwise_push(op: int, n: int, out_shape: PackedInt32Array,
		sa: PackedInt32Array, sb: PackedInt32Array, sc: PackedInt32Array,
		alpha := 0.0) -> PackedByteArray:
	var rank := out_shape.size()
	if rank > MAX_RANK:
		push_error("KokoroOps: rank %d exceeds kernel maximum %d"
				% [rank, MAX_RANK])
		rank = MAX_RANK

	var buf := PackedByteArray()
	buf.resize(80)
	buf.encode_u32(0, op)
	buf.encode_u32(4, n)
	buf.encode_u32(8, rank)
	buf.encode_float(12, alpha)
	# Axes are stored right-aligned so index 0 is the outermost of `rank`.
	for i in MAX_RANK:
		var v := out_shape[i] if i < rank else 1
		buf.encode_u32(16 + i * 4, v)
		buf.encode_u32(32 + i * 4, sa[i] if i < rank else 0)
		buf.encode_u32(48 + i * 4, sb[i] if i < rank else 0)
		buf.encode_u32(64 + i * 4, sc[i] if i < rank else 0)
	return buf
