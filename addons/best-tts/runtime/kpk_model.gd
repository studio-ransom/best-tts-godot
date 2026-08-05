@tool
class_name KpkModel
extends RefCounted

## Loads `model.kpk`, the packed Kokoro graph produced by `tools/pack_model.py`.
##
## The file is deliberately trivial to read so that no protobuf parsing is
## needed at runtime: a 16-byte header, a JSON graph description, and one flat
## weight blob. See tools/NOTES.md for the format.

const MAGIC := 0x314B504B  # "KPK1" little-endian

## Element type codes, shared with the packer.
enum DType { F32 = 0, I8 = 1, U8 = 2, I32 = 3, I64 = 4, BOOL = 5, F16 = 6 }

## Where a node runs. CPU nodes only move shapes and indices around.
enum Place { CPU = 0, GPU = 1 }

const DTYPE_SIZE := {
	DType.F32: 4, DType.I8: 1, DType.U8: 1, DType.I32: 4,
	DType.I64: 8, DType.BOOL: 1, DType.F16: 2,
}

## op_type string for each op id, indexed by `Node.op`.
var ops: PackedStringArray
## Original ONNX tensor name for each tensor id. Debug aid only.
var tensor_names: PackedStringArray
## tensor id -> Weight
var weights: Dictionary
## Array[OpNode] in execution order.
var nodes: Array[OpNode]

var input_ids_tid := -1
var style_tid := -1
var speed_tid := -1
var output_tid := -1
var sample_rate := 24000
var samples_per_frame := 300

## Raw weight blob. Kept so buffers can be uploaded without a second read.
var blob: PackedByteArray


class Weight extends RefCounted:
	var offset: int
	var nbytes: int
	var dtype: int
	var shape: PackedInt32Array

	func count() -> int:
		return nbytes / KpkModel.DTYPE_SIZE[dtype]


class OpNode extends RefCounted:
	var op: int
	var op_name: String       ## resolved for readability in errors
	var inputs: PackedInt32Array   ## -1 marks an omitted optional input
	var outputs: PackedInt32Array
	var attrs: Dictionary
	var place: int

	func _to_string() -> String:
		return "%s(in=%s out=%s)" % [op_name, inputs, outputs]


## Reads and validates the packed model. Returns OK or an error code.
func load_from(path: String) -> Error:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("KpkModel: cannot open %s (%s)" % [path, FileAccess.get_open_error()])
		return ERR_FILE_CANT_OPEN

	if f.get_32() != MAGIC:
		push_error("KpkModel: %s is not a .kpk file" % path)
		return ERR_FILE_UNRECOGNIZED

	var version := f.get_32()
	if version != 1:
		push_error("KpkModel: unsupported version %d" % version)
		return ERR_FILE_UNRECOGNIZED

	var json_len := f.get_32()
	var blob_len := f.get_32()
	var json_bytes := f.get_buffer(json_len)
	blob = f.get_buffer(blob_len)
	f.close()

	if blob.size() != blob_len:
		push_error("KpkModel: truncated weight blob")
		return ERR_FILE_CORRUPT

	var desc = JSON.parse_string(json_bytes.get_string_from_utf8())
	if typeof(desc) != TYPE_DICTIONARY:
		push_error("KpkModel: malformed graph description")
		return ERR_FILE_CORRUPT

	return _build(desc)


func _build(desc: Dictionary) -> Error:
	ops = PackedStringArray(desc["ops"])
	tensor_names = PackedStringArray(desc["tensors"])
	sample_rate = int(desc.get("sample_rate", 24000))
	samples_per_frame = int(desc.get("samples_per_frame", 300))

	var inputs: Dictionary = desc["inputs"]
	input_ids_tid = int(inputs["input_ids"])
	style_tid = int(inputs["style"])
	speed_tid = int(inputs["speed"])
	output_tid = int(desc["output"])

	weights = {}
	for key in desc["weights"]:
		var w: Dictionary = desc["weights"][key]
		var entry := Weight.new()
		entry.offset = int(w["o"])
		entry.nbytes = int(w["n"])
		entry.dtype = int(w["d"])
		entry.shape = PackedInt32Array(w["s"])
		if entry.offset + entry.nbytes > blob.size():
			push_error("KpkModel: weight %d runs past the blob" % int(key))
			return ERR_FILE_CORRUPT
		weights[int(key)] = entry

	nodes = []
	nodes.resize(desc["nodes"].size())
	var i := 0
	for raw in desc["nodes"]:
		var n := OpNode.new()
		n.op = int(raw[0])
		n.op_name = ops[n.op]
		n.inputs = PackedInt32Array(raw[1])
		n.outputs = PackedInt32Array(raw[2])
		n.attrs = raw[3] if raw[3] != null else {}
		n.place = int(raw[4])
		nodes[i] = n
		i += 1

	return OK


## Bytes for one weight tensor, without copying the whole blob.
func weight_bytes(tid: int) -> PackedByteArray:
	var w: Weight = weights[tid]
	return blob.slice(w.offset, w.offset + w.nbytes)


## Name of a tensor, for error messages.
func name_of(tid: int) -> String:
	if tid < 0 or tid >= tensor_names.size():
		return "<%d>" % tid
	return tensor_names[tid]


## Frees the CPU-side blob once every weight has been uploaded to the GPU.
func release_blob() -> void:
	blob = PackedByteArray()
