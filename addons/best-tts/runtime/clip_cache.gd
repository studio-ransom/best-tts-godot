@tool
class_name KokoroClipCache
extends RefCounted

## Remembers clips that have already been synthesized.
##
## Games say the same things over and over — "I have wares if you have coin",
## every UI confirmation, every hit grunt. Synthesizing those again costs
## ~400 ms and half a gigabyte of VRAM traffic for a byte-identical result.
##
## Two tiers. The memory tier is checked on the calling thread, so a repeated
## line comes back in the same frame it was asked for. The disk tier is checked
## on the worker, because it touches the filesystem; it survives a restart, so
## a game's fixed dialogue is only ever synthesized on the player's first
## playthrough.
##
## Both tiers are bounded and evict least-recently-used first. The cache is
## reached from the main thread and the engine thread, so every entry point
## takes the mutex.

## File header: magic, version, mix rate, PCM bytes, timing JSON bytes.
const MAGIC := 0x3143544B  # "KTC1" little-endian
const HEADER_SIZE := 20

var memory_budget := 32 << 20
var disk_dir := ""
var disk_budget := 128 << 20

var hits := 0
var disk_hits := 0
var misses := 0
var stores := 0
var evictions := 0

## key -> {clip, timings, bytes}. Insertion-ordered, and an entry is re-inserted
## when it is read, which makes `keys()[0]` the least recently used.
var _mem: Dictionary = {}
var _mem_bytes := 0
## Absolute path -> byte size, built once by listing `disk_dir`.
var _disk: Dictionary = {}
var _disk_bytes := 0
var _disk_scanned := false
var _mutex := Mutex.new()


## Identifies a clip by everything that changes its samples. The filter flag is
## in here because flipping `remove_vocoder_tone` has to invalidate the cache
## rather than quietly serve the unfiltered version.
static func key_for(text: String, voice: String, speed: float,
		filtered: bool, gain := 1.0) -> String:
	return "%s|%.4f|%d|%.4f|%s" % [voice, speed, 1 if filtered else 0, gain,
			text]


# ------------------------------------------------------------------- memory

## Looks in the memory tier only. Returns `{clip, timings}` or `{}`.
func peek(key: String) -> Dictionary:
	_mutex.lock()
	var entry = _mem.get(key)
	if entry == null:
		misses += 1
		_mutex.unlock()
		return {}
	# Re-insert to move it to the back of the eviction order.
	_mem.erase(key)
	_mem[key] = entry
	hits += 1
	_mutex.unlock()
	return {"clip": _share(entry["clip"]), "timings": entry["timings"]}


func store(key: String, clip: AudioStreamWAV, timings: Dictionary,
		to_disk: bool) -> void:
	if clip == null or clip.data.is_empty():
		return
	_mutex.lock()
	var bytes := clip.data.size()
	if _mem.has(key):
		_mem_bytes -= int(_mem[key]["bytes"])
		_mem.erase(key)
	_mem[key] = {"clip": clip, "timings": timings, "bytes": bytes}
	_mem_bytes += bytes
	stores += 1
	while _mem_bytes > memory_budget and _mem.size() > 1:
		var oldest = _mem.keys()[0]
		_mem_bytes -= int(_mem[oldest]["bytes"])
		_mem.erase(oldest)
		evictions += 1
	_mutex.unlock()

	if to_disk and disk_dir != "":
		_write_disk(key, clip, timings)


## A separate Resource over the same bytes. PackedByteArray is copy-on-write, so
## this costs nothing, and it means a caller setting `loop_mode` on the clip it
## was handed cannot change what the next caller gets.
static func _share(clip: AudioStreamWAV) -> AudioStreamWAV:
	var out := AudioStreamWAV.new()
	out.format = clip.format
	out.stereo = clip.stereo
	out.mix_rate = clip.mix_rate
	out.data = clip.data
	return out


# --------------------------------------------------------------------- disk

## Looks in the disk tier and promotes a hit into memory. Worker thread only.
func load_from_disk(key: String) -> Dictionary:
	if disk_dir == "":
		return {}
	var path := _path_for(key)
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	if f.get_length() < HEADER_SIZE or f.get_32() != MAGIC or f.get_32() != 1:
		f.close()
		# A file we cannot read is a file from an older build; drop it rather
		# than failing this lookup forever.
		DirAccess.remove_absolute(path)
		return {}
	var rate := f.get_32()
	var pcm_len := f.get_32()
	var json_len := f.get_32()
	var json_bytes := f.get_buffer(json_len)
	var pcm := f.get_buffer(pcm_len)
	f.close()
	if pcm.size() != pcm_len:
		DirAccess.remove_absolute(path)
		return {}

	var clip := AudioStreamWAV.new()
	clip.format = AudioStreamWAV.FORMAT_16_BITS
	clip.stereo = false
	clip.mix_rate = rate
	clip.data = pcm

	var timings := {}
	if json_len > 0:
		var parsed = JSON.parse_string(json_bytes.get_string_from_utf8())
		if typeof(parsed) == TYPE_DICTIONARY:
			timings = parsed

	_mutex.lock()
	disk_hits += 1
	# A disk hit is a memory miss that peek() already counted; un-count it so
	# the hit rate reads as "did we avoid synthesis", which is the useful number.
	misses = maxi(0, misses - 1)
	_mutex.unlock()
	store(key, clip, timings, false)
	return {"clip": _share(clip), "timings": timings}


func _write_disk(key: String, clip: AudioStreamWAV,
		timings: Dictionary) -> void:
	if not _ensure_dir():
		return
	var json := JSON.stringify(timings) if not timings.is_empty() else ""
	var json_bytes := json.to_utf8_buffer()
	var path := _path_for(key)
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f == null:
		return
	f.store_32(MAGIC)
	f.store_32(1)
	f.store_32(clip.mix_rate)
	f.store_32(clip.data.size())
	f.store_32(json_bytes.size())
	f.store_buffer(json_bytes)
	f.store_buffer(clip.data)
	var size := f.get_length()
	f.close()

	_mutex.lock()
	if _disk.has(path):
		_disk_bytes -= int(_disk[path])
	_disk[path] = size
	_disk_bytes += size
	var over := _disk_bytes > disk_budget
	_mutex.unlock()
	if over:
		_evict_disk()


## Oldest-modified first. Access time would be a better ordering but Godot only
## exposes modification time, and rewriting a file on every hit to refresh it
## would cost more than the occasional wrong eviction.
func _evict_disk() -> void:
	_mutex.lock()
	var paths := _disk.keys()
	_mutex.unlock()
	paths.sort_custom(func(a, b):
		return FileAccess.get_modified_time(a) < FileAccess.get_modified_time(b))
	for p in paths:
		_mutex.lock()
		var done := _disk_bytes <= disk_budget
		if not done:
			_disk_bytes -= int(_disk.get(p, 0))
			_disk.erase(p)
			evictions += 1
		_mutex.unlock()
		if done:
			return
		DirAccess.remove_absolute(p)


func _path_for(key: String) -> String:
	return disk_dir.path_join(key.sha256_text() + ".ktc")


func _ensure_dir() -> bool:
	if not _disk_scanned:
		DirAccess.make_dir_recursive_absolute(disk_dir)
		_scan_disk()
	return DirAccess.dir_exists_absolute(disk_dir)


func _scan_disk() -> void:
	_disk_scanned = true
	var dir := DirAccess.open(disk_dir)
	if dir == null:
		return
	_mutex.lock()
	_disk.clear()
	_disk_bytes = 0
	for name in dir.get_files():
		if not name.ends_with(".ktc"):
			continue
		var path := disk_dir.path_join(name)
		var f := FileAccess.open(path, FileAccess.READ)
		if f == null:
			continue
		var size := f.get_length()
		f.close()
		_disk[path] = size
		_disk_bytes += size
	_mutex.unlock()


# -------------------------------------------------------------------- admin

func clear(also_disk := false) -> void:
	_mutex.lock()
	_mem.clear()
	_mem_bytes = 0
	var paths := _disk.keys() if also_disk else []
	if also_disk:
		_disk.clear()
		_disk_bytes = 0
	_mutex.unlock()
	for p in paths:
		DirAccess.remove_absolute(p)


func stats() -> Dictionary:
	_mutex.lock()
	var total := hits + disk_hits + misses
	var out := {
		"entries": _mem.size(),
		"memory_bytes": _mem_bytes,
		"memory_budget": memory_budget,
		"disk_files": _disk.size(),
		"disk_bytes": _disk_bytes,
		"hits": hits,
		"disk_hits": disk_hits,
		"misses": misses,
		"stores": stores,
		"evictions": evictions,
		"hit_rate": float(hits + disk_hits) / maxf(1.0, float(total)),
	}
	_mutex.unlock()
	return out
