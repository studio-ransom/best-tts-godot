#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Strided region copy. One kernel covers every pure data-movement op in the
// graph: Transpose, Slice, Concat (one dispatch per input slab), Split,
// Expand (source stride 0), Reshape/Squeeze/Unsqueeze (identity strides),
// constant Pad (fill, then copy the interior back), and ConstantOfShape
// (all source strides 0 against a one-element buffer).

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufSrc { float src[]; };
layout(set = 0, binding = 1) writeonly buffer BufDst { float dst[]; };

layout(push_constant) uniform Params {
	uint n;          // elements in the region
	uint rank;
	uint src_base;
	uint dst_base;
	uvec4 rshape;    // region extents, outermost first
	uvec4 sstride;   // source stride per region axis (0 broadcasts)
	uvec4 dstride;   // destination stride per region axis
} pc;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= pc.n) { return; }

	uint rem = i;
	uint s = pc.src_base;
	uint d = pc.dst_base;
	for (uint k = pc.rank; k > 0u; k--) {
		uint axis = k - 1u;
		uint sz = pc.rshape[axis];
		uint c = rem % sz;
		rem /= sz;
		s += c * pc.sstride[axis];
		d += c * pc.dstride[axis];
	}
	dst[d] = src[s];
}
