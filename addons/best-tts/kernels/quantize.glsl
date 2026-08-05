#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Second half of DynamicQuantizeLinear: turn floats into packed uint8 using
// the {min, max} produced by minmax.glsl, and publish the scale and zero
// point that the rest of the graph multiplies back in.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufIn { float src[]; };
layout(set = 0, binding = 1) readonly buffer BufMinMax { float mm[]; };
layout(set = 0, binding = 2) buffer BufQ { uint q[]; };
layout(set = 0, binding = 3) writeonly buffer BufScale { float scale_out[]; };
layout(set = 0, binding = 4) writeonly buffer BufZp { float zp_out[]; };

layout(push_constant) uniform Params {
	uint n;        // element count
	uint words;    // ceil(n / 4)
	uint pad0;
	uint pad1;
} pc;

void main() {
	uint w = gl_GlobalInvocationID.x;
	if (w >= pc.words) { return; }

	float lo = mm[0];
	float hi = mm[1];
	float scale = (hi - lo) / 255.0;
	if (scale == 0.0) { scale = 1.0; }
	float zp = clamp(round(-lo / scale), 0.0, 255.0);

	if (w == 0u) {
		scale_out[0] = scale;
		zp_out[0] = zp;
	}

	// Pack four samples per word so the matmul can read bytes directly.
	uint packed_word = 0u;
	for (uint k = 0u; k < 4u; k++) {
		uint i = w * 4u + k;
		uint v = 0u;
		if (i < pc.n) {
			v = uint(clamp(round(src[i] / scale) + zp, 0.0, 255.0));
		}
		packed_word |= (v & 0xFFu) << (k * 8u);
	}
	q[w] = packed_word;
}
