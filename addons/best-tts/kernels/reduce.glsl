#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// ReduceMean / ReduceSum over a single axis, viewed as [outer, dim, inner].
// One workgroup cooperatively reduces one (outer, inner) lane.

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufIn { float src[]; };
layout(set = 0, binding = 1) writeonly buffer BufOut { float dst[]; };

#define RED_SUM  0u
#define RED_MEAN 1u
#define RED_MAX  2u

layout(push_constant) uniform Params {
	uint lanes;   // outer * inner
	uint dim;
	uint inner;
	uint mode;
} pc;

shared float partial[256];

void main() {
	uint lane = gl_WorkGroupID.x;
	if (lane >= pc.lanes) { return; }

	uint o = lane / pc.inner;
	uint j = lane % pc.inner;
	uint base = o * pc.dim * pc.inner + j;

	float acc = (pc.mode == RED_MAX) ? -3.402823466e38 : 0.0;
	for (uint d = gl_LocalInvocationID.x; d < pc.dim; d += gl_WorkGroupSize.x) {
		float v = src[base + d * pc.inner];
		acc = (pc.mode == RED_MAX) ? max(acc, v) : acc + v;
	}
	partial[gl_LocalInvocationID.x] = acc;
	barrier();

	for (uint s = gl_WorkGroupSize.x / 2u; s > 0u; s >>= 1u) {
		if (gl_LocalInvocationID.x < s) {
			float a = partial[gl_LocalInvocationID.x];
			float b = partial[gl_LocalInvocationID.x + s];
			partial[gl_LocalInvocationID.x] =
					(pc.mode == RED_MAX) ? max(a, b) : a + b;
		}
		barrier();
	}

	if (gl_LocalInvocationID.x == 0u) {
		float r = partial[0];
		if (pc.mode == RED_MEAN) { r /= float(pc.dim); }
		dst[lane] = r;
	}
}
