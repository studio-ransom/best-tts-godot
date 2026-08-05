#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Row-wise softmax over the last axis, max-shifted for stability.
// One workgroup per row; used by the ALBERT attention.

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufIn { float src[]; };
layout(set = 0, binding = 1) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint rows;
	uint cols;
	uint pad0;
	uint pad1;
} pc;

shared float red[256];

void main() {
	uint row = gl_WorkGroupID.x;
	if (row >= pc.rows) { return; }
	uint base = row * pc.cols;
	uint tid = gl_LocalInvocationID.x;

	float m = -3.402823466e38;
	for (uint c = tid; c < pc.cols; c += gl_WorkGroupSize.x) {
		m = max(m, src[base + c]);
	}
	red[tid] = m;
	barrier();
	for (uint k = gl_WorkGroupSize.x / 2u; k > 0u; k >>= 1u) {
		if (tid < k) { red[tid] = max(red[tid], red[tid + k]); }
		barrier();
	}
	float row_max = red[0];
	barrier();

	float s = 0.0;
	for (uint c = tid; c < pc.cols; c += gl_WorkGroupSize.x) {
		s += exp(src[base + c] - row_max);
	}
	red[tid] = s;
	barrier();
	for (uint k = gl_WorkGroupSize.x / 2u; k > 0u; k >>= 1u) {
		if (tid < k) { red[tid] += red[tid + k]; }
		barrier();
	}
	float inv = 1.0 / red[0];
	barrier();

	for (uint c = tid; c < pc.cols; c += gl_WorkGroupSize.x) {
		dst[base + c] = exp(src[base + c] - row_max) * inv;
	}
}
