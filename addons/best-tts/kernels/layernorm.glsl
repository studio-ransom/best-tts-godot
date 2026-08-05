#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// LayerNormalization over the last axis, and the fused
// SkipLayerNormalization (com.microsoft) which normalizes x + skip + bias.
// One workgroup per row.

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufX { float x[]; };
layout(set = 0, binding = 1) readonly buffer BufSkip { float skip[]; };
layout(set = 0, binding = 2) readonly buffer BufGamma { float gamma[]; };
layout(set = 0, binding = 3) readonly buffer BufBeta { float beta[]; };
layout(set = 0, binding = 4) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint rows;
	uint cols;
	float eps;
	uint use_skip;    // add the skip tensor before normalizing
	uint use_beta;    // bias is optional
	uint pad0;
	uint pad1;
	uint pad2;
} pc;

shared float red_sum[256];
shared float red_sqsum[256];

void main() {
	uint row = gl_WorkGroupID.x;
	if (row >= pc.rows) { return; }
	uint base = row * pc.cols;
	uint tid = gl_LocalInvocationID.x;

	float s = 0.0;
	float sq = 0.0;
	for (uint c = tid; c < pc.cols; c += gl_WorkGroupSize.x) {
		float v = x[base + c];
		if (pc.use_skip != 0u) { v += skip[base + c]; }
		s += v;
		sq += v * v;
	}
	red_sum[tid] = s;
	red_sqsum[tid] = sq;
	barrier();

	for (uint k = gl_WorkGroupSize.x / 2u; k > 0u; k >>= 1u) {
		if (tid < k) {
			red_sum[tid] += red_sum[tid + k];
			red_sqsum[tid] += red_sqsum[tid + k];
		}
		barrier();
	}

	float inv_n = 1.0 / float(pc.cols);
	float mean = red_sum[0] * inv_n;
	// E[x^2] - E[x]^2 can go slightly negative for near-constant rows.
	float var_ = max(red_sqsum[0] * inv_n - mean * mean, 0.0);
	float inv_std = inversesqrt(var_ + pc.eps);
	barrier();

	for (uint c = tid; c < pc.cols; c += gl_WorkGroupSize.x) {
		float v = x[base + c];
		if (pc.use_skip != 0u) { v += skip[base + c]; }
		float r = (v - mean) * inv_std * gamma[c];
		if (pc.use_beta != 0u) { r += beta[c]; }
		dst[base + c] = r;
	}
}
