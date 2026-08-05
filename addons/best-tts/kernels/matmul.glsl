#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Batched float matmul with optional operand transposes, covering MatMul and
// the com.microsoft FusedMatMul (which this graph uses once, with transA).
//
// A is [batch, M, K], B is [batch, K, N] after any transpose; the batch axis
// broadcasts so a shared weight matrix can be applied across heads.

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint M;
	uint N;
	uint K;
	uint batch;
	uint a_batch_stride;   // 0 when A is shared across the batch
	uint b_batch_stride;
	uint trans_a;
	uint trans_b;
	float alpha;
	uint pad0;
	uint pad1;
	uint pad2;
} pc;

void main() {
	uint col = gl_GlobalInvocationID.x;
	uint row = gl_GlobalInvocationID.y;
	uint bt = gl_GlobalInvocationID.z;
	if (col >= pc.N || row >= pc.M || bt >= pc.batch) { return; }

	uint abase = bt * pc.a_batch_stride;
	uint bbase = bt * pc.b_batch_stride;

	float acc = 0.0;
	for (uint k = 0u; k < pc.K; k++) {
		float av = (pc.trans_a != 0u)
				? a[abase + k * pc.M + row]
				: a[abase + row * pc.K + k];
		float bv = (pc.trans_b != 0u)
				? b[bbase + col * pc.K + k]
				: b[bbase + k * pc.N + col];
		acc += av * bv;
	}
	dst[bt * pc.M * pc.N + row * pc.N + col] = acc * pc.alpha;
}
