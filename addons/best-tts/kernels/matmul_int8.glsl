#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// MatMulInteger: uint8 activations times int8 weights, both zero-point
// corrected. This is the graph's dominant op (148 nodes).
//
// The integer accumulator is written as float. Worst case here is
// K * 255 * 128 with K = 512, just inside float32's exact-integer range, and
// the result is immediately scaled down by ~1e-4 anyway, so a 1-LSB rounding
// at the extreme is far below the quantization noise already present.

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) readonly buffer BufA { uint a_raw[]; };
layout(set = 0, binding = 1) readonly buffer BufB { uint b_raw[]; };
layout(set = 0, binding = 2) readonly buffer BufAZp { float a_zp[]; };
layout(set = 0, binding = 3) readonly buffer BufBZp { float b_zp[]; };
layout(set = 0, binding = 4) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint M;
	uint N;
	uint K;
	uint batch;
	uint a_batch_stride;
	uint b_batch_stride;
	uint b_zp_per_column;  // per-column zero points for B
	uint b_signed;         // B may be int8 or uint8 depending on the node
} pc;

int byte_at(uint arr_index, bool signed_, uint base) {
	uint i = base + arr_index;
	uint word = (signed_ ? b_raw[i >> 2u] : a_raw[i >> 2u]);
	uint b = (word >> ((i & 3u) * 8u)) & 0xFFu;
	if (signed_ && pc.b_signed != 0u && b > 127u) { return int(b) - 256; }
	return int(b);
}

void main() {
	uint col = gl_GlobalInvocationID.x;
	uint row = gl_GlobalInvocationID.y;
	uint bt = gl_GlobalInvocationID.z;
	if (col >= pc.N || row >= pc.M || bt >= pc.batch) { return; }

	int azp = int(a_zp[0]);
	int bzp = int(b_zp[pc.b_zp_per_column != 0u ? col : 0u]);

	uint abase = bt * pc.a_batch_stride;
	uint bbase = bt * pc.b_batch_stride;

	int acc = 0;
	for (uint k = 0u; k < pc.K; k++) {
		int av = byte_at(row * pc.K + k, false, abase) - azp;
		int bv = byte_at(k * pc.N + col, true, bbase) - bzp;
		acc += av * bv;
	}
	dst[bt * pc.M * pc.N + row * pc.N + col] = float(acc);
}
