#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Gather along one axis: out[o, k, j] = data[o, idx[k], j].
//
// The token embedding table is int8, so the source can be read either as
// float32 or as packed 8-bit values; quantized sources come out as raw integer
// values and a following DequantizeLinear applies the scale.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufData { uint data_raw[]; };
layout(set = 0, binding = 1) readonly buffer BufIdx { float indices[]; };
layout(set = 0, binding = 2) writeonly buffer BufOut { float dst[]; };

#define SRC_F32 0u
#define SRC_I8  1u
#define SRC_U8  2u

layout(push_constant) uniform Params {
	uint n;        // total output elements
	uint outer;    // product of dims before the axis
	uint n_idx;    // number of indices
	uint inner;    // product of dims after the axis
	uint dim;      // extent of the gathered axis
	uint mode;
	uint pad0;
	uint pad1;
} pc;

float fetch(uint i) {
	if (pc.mode == SRC_F32) {
		return uintBitsToFloat(data_raw[i]);
	}
	uint word = data_raw[i >> 2u];
	uint b = (word >> ((i & 3u) * 8u)) & 0xFFu;
	if (pc.mode == SRC_I8 && b > 127u) {
		return float(int(b) - 256);
	}
	return float(b);
}

void main() {
	uint g = gl_GlobalInvocationID.x;
	if (g >= pc.n) { return; }

	uint j = g % pc.inner;
	uint k = (g / pc.inner) % pc.n_idx;
	uint o = g / (pc.inner * pc.n_idx);

	int idx = int(indices[k]);
	if (idx < 0) { idx += int(pc.dim); }
	idx = clamp(idx, 0, int(pc.dim) - 1);

	dst[g] = fetch((o * pc.dim + uint(idx)) * pc.inner + j);
}
