#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Whole-tensor min/max, the first half of DynamicQuantizeLinear.
//
// ONNX quantizes per tensor, so this must be a single global reduction: one
// workgroup grid-strides the whole buffer and writes {min, max} for the
// quantize pass to consume without a round trip to the CPU.

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufIn { float src[]; };
layout(set = 0, binding = 1) writeonly buffer BufMinMax { float mm[]; };

layout(push_constant) uniform Params {
	uint n;
	uint pad0;
	uint pad1;
	uint pad2;
} pc;

shared float lo[256];
shared float hi[256];

void main() {
	uint tid = gl_LocalInvocationID.x;
	float a = 3.402823466e38;
	float b = -3.402823466e38;
	for (uint i = tid; i < pc.n; i += gl_WorkGroupSize.x) {
		float v = src[i];
		a = min(a, v);
		b = max(b, v);
	}
	lo[tid] = a;
	hi[tid] = b;
	barrier();

	for (uint k = gl_WorkGroupSize.x / 2u; k > 0u; k >>= 1u) {
		if (tid < k) {
			lo[tid] = min(lo[tid], lo[tid + k]);
			hi[tid] = max(hi[tid], hi[tid + k]);
		}
		barrier();
	}

	if (tid == 0u) {
		// The ONNX range always spans zero so the zero point is representable.
		mm[0] = min(lo[0], 0.0);
		mm[1] = max(hi[0], 0.0);
	}
}
