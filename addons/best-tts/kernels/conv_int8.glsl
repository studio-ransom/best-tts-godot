#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// ConvInteger, 1-D with groups and dilation. Same integer contract as
// matmul_int8: uint8 activations, int8 weights, zero-point corrected, integer
// accumulator written as float for a later scale multiply.
//
// Layout is x[N, Cin, Lin], w[Cout, Cin/group, K], out[N, Cout, Lout].

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufX { uint x_raw[]; };
layout(set = 0, binding = 1) readonly buffer BufW { uint w_raw[]; };
layout(set = 0, binding = 2) readonly buffer BufXZp { float x_zp[]; };
layout(set = 0, binding = 3) readonly buffer BufWZp { float w_zp[]; };
layout(set = 0, binding = 4) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint n_out;    // N * Cout * Lout
	uint Cin_g;    // Cin / group
	uint Cout;
	uint Lin;
	uint Lout;
	uint K;
	uint stride;
	uint dilation;
	int pad_begin;
	uint group;
	uint Cout_g;   // Cout / group
	uint w_signed; // weights may be int8 or uint8 depending on the node
} pc;

int x_byte(uint i) {
	uint word = x_raw[i >> 2u];
	return int((word >> ((i & 3u) * 8u)) & 0xFFu);
}

int w_byte(uint i) {
	uint word = w_raw[i >> 2u];
	uint b = (word >> ((i & 3u) * 8u)) & 0xFFu;
	// Only sign-extend when the tensor really is int8; the quantizer emits
	// uint8 weights for some layers and those carry a non-zero zero point.
	return (pc.w_signed != 0u && b > 127u) ? int(b) - 256 : int(b);
}

void main() {
	uint g = gl_GlobalInvocationID.x;
	if (g >= pc.n_out) { return; }

	uint l = g % pc.Lout;
	uint oc = (g / pc.Lout) % pc.Cout;
	uint n = g / (pc.Lout * pc.Cout);

	uint grp = oc / pc.Cout_g;
	int xzp = int(x_zp[0]);
	int wzp = int(w_zp[0]);

	int acc = 0;
	for (uint ci = 0u; ci < pc.Cin_g; ci++) {
		uint in_c = grp * pc.Cin_g + ci;
		uint x_base = (n * (pc.Cin_g * pc.group) + in_c) * pc.Lin;
		uint w_base = (oc * pc.Cin_g + ci) * pc.K;
		for (uint k = 0u; k < pc.K; k++) {
			int pos = int(l * pc.stride) + int(k * pc.dilation) + pc.pad_begin;
			if (pos < 0 || pos >= int(pc.Lin)) { continue; }
			acc += (x_byte(x_base + uint(pos)) - xzp)
					* (w_byte(w_base + k) - wzp);
		}
	}
	dst[g] = float(acc);
}
