#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Float 1-D convolution, forward and transposed, with groups and dilation.
// Used by the iSTFTNet generator's upsampling stack and the F0/N pooling.
//
//   forward     x[N,Cin,Lin]  w[Cout, Cin/g, K]   -> [N, Cout, Lout]
//   transposed  x[N,Cin,Lin]  w[Cin, Cout/g, K]   -> [N, Cout, Lout]

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufX { float x[]; };
layout(set = 0, binding = 1) readonly buffer BufW { float w[]; };
layout(set = 0, binding = 2) readonly buffer BufBias { float bias[]; };
layout(set = 0, binding = 3) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint n_out;
	uint Cin_g;
	uint Cout;
	uint Cout_g;
	uint Lin;
	uint Lout;
	uint K;
	uint stride;
	uint dilation;
	int pad_begin;
	uint group;
	uint mode;        // 0 forward, 1 transposed
	uint use_bias;
	uint pad0;
	uint pad1;
	uint pad2;
} pc;

void main() {
	uint g = gl_GlobalInvocationID.x;
	if (g >= pc.n_out) { return; }

	uint l = g % pc.Lout;
	uint oc = (g / pc.Lout) % pc.Cout;
	uint n = g / (pc.Lout * pc.Cout);

	uint grp = oc / pc.Cout_g;
	uint Cin = pc.Cin_g * pc.group;
	float acc = 0.0;

	if (pc.mode == 0u) {
		for (uint ci = 0u; ci < pc.Cin_g; ci++) {
			uint xb = (n * Cin + grp * pc.Cin_g + ci) * pc.Lin;
			uint wb = (oc * pc.Cin_g + ci) * pc.K;
			for (uint k = 0u; k < pc.K; k++) {
				int pos = int(l * pc.stride) + int(k * pc.dilation)
						+ pc.pad_begin;
				if (pos < 0 || pos >= int(pc.Lin)) { continue; }
				acc += x[xb + uint(pos)] * w[wb + k];
			}
		}
	} else {
		// Scatter form read backwards: find every input tap that lands on l.
		uint oc_g = oc % pc.Cout_g;
		for (uint ci = 0u; ci < pc.Cin_g; ci++) {
			uint in_c = grp * pc.Cin_g + ci;
			uint xb = (n * Cin + in_c) * pc.Lin;
			uint wb = (in_c * pc.Cout_g + oc_g) * pc.K;
			for (uint k = 0u; k < pc.K; k++) {
				int num = int(l) - pc.pad_begin - int(k * pc.dilation);
				if (num < 0) { continue; }
				if (uint(num) % pc.stride != 0u) { continue; }
				uint i = uint(num) / pc.stride;
				if (i >= pc.Lin) { continue; }
				acc += x[xb + i] * w[wb + k];
			}
		}
	}

	if (pc.use_bias != 0u) { acc += bias[oc]; }
	dst[g] = acc;
}
