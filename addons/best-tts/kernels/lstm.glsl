#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Recurrent half of one LSTM direction.
//
// The input projection X·W and the biases are folded in beforehand by
// matmul.glsl, so this kernel only carries the sequential part: per timestep,
// h·R plus the gate nonlinearities. That is the piece that cannot be
// parallelized over time, so it runs as a single workgroup with one thread per
// hidden unit and the state living in shared memory.
//
// Gate order is ONNX's i, o, f, c, and R is [H, 4H].

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BufXW { float xw[]; };   // [seq, 4H]
layout(set = 0, binding = 1) readonly buffer BufR { float r[]; };     // [H, 4H]
layout(set = 0, binding = 2) readonly buffer BufH0 { float h0[]; };
layout(set = 0, binding = 3) readonly buffer BufC0 { float c0[]; };
layout(set = 0, binding = 4) writeonly buffer BufY { float y[]; };    // [seq, H]
layout(set = 0, binding = 5) writeonly buffer BufYh { float yh[]; };  // final h
layout(set = 0, binding = 6) writeonly buffer BufYc { float yc[]; };  // final c

layout(push_constant) uniform Params {
	uint seq;
	uint H;
	uint reverse;
	uint use_initial;
	uint y_stride;     // element step between timesteps in y
	uint y_base;       // where this direction's slice starts
	uint state_base;   // where this direction's final h/c go
	uint pad2;
} pc;

shared float sh_h[256];
shared float sh_c[256];

void main() {
	uint j = gl_LocalInvocationID.x;
	uint H = pc.H;
	uint H4 = H * 4u;
	if (j >= H) { return; }

	sh_h[j] = (pc.use_initial != 0u) ? h0[j] : 0.0;
	sh_c[j] = (pc.use_initial != 0u) ? c0[j] : 0.0;
	barrier();

	for (uint step = 0u; step < pc.seq; step++) {
		uint t = (pc.reverse != 0u) ? (pc.seq - 1u - step) : step;
		uint xb = t * H4;

		float gi = xw[xb + j];
		float go = xw[xb + H + j];
		float gf = xw[xb + 2u * H + j];
		float gc = xw[xb + 3u * H + j];

		for (uint m = 0u; m < H; m++) {
			float hv = sh_h[m];
			uint rb = m * H4;
			gi += hv * r[rb + j];
			go += hv * r[rb + H + j];
			gf += hv * r[rb + 2u * H + j];
			gc += hv * r[rb + 3u * H + j];
		}

		float i_g = 1.0 / (1.0 + exp(-gi));
		float o_g = 1.0 / (1.0 + exp(-go));
		float f_g = 1.0 / (1.0 + exp(-gf));
		float c_g = tanh(gc);

		float new_c = f_g * sh_c[j] + i_g * c_g;
		float new_h = o_g * tanh(new_c);

		// Every thread must finish reading sh_h before any thread rewrites it.
		barrier();
		sh_h[j] = new_h;
		sh_c[j] = new_c;
		y[pc.y_base + t * pc.y_stride + j] = new_h;
		barrier();
	}

	// Y_h / Y_c: the state after the last timestep this direction consumed.
	yh[pc.state_base + j] = sh_h[j];
	yc[pc.state_base + j] = sh_c[j];
}
