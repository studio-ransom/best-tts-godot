#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Small ops that each appear once or twice, folded into one kernel so the
// executor does not pay a pipeline switch for a handful of nodes:
// CumSum, Resize (nearest and linear), reflect Pad, ScatterND, and the
// fp16 -> fp32 unpack used when weights are uploaded.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) readonly buffer BufRaw { uint raw[]; };
layout(set = 0, binding = 3) writeonly buffer BufOut { float dst[]; };

#define MODE_CUMSUM        0u
#define MODE_RESIZE_NEAREST 1u
#define MODE_RESIZE_LINEAR  2u
#define MODE_PAD_REFLECT    3u
#define MODE_SCATTER_ND     4u
#define MODE_UNPACK_F16     5u
#define MODE_DEQUANT_I8     6u

layout(push_constant) uniform Params {
	uint mode;
	uint n;        // output elements
	uint outer;
	uint dim;      // source extent along the active axis
	uint inner;
	uint out_dim;  // destination extent along the active axis
	float scale;   // resize ratio, out/in
	int pad_begin;
	uint update_count;
	uint pad0;
	uint pad1;
	uint pad2;
} pc;

void main() {
	uint g = gl_GlobalInvocationID.x;

	if (pc.mode == MODE_UNPACK_F16) {
		if (g >= pc.n) { return; }
		vec2 pair = unpackHalf2x16(raw[g >> 1u]);
		dst[g] = (g & 1u) == 0u ? pair.x : pair.y;
		return;
	}

	if (pc.mode == MODE_DEQUANT_I8) {
		// LSTM weights are the only int8 tensors we expand to float up front:
		// the recurrent kernel reads them once per timestep, so unpacking per
		// access would dominate. `scale` carries the step, pad_begin the zero
		// point, and src_base (in `outer`) the offset into the packed buffer.
		if (g >= pc.n) { return; }
		uint i = pc.outer + g;
		uint word = raw[i >> 2u];
		uint byte_ = (word >> ((i & 3u) * 8u)) & 0xFFu;
		int v = (byte_ > 127u) ? int(byte_) - 256 : int(byte_);
		dst[g] = float(v - pc.pad_begin) * pc.scale;
		return;
	}

	if (pc.mode == MODE_SCATTER_ND) {
		// b holds flat destination indices, a holds the updates.
		if (g >= pc.update_count) { return; }
		dst[uint(b[g])] = a[g];
		return;
	}

	if (g >= pc.n) { return; }

	uint j = g % pc.inner;
	uint l = (g / pc.inner) % pc.out_dim;
	uint o = g / (pc.inner * pc.out_dim);
	uint base = o * pc.dim * pc.inner + j;

	if (pc.mode == MODE_CUMSUM) {
		float acc = 0.0;
		for (uint k = 0u; k <= l; k++) {
			acc += a[base + k * pc.inner];
		}
		dst[g] = acc;
	} else if (pc.mode == MODE_RESIZE_NEAREST) {
		// Asymmetric coordinate transform with floor rounding: s = l*dim/out_dim.
		//
		// This must be integer arithmetic. Written as floor(float(l)/scale) the
		// driver landed a hair below the integer at exact frame boundaries and
		// floor() dropped to the previous frame, so every upsampled tensor
		// carried one stale sample per frame — 72 of 108 frames on the test
		// case. Measured effect on the waveform is under 0.25 dB in any band,
		// so this is a correctness fix rather than an audible one, but a gather
		// that does not gather is not something to leave in place.
		uint s;
		uint ratio = pc.out_dim / pc.dim;
		if (ratio * pc.dim == pc.out_dim) {
			s = l / ratio;
		} else {
			// No exact integer ratio. Nothing in this graph takes this path;
			// l*dim would overflow 32 bits at real sequence lengths.
			s = uint(floor(float(l) * float(pc.dim) / float(pc.out_dim)));
		}
		s = min(s, pc.dim - 1u);
		dst[g] = a[base + s * pc.inner];
	} else if (pc.mode == MODE_RESIZE_LINEAR) {
		// half-pixel coordinate transform, kept as a multiply-then-divide so it
		// does not inherit the rounding of a precomputed ratio
		float pos = (float(l) + 0.5) * float(pc.dim) / float(pc.out_dim) - 0.5;
		pos = clamp(pos, 0.0, float(pc.dim - 1u));
		uint i0 = uint(floor(pos));
		uint i1 = min(i0 + 1u, pc.dim - 1u);
		float frac = pos - float(i0);
		dst[g] = mix(a[base + i0 * pc.inner], a[base + i1 * pc.inner], frac);
	} else if (pc.mode == MODE_PAD_REFLECT) {
		int pos = int(l) - pc.pad_begin;
		int last = int(pc.dim) - 1;
		// reflect without repeating the edge sample
		if (last > 0) {
			int period = 2 * last;
			pos = ((pos % period) + period) % period;
			if (pos > last) { pos = period - pos; }
		} else {
			pos = 0;
		}
		dst[g] = a[base + uint(pos) * pc.inner];
	}
}
