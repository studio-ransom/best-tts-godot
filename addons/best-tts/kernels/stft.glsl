#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Real one-sided STFT. This graph uses n_fft = 20 with hop 5, so a direct DFT
// per frame is cheaper than any FFT machinery: 11 output bins, 20 taps each.
//
// Output is [batch, frames, freq, 2] interleaved real/imaginary, matching ONNX.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufSignal { float signal[]; };
layout(set = 0, binding = 1) readonly buffer BufWindow { float win[]; };
layout(set = 0, binding = 2) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint batch;
	uint frames;
	uint freq;
	uint n_fft;
	uint hop;
	uint signal_len;
	uint pad0;
	uint pad1;
} pc;

const float TAU = 6.283185307179586;

void main() {
	uint g = gl_GlobalInvocationID.x;
	uint total = pc.batch * pc.frames * pc.freq;
	if (g >= total) { return; }

	uint k = g % pc.freq;
	uint f = (g / pc.freq) % pc.frames;
	uint bi = g / (pc.freq * pc.frames);

	uint base = bi * pc.signal_len + f * pc.hop;
	float re = 0.0;
	float im = 0.0;
	for (uint t = 0u; t < pc.n_fft; t++) {
		float v = signal[base + t] * win[t];
		float ang = -TAU * float(k) * float(t) / float(pc.n_fft);
		re += v * cos(ang);
		im += v * sin(ang);
	}

	uint o = ((bi * pc.frames + f) * pc.freq + k) * 2u;
	dst[o] = re;
	dst[o + 1u] = im;
}
