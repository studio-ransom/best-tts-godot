#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Final stage: convert the generated float waveform to 16-bit PCM on the GPU,
// so the readback is already in the byte layout AudioStreamWAV wants and the
// CPU never touches per-sample data.
//
// Each invocation packs two consecutive samples into one uint, matching the
// little-endian interleaving of a mono 16-bit stream.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufIn { float samples[]; };
layout(set = 0, binding = 1) writeonly buffer BufOut { uint packed_pairs[]; };

layout(push_constant) uniform Params {
	uint sample_count;
	uint pair_count;
	float gain;
	uint pad_;
} pc;

int to_pcm16(float v) {
	// Clamp before scaling: the vocoder can overshoot slightly and wrapping
	// would turn a peak into a loud click.
	float c = clamp(v * pc.gain, -1.0, 1.0);
	return int(round(c * 32767.0));
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= pc.pair_count) { return; }

	uint i0 = i * 2u;
	uint i1 = i0 + 1u;

	int lo = to_pcm16(samples[i0]);
	int hi = (i1 < pc.sample_count) ? to_pcm16(samples[i1]) : 0;

	packed_pairs[i] = (uint(lo) & 0xFFFFu) | ((uint(hi) & 0xFFFFu) << 16);
}
