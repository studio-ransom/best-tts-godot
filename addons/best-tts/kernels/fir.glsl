#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Linear-phase FIR applied to the finished waveform.
//
// It exists for one job: Kokoro's exported iSTFT overlap-adds with hop 5, and
// the window it ships does not sum to a constant, so the output carries a
// steady tone at the hop rate (24000/5 = 4800 Hz) and its octave at 9600 Hz,
// roughly 25 dB above the surrounding noise floor. The numpy reference has the
// same tone, so this is the model's artefact rather than a bug in the runtime
// -- but it is audible as a ring behind the speech, and two narrow notches
// remove it for about a millisecond of GPU time.
//
// The taps are symmetric, so the filter is zero-phase once the centre offset is
// taken out; edges clamp to the first and last sample.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufIn  { float samples[]; };
layout(set = 0, binding = 1) readonly buffer BufH   { float h[]; };
layout(set = 0, binding = 2) writeonly buffer BufOut { float dst[]; };

layout(push_constant) uniform Params {
	uint sample_count;
	uint taps;
	uint center;
	uint pad_;
} pc;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= pc.sample_count) { return; }

	int n = int(pc.sample_count);
	int base = int(i) - int(pc.center);
	float acc = 0.0;
	for (uint k = 0u; k < pc.taps; k++) {
		int s = clamp(base + int(k), 0, n - 1);
		acc += h[k] * samples[s];
	}
	dst[i] = acc;
}
