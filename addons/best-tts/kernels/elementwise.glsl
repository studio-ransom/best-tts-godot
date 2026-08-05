#[compute]
#version 450

// Copyright 2026 Studio Ransom.
// Licensed under the Apache License, Version 2.0. See LICENSE at the
// repository root. Part of best-tts: Kokoro-82M TTS for Godot in pure
// GDScript and GLSL.

// Unary/binary/ternary elementwise ops with full NumPy-style broadcasting.
//
// This single kernel covers roughly three quarters of the graph's GPU nodes
// (Mul, Add, Cast, Div, Sub, Sqrt, Floor, Sin, Pow, LeakyRelu, comparisons,
// Where, ...). The op is selected by a push constant so the executor can
// dispatch any of them without swapping pipelines.
//
// Every GPU tensor in this runtime is float32. Integer-valued tensors that
// reach the GPU are frame counts and indices well under 2^24, so they are
// exact in float; anything needing real integer semantics (quantized matmuls,
// embeddings) has its own kernel.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) readonly buffer BufA { float a[]; };
layout(set = 0, binding = 1) readonly buffer BufB { float b[]; };
layout(set = 0, binding = 2) readonly buffer BufC { float c[]; };
layout(set = 0, binding = 3) writeonly buffer BufOut { float dst[]; };

// 80 bytes, inside the 128-byte push-constant floor Vulkan guarantees.
// Rank never exceeds 4 in this graph, and the executor collapses contiguous
// axes first, so most dispatches run at rank 1 or 2.
layout(push_constant) uniform Params {
	uint op;
	uint n;         // number of output elements
	uint rank;
	float alpha;    // LeakyRelu slope
	uvec4 oshape;
	uvec4 astride;
	uvec4 bstride;
	uvec4 cstride;
} pc;

// keep in sync with runtime/ops.gd
#define OP_ADD        0u
#define OP_SUB        1u
#define OP_MUL        2u
#define OP_DIV        3u
#define OP_POW        4u
#define OP_SQRT       5u
#define OP_EXP        6u
#define OP_SIN        7u
#define OP_COS        8u
#define OP_ATAN       9u
#define OP_TANH      10u
#define OP_SIGMOID   11u
#define OP_FLOOR     12u
#define OP_ROUND     13u
#define OP_LEAKYRELU 14u
#define OP_CLIP      15u
#define OP_EQUAL     16u
#define OP_GREATER   17u
#define OP_LESS      18u
#define OP_GEQ       19u
#define OP_AND       20u
#define OP_WHERE     21u
#define OP_CAST_INT  22u   // truncate toward zero (Cast to an integer type)
#define OP_CAST_BOOL 23u
#define OP_IDENTITY  24u   // Cast between float types, Reshape-as-copy
#define OP_NEG       25u
#define OP_ABS       26u
#define OP_LOG       27u
#define OP_MIN       28u
#define OP_MAX       29u
#define OP_ERF       30u
#define OP_GELU      31u   // tanh-approximate GELU (com.microsoft FastGelu)

uint dim(uint i) { return pc.oshape[i]; }
uint stride_a(uint i) { return pc.astride[i]; }
uint stride_b(uint i) { return pc.bstride[i]; }
uint stride_c(uint i) { return pc.cstride[i]; }

// Rational approximation of erf, max abs error ~1.5e-7 (Abramowitz & Stegun).
float erf_approx(float x) {
	float s = sign(x);
	float ax = abs(x);
	float t = 1.0 / (1.0 + 0.3275911 * ax);
	float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741)
			* t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
	return s * y;
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= pc.n) { return; }

	// Walk the output coordinates from the fastest-moving axis outward,
	// accumulating each source offset with its (possibly zero) stride.
	uint rem = i;
	uint ia = 0u;
	uint ib = 0u;
	uint ic = 0u;
	for (uint d = pc.rank; d > 0u; d--) {
		uint axis = d - 1u;
		uint sz = dim(axis);
		uint coord = rem % sz;
		rem /= sz;
		ia += coord * stride_a(axis);
		ib += coord * stride_b(axis);
		ic += coord * stride_c(axis);
	}

	float x = a[ia];
	float y = b[ib];
	float r;

	switch (pc.op) {
		case OP_ADD:        r = x + y; break;
		case OP_SUB:        r = x - y; break;
		case OP_MUL:        r = x * y; break;
		case OP_DIV:        r = x / y; break;
		case OP_POW:
			// GLSL pow() is undefined for negative bases (NaN in practice),
			// but ONNX allows them with integer exponents — the STFT branch
			// squares signed spectra, so this path is load-bearing.
			r = pow(abs(x), y);
			if (x < 0.0 && fract(y) == 0.0 && mod(y, 2.0) != 0.0) { r = -r; }
			break;
		case OP_SQRT:       r = sqrt(x); break;
		case OP_EXP:        r = exp(x); break;
		case OP_SIN:        r = sin(x); break;
		case OP_COS:        r = cos(x); break;
		case OP_ATAN:       r = atan(x); break;
		case OP_TANH:       r = tanh(x); break;
		case OP_SIGMOID:    r = 1.0 / (1.0 + exp(-x)); break;
		case OP_FLOOR:      r = floor(x); break;
		case OP_ROUND:      r = roundEven(x); break;
		case OP_LEAKYRELU:  r = x < 0.0 ? x * pc.alpha : x; break;
		case OP_CLIP:       r = clamp(x, y, c[ic]); break;
		case OP_EQUAL:      r = (x == y) ? 1.0 : 0.0; break;
		case OP_GREATER:    r = (x > y) ? 1.0 : 0.0; break;
		case OP_LESS:       r = (x < y) ? 1.0 : 0.0; break;
		case OP_GEQ:        r = (x >= y) ? 1.0 : 0.0; break;
		case OP_AND:        r = (x != 0.0 && y != 0.0) ? 1.0 : 0.0; break;
		case OP_WHERE:      r = (x != 0.0) ? y : c[ic]; break;
		case OP_CAST_INT:   r = trunc(x); break;
		case OP_CAST_BOOL:  r = (x != 0.0) ? 1.0 : 0.0; break;
		case OP_IDENTITY:   r = x; break;
		case OP_NEG:        r = -x; break;
		case OP_ABS:        r = abs(x); break;
		case OP_LOG:        r = log(x); break;
		case OP_MIN:        r = min(x, y); break;
		case OP_MAX:        r = max(x, y); break;
		case OP_ERF:        r = erf_approx(x); break;
		case OP_GELU: {
			float u = x + y;   // FastGelu folds its bias into the same op
			r = 0.5 * u * (1.0 + tanh(0.7978845608028654
					* (u + 0.044715 * u * u * u)));
			break;
		}
		default:            r = x; break;
	}

	dst[i] = r;
}
