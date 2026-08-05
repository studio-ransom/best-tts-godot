"""Dev-only: a numpy interpreter for the Kokoro q8f16 graph.

Why this exists: onnxruntime cannot be coaxed into revealing trustworthy
intermediate tensors for this model (exposing them perturbs its mandatory
fp16->fp32 cast insertion and corrupts the run). So we reimplement the graph in
numpy, validate the final waveform against ORT's golden output, and then use
this as the per-node oracle for every GLSL kernel.

It doubles as the executable specification: the GDScript/GLSL runtime is a port
of the semantics written here.
"""
import os

import numpy as np
import onnx
from onnx import numpy_helper

from voice_util import ROOT, load_voice, sample_tokens

# Dev-only source model. It lives under tools/ (which carries a .gdignore) so
# Godot never scans 86 MB on project import; the addon ships only model.kpk.
MODEL = os.path.join(ROOT, "tools", "model_src", "model_q8f16.onnx")

ONNX_DTYPE = {
    1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
    6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.float64,
    12: np.uint32, 13: np.uint64,
}


def _attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.FLOATS:
            out[a.name] = list(a.floats)
        elif a.type == onnx.AttributeProto.TENSOR:
            out[a.name] = numpy_helper.to_array(a.t)
    return out


def _f32(x):
    """Upcast fp16 to fp32 for compute, mirroring ORT's CPU cast insertion."""
    return x.astype(np.float32) if x.dtype == np.float16 else x


# --------------------------------------------------------------------------
# op implementations
# --------------------------------------------------------------------------

def op_conv1d(x, w, b, pads, strides, dilations, group, transposed=False,
              output_padding=0):
    """Shared 1-D conv / conv-transpose in fp32 (all convs in this model are 1-D)."""
    x = _f32(x).astype(np.float32)
    w = _f32(w).astype(np.float32)
    p0, p1 = pads
    s = strides
    d = dilations

    if not transposed:
        n, cin, lin = x.shape
        cout, cing, k = w.shape
        xp = np.pad(x, ((0, 0), (0, 0), (p0, p1)))
        lout = (xp.shape[2] - d * (k - 1) - 1) // s + 1
        # im2col per group, then one matmul per group
        out = np.empty((n, cout, lout), dtype=np.float32)
        cout_g = cout // group
        for g in range(group):
            xs = xp[:, g * cing:(g + 1) * cing, :]
            cols = np.empty((n, cing * k, lout), dtype=np.float32)
            for j in range(k):
                cols[:, j::k, :] = xs[:, :, j * d: j * d + lout * s: s]
            wg = w[g * cout_g:(g + 1) * cout_g].reshape(cout_g, cing * k)
            out[:, g * cout_g:(g + 1) * cout_g, :] = np.einsum(
                "oc,ncl->nol", wg, cols)
    else:
        # ConvTranspose: w is [cin, cout/group, k]
        n, cin, lin = x.shape
        _, cout_g, k = w.shape
        cout = cout_g * group
        cin_g = cin // group
        lfull = (lin - 1) * s + d * (k - 1) + 1 + output_padding
        acc = np.zeros((n, cout, lfull), dtype=np.float32)
        for g in range(group):
            xs = x[:, g * cin_g:(g + 1) * cin_g, :]
            wg = w[g * cin_g:(g + 1) * cin_g]           # [cin_g, cout_g, k]
            contrib = np.einsum("nil,iok->nokl", xs, wg)  # [n,cout_g,k,lin]
            for j in range(k):
                start = j * d
                acc[:, g * cout_g:(g + 1) * cout_g,
                    start:start + (lin - 1) * s + 1:s] += contrib[:, :, j, :]
        out = acc[:, :, p0: lfull - p1] if (p0 or p1) else acc

    if b is not None:
        out = out + _f32(b).astype(np.float32).reshape(1, -1, 1)
    return out


def op_dynamic_quantize_linear(x):
    x = _f32(x).astype(np.float32)
    qmin, qmax = 0.0, 255.0
    lo = min(0.0, float(x.min()))
    hi = max(0.0, float(x.max()))
    scale = (hi - lo) / (qmax - qmin)
    if scale == 0:
        scale = 1.0
    zp = np.clip(np.rint(qmin - lo / scale), qmin, qmax)
    y = np.clip(np.rint(x / scale) + zp, qmin, qmax).astype(np.uint8)
    return y, np.float32(scale), np.uint8(zp)


def op_matmul_integer(a, b, a_zp, b_zp):
    ai = a.astype(np.int32)
    bi = b.astype(np.int32)
    if a_zp is not None:
        ai = ai - np.asarray(a_zp).astype(np.int32)
    if b_zp is not None:
        z = np.asarray(b_zp).astype(np.int32)
        bi = bi - (z if z.ndim == 0 else z.reshape(1, -1))
    return (ai @ bi).astype(np.int32)


def op_conv_integer(x, w, x_zp, w_zp, pads, strides, dilations, group):
    xi = x.astype(np.int32) - (0 if x_zp is None else np.int32(x_zp))
    wi = w.astype(np.int32) - (0 if w_zp is None else np.int32(w_zp))
    return op_conv1d(xi.astype(np.float32), wi.astype(np.float32), None,
                     pads, strides, dilations, group).round().astype(np.int32)


def op_lstm(x, w, r, b, init_h, init_c, w_scale, w_zp, r_scale, r_zp,
            hidden_size, bidirectional, quantize_activations=False):
    """DynamicQuantizeLSTM (com.microsoft).

    W is [dir, input_size, 4H] and R is [dir, H, 4H] -- already transposed
    relative to the standard ONNX LSTM layout. Gate order is ONNX's i, o, f, c.

    ``quantize_activations`` reproduces what the op literally does: quantize X
    and each hidden state to uint8 at runtime and use integer matmuls. That is
    only needed to prove parity with onnxruntime -- the shipped runtime keeps
    the weights dequantized in fp32, which carries less error.
    """
    x = _f32(x).astype(np.float32)
    seq, batch, _ = x.shape
    ndir = 2 if bidirectional else 1
    hs = hidden_size

    def deq(q, scale, zp):
        q = q.astype(np.float32)
        if zp is not None:
            q = q - np.asarray(zp).astype(np.float32)
        return q * np.asarray(scale, dtype=np.float32)

    ys = np.zeros((seq, ndir, batch, hs), dtype=np.float32)
    y_h = np.zeros((ndir, batch, hs), dtype=np.float32)
    y_c = np.zeros((ndir, batch, hs), dtype=np.float32)

    for d in range(ndir):
        wz = None if w_zp is None else np.asarray(w_zp).reshape(-1)[d]
        rz = None if r_zp is None else np.asarray(r_zp).reshape(-1)[d]
        ws = np.asarray(w_scale).reshape(-1)
        rs = np.asarray(r_scale).reshape(-1)
        wd = deq(w[d], ws[d] if ws.size == ndir else ws, wz)
        rd = deq(r[d], rs[d] if rs.size == ndir else rs, rz)
        bd = (np.zeros(8 * hs, np.float32) if b is None
              else _f32(b)[d].astype(np.float32))
        wb, rb = bd[:4 * hs], bd[4 * hs:]

        h = (np.zeros((batch, hs), np.float32) if init_h is None
             else _f32(init_h)[d].astype(np.float32).copy())
        c = (np.zeros((batch, hs), np.float32) if init_c is None
             else _f32(init_c)[d].astype(np.float32).copy())

        if quantize_activations:
            xq, xs, xzp = op_dynamic_quantize_linear(x.reshape(seq * batch, -1))
            wq = w[d].astype(np.int32) - (0 if wz is None else int(wz))
            xw = ((xq.astype(np.int32) - int(xzp)) @ wq).astype(np.float32)
            xw = xw * (np.float32(xs) * np.float32(
                ws[d] if ws.size == ndir else ws))
        else:
            xw = x.reshape(seq * batch, -1) @ wd      # precompute input terms
        xw = xw.reshape(seq, batch, 4 * hs) + (wb + rb)

        rq = (r[d].astype(np.int32) - (0 if rz is None else int(rz))
              if quantize_activations else None)
        rsc = np.float32(rs[d] if rs.size == ndir else rs)

        order = range(seq) if d == 0 else range(seq - 1, -1, -1)
        for t in order:
            if quantize_activations:
                hq, hsc, hzp = op_dynamic_quantize_linear(h)
                hr = ((hq.astype(np.int32) - int(hzp)) @ rq).astype(np.float32)
                g = xw[t] + hr * (np.float32(hsc) * rsc)
            else:
                g = xw[t] + h @ rd
            i = 1.0 / (1.0 + np.exp(-g[:, 0:hs]))
            o = 1.0 / (1.0 + np.exp(-g[:, hs:2 * hs]))
            f = 1.0 / (1.0 + np.exp(-g[:, 2 * hs:3 * hs]))
            cc = np.tanh(g[:, 3 * hs:4 * hs])
            c = f * c + i * cc
            h = o * np.tanh(c)
            ys[t, d] = h
        y_h[d], y_c[d] = h, c
    return ys, y_h, y_c


def op_stft(signal, frame_step, window, frame_length, onesided):
    sig = _f32(signal).astype(np.float32)
    if sig.ndim == 3 and sig.shape[-1] == 1:
        sig = sig[:, :, 0]
    win = _f32(window).astype(np.float32)
    n = int(frame_length)
    step = int(frame_step)
    batch, slen = sig.shape
    nframes = (slen - n) // step + 1
    nfreq = n // 2 + 1 if onesided else n
    k = np.arange(nfreq)[:, None]
    t = np.arange(n)[None, :]
    ang = -2.0 * np.pi * k * t / n
    cos_b, sin_b = np.cos(ang), np.sin(ang)
    out = np.empty((batch, nframes, nfreq, 2), dtype=np.float32)
    idx = np.arange(nframes)[:, None] * step + np.arange(n)[None, :]
    for bi in range(batch):
        frames = sig[bi][idx] * win                    # [nframes, n]
        out[bi, :, :, 0] = frames @ cos_b.T
        out[bi, :, :, 1] = frames @ sin_b.T
    return out


def op_resize(x, scales, mode, ctm, nearest_mode):
    """1-D resize along the last axis (the only form this graph uses)."""
    x = _f32(x).astype(np.float32)
    scale = float(np.asarray(scales).reshape(-1)[-1])
    lin = x.shape[-1]
    lout = int(np.floor(lin * scale))
    if mode == "nearest":
        if ctm != "asymmetric" or nearest_mode != "floor":
            raise NotImplementedError(f"resize nearest {ctm}/{nearest_mode}")
        src = np.floor(np.arange(lout) / scale).astype(np.int64)
        return x[..., np.clip(src, 0, lin - 1)]
    if mode == "linear":
        if ctm != "half_pixel":
            raise NotImplementedError(f"resize linear {ctm}")
        pos = (np.arange(lout) + 0.5) / scale - 0.5
        pos = np.clip(pos, 0, lin - 1)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.clip(i0 + 1, 0, lin - 1)
        frac = (pos - i0).astype(np.float32)
        return x[..., i0] * (1.0 - frac) + x[..., i1] * frac
    raise NotImplementedError(f"resize mode {mode}")


def op_pad(x, pads, mode, value=0.0):
    r = x.ndim
    widths = [(int(pads[i]), int(pads[i + r])) for i in range(r)]
    if mode == "reflect":
        return np.pad(x, widths, mode="reflect")
    if mode == "edge":
        return np.pad(x, widths, mode="edge")
    return np.pad(x, widths, mode="constant", constant_values=value)


def op_softmax(x, axis):
    x = _f32(x).astype(np.float32)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def op_layernorm(x, scale, bias, axis, eps):
    x = _f32(x).astype(np.float32)
    axes = tuple(range(axis if axis >= 0 else x.ndim + axis, x.ndim))
    mu = x.mean(axis=axes, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=axes, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    y = y * _f32(scale).astype(np.float32)
    if bias is not None:
        y = y + _f32(bias).astype(np.float32)
    return y


def op_fast_gelu(x, bias):
    x = _f32(x).astype(np.float32)
    if bias is not None:
        x = x + _f32(bias).astype(np.float32)
    c = np.float32(0.7978845608028654)   # sqrt(2/pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def op_scatternd(data, indices, updates):
    out = data.copy()
    idx = indices.astype(np.int64)
    k = idx.shape[-1]
    flat = idx.reshape(-1, k)
    upd = updates.reshape((flat.shape[0],) + data.shape[k:])
    for i in range(flat.shape[0]):
        out[tuple(flat[i])] = upd[i]
    return out


def op_slice(data, starts, ends, axes, steps):
    r = data.ndim
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)
    sl = [slice(None)] * r
    for s, e, a, st in zip(starts, ends, axes, steps):
        a = int(a) % r
        s, e, st = int(s), int(e), int(st)
        dim = data.shape[a]
        if st > 0:
            s = min(max(s if s >= 0 else s + dim, 0), dim)
            e = min(max(e if e >= 0 else e + dim, 0), dim)
        else:
            s = min(max(s if s >= 0 else s + dim, -1), dim - 1)
            e = min(max(e if e >= 0 else e + dim, -1), dim - 1)
        sl[a] = slice(s, e, st)
    return data[tuple(sl)]


# --------------------------------------------------------------------------
# interpreter
# --------------------------------------------------------------------------

class RefRuntime:
    """Interpreter for the Kokoro graph.

    ``fp16_boundaries`` reproduces onnxruntime's CPU behaviour: it has no fp16
    kernels, so it computes each node in fp32 and rounds the result back to the
    node's declared fp16 output type. Enabling it lets us match ORT closely
    enough to prove each op is implemented correctly; the shipped runtime keeps
    it off and stays in fp32, which is strictly closer to the original
    unquantized model.
    """

    def __init__(self, path=MODEL, fp16_boundaries=False,
                 quantized_lstm=False):
        self.model = onnx.load(path)
        self.graph = self.model.graph
        self.inits = {t.name: numpy_helper.to_array(t)
                      for t in self.graph.initializer}
        self.quantized_lstm = quantized_lstm
        self.fp16_boundaries = fp16_boundaries
        self.fp16_outputs = self._find_fp16_outputs() if fp16_boundaries else set()

    def _find_fp16_outputs(self):
        """Tensor names ORT materializes as fp16."""
        inferred = onnx.shape_inference.infer_shapes(
            self.model, strict_mode=False, data_prop=True)
        declared = {}
        for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
            declared[vi.name] = vi.type.tensor_type.elem_type
        fp16 = {n for n, t in declared.items() if t == 10}
        init_fp16 = {t.name for t in self.graph.initializer if t.data_type == 10}
        # Cast nodes targeting fp16 are eliminated by ORT rather than honoured:
        # the graph casts int32 quantized accumulators straight to fp16, which
        # would overflow. Their results stay in fp32 until a later boundary.
        for node in self.graph.node:
            if node.op_type == "Cast":
                fp16.discard(node.output[0])
        # Contrib ops (SkipLayerNormalization, FastGelu, ...) have no inferred
        # type; they stay in whatever precision their data input carries.
        for node in self.graph.node:
            for o in node.output:
                if not o or o in declared:
                    continue
                if any(i in fp16 or i in init_fp16 for i in node.input if i):
                    fp16.add(o)
        return fp16

    def run(self, input_ids, style, speed=1.0, capture=None, strict=False):
        """Execute the graph; returns dict of every tensor (or only `capture`).

        With ``strict`` the run aborts at the first node that produces a
        non-finite value, which is how we localize porting bugs.
        """
        env = dict(self.inits)
        env["input_ids"] = np.asarray(input_ids, dtype=np.int64).reshape(1, -1)
        env["style"] = np.asarray(style, dtype=np.float32).reshape(1, 256)
        env["speed"] = np.asarray([speed], dtype=np.float32)

        for node in self.graph.node:
            try:
                self._exec(node, env)
                if self.fp16_boundaries:
                    for o in node.output:
                        v = env.get(o)
                        if (o in self.fp16_outputs and v is not None
                                and v.dtype == np.float32):
                            env[o] = v.astype(np.float16).astype(np.float32)
            except Exception as exc:
                shapes = [(i, getattr(env.get(i), "shape", "?"))
                          for i in node.input if i]
                raise RuntimeError(
                    f"{node.op_type} '{node.name}' failed: {exc}\n"
                    f"  inputs: {shapes}") from exc
            if strict:
                for o in node.output:
                    v = env.get(o)
                    if v is None or v.dtype.kind != "f" or not v.size:
                        continue
                    if not np.isfinite(v).all():
                        ins = [(i, env[i].shape, str(env[i].dtype),
                                (float(np.nanmin(_f32(env[i]))),
                                 float(np.nanmax(_f32(env[i]))))
                                if env[i].dtype.kind in "fiu" and env[i].size
                                else None)
                               for i in node.input if i and i in env]
                        raise RuntimeError(
                            f"non-finite output at {node.op_type} "
                            f"'{node.name}' -> {o} {v.shape}\n  inputs: {ins}")

        if capture is not None:
            return {k: env[k] for k in capture if k in env}
        return env

    def _exec(self, node, env):
        self.dispatch(node.op_type, list(node.input), list(node.output),
                      _attrs(node), env)

    def dispatch(self, op, node_input, node_output, at, env):
        """Execute one op. Driven either by an ONNX node or by a packed one,
        so tools/kpk_runtime.py can share exactly these semantics."""
        class _N:
            input = node_input
            output = node_output
        node = _N

        def inp(i, default=None):
            if i >= len(node.input) or not node.input[i]:
                return default
            return env[node.input[i]]

        x = inp(0)
        y = inp(1)
        out = None

        # -- elementwise / math ------------------------------------------
        if op == "Add":
            out = _f32(x) + _f32(y)
        elif op == "Sub":
            out = _f32(x) - _f32(y)
        elif op == "Mul":
            out = _f32(x) * _f32(y)
        elif op == "Div":
            a, b = _f32(x), _f32(y)
            out = (a // b if np.issubdtype(a.dtype, np.integer) and
                   np.issubdtype(b.dtype, np.integer) else a / b)
        elif op == "Pow":
            out = np.power(_f32(x), _f32(y))
        elif op == "Sqrt":
            out = np.sqrt(_f32(x))
        elif op == "Exp":
            out = np.exp(_f32(x))
        elif op == "Sin":
            out = np.sin(_f32(x))
        elif op == "Cos":
            out = np.cos(_f32(x))
        elif op == "Atan":
            out = np.arctan(_f32(x))
        elif op == "Tanh":
            out = np.tanh(_f32(x))
        elif op == "Sigmoid":
            out = 1.0 / (1.0 + np.exp(-_f32(x)))
        elif op == "Floor":
            out = np.floor(_f32(x))
        elif op == "Round":
            out = np.rint(_f32(x))
        elif op == "LeakyRelu":
            a = np.float32(at.get("alpha", 0.01))
            v = _f32(x)
            out = np.where(v < 0, v * a, v)
        elif op == "Clip":
            lo, hi = inp(1), inp(2)
            out = _f32(x)
            if lo is not None:
                out = np.maximum(out, _f32(lo))
            if hi is not None:
                out = np.minimum(out, _f32(hi))
        elif op == "Erf":
            from scipy.special import erf
            out = erf(_f32(x))

        # -- comparison / logic ------------------------------------------
        elif op == "Equal":
            out = np.equal(x, y)
        elif op == "Greater":
            out = np.greater(_f32(x), _f32(y))
        elif op == "Less":
            out = np.less(_f32(x), _f32(y))
        elif op == "GreaterOrEqual":
            out = np.greater_equal(_f32(x), _f32(y))
        elif op == "And":
            out = np.logical_and(x, y)
        elif op == "Where":
            out = np.where(x, y, inp(2))

        # -- shape / layout ----------------------------------------------
        elif op == "Shape":
            s = np.array(x.shape, dtype=np.int64)
            start = at.get("start", 0)
            end = at.get("end", len(s))
            out = s[start:end]
        elif op == "Reshape":
            shape = [int(v) for v in np.asarray(y).reshape(-1)]
            if not at.get("allowzero", 0):
                shape = [x.shape[i] if v == 0 else v
                         for i, v in enumerate(shape)]
            out = x.reshape(shape)
        elif op == "Transpose":
            out = np.transpose(x, at.get("perm"))
        elif op == "Unsqueeze":
            axes = inp(1) if len(node.input) > 1 else at.get("axes")
            for a in sorted(int(v) for v in np.asarray(axes).reshape(-1)):
                x = np.expand_dims(x, a)
            out = x
        elif op == "Squeeze":
            axes = inp(1) if len(node.input) > 1 else at.get("axes")
            if axes is None:
                out = np.squeeze(x)
            else:
                out = np.squeeze(x, tuple(int(v) for v in
                                          np.asarray(axes).reshape(-1)))
        elif op == "Concat":
            vals = [env[i] for i in node.input if i]
            if any(v.dtype == np.float16 for v in vals):
                vals = [_f32(v) for v in vals]
            out = np.concatenate(vals, axis=at["axis"])
        elif op == "Split":
            axis = at.get("axis", 0)
            split = inp(1)
            if split is None:
                parts = np.array_split(x, len(node.output), axis=axis)
            else:
                cuts = np.cumsum([int(v) for v in split])[:-1]
                parts = np.split(x, cuts, axis=axis)
            for name, p in zip(node.output, parts):
                if name:
                    env[name] = p
            return
        elif op == "Slice":
            out = op_slice(x, inp(1), inp(2), inp(3), inp(4))
        elif op == "Gather":
            axis = at.get("axis", 0)
            out = np.take(x, np.asarray(y).astype(np.int64), axis=axis)
        elif op == "Expand":
            shape = np.broadcast_shapes(x.shape, tuple(int(v) for v in y))
            out = np.broadcast_to(x, shape).copy()
        elif op == "ConstantOfShape":
            v = at.get("value", np.array([0], dtype=np.float32))
            dt = np.float32 if v.dtype == np.float16 else v.dtype
            out = np.full([int(s) for s in x], v.reshape(-1)[0], dtype=dt)
        elif op == "Range":
            # ONNX Range takes its element type from `start`.
            out = np.arange(x.item(), y.item(), inp(2).item(), dtype=x.dtype)
        elif op == "NonZero":
            out = np.array(np.nonzero(x), dtype=np.int64)
        elif op == "CumSum":
            # Keeps the input dtype: the alignment path cumsums int64 durations.
            out = np.cumsum(_f32(x), axis=int(np.asarray(y).item()))
        elif op == "Pad":
            out = op_pad(_f32(x), np.asarray(inp(1)).reshape(-1),
                         at.get("mode", "constant"),
                         0.0 if inp(2) is None else float(inp(2)))
        elif op == "Cast":
            # fp16 casts are elided and computed in fp32. This is what ORT's
            # CPU EP does (it has no fp16 kernels, so it rewrites fp16 regions
            # to fp32), and honouring them literally would overflow: the graph
            # casts int32 quantized accumulators straight to fp16.
            to = ONNX_DTYPE[at["to"]]
            out = _f32(x).astype(np.float32 if to == np.float16 else to)
        elif op == "ScatterND":
            out = op_scatternd(_f32(x), y, _f32(inp(2)))
        elif op == "Resize":
            out = op_resize(x, inp(2) if inp(2) is not None else inp(3),
                            at.get("mode", "nearest"),
                            at.get("coordinate_transformation_mode",
                                   "half_pixel"),
                            at.get("nearest_mode", "round_prefer_floor"))

        # -- reductions / norms ------------------------------------------
        elif op in ("ReduceMean", "ReduceSum"):
            axes = inp(1) if len(node.input) > 1 else at.get("axes")
            keep = bool(at.get("keepdims", 1))
            v = _f32(x)
            if axes is None:
                axes_t = None if not at.get("noop_with_empty_axes", 0) else ()
            else:
                axes_t = tuple(int(a) for a in np.asarray(axes).reshape(-1))
            fn = np.mean if op == "ReduceMean" else np.sum
            out = v if axes_t == () else fn(v, axis=axes_t, keepdims=keep)
        elif op == "Softmax":
            out = op_softmax(x, at.get("axis", -1))
        elif op == "LayerNormalization":
            out = op_layernorm(x, y, inp(2), at.get("axis", -1),
                               at.get("epsilon", 1e-5))
        elif op == "SkipLayerNormalization":
            s = _f32(x) + _f32(y)
            bias = inp(4)
            if bias is not None:
                s = s + _f32(bias)
            out = op_layernorm(s, inp(2), inp(3), -1, at.get("epsilon", 1e-12))
        elif op == "FastGelu":
            out = op_fast_gelu(x, y)

        # -- matmul / conv -----------------------------------------------
        elif op == "MatMul":
            out = _f32(x) @ _f32(y)
        elif op == "FusedMatMul":
            a, b = _f32(x), _f32(y)
            if at.get("transA", 0):
                a = np.swapaxes(a, -1, -2)
            if at.get("transB", 0):
                b = np.swapaxes(b, -1, -2)
            out = (a @ b) * np.float32(at.get("alpha", 1.0))
        elif op == "MatMulInteger":
            out = op_matmul_integer(x, y, inp(2), inp(3))
        elif op == "DynamicQuantizeLinear":
            q, sc, zp = op_dynamic_quantize_linear(x)
            for name, v in zip(node.output, (q, sc, zp)):
                if name:
                    env[name] = v
            return
        elif op == "DequantizeLinear":
            zp = inp(2)
            v = x.astype(np.float32)
            if zp is not None:
                v = v - np.asarray(zp).astype(np.float32)
            out = v * _f32(y).astype(np.float32)
        elif op == "Conv":
            out = op_conv1d(x, y, inp(2), at.get("pads", [0, 0]),
                            at.get("strides", [1])[0],
                            at.get("dilations", [1])[0], at.get("group", 1))
        elif op == "ConvInteger":
            out = op_conv_integer(x, y, inp(2), inp(3), at.get("pads", [0, 0]),
                                  at.get("strides", [1])[0],
                                  at.get("dilations", [1])[0],
                                  at.get("group", 1))
        elif op == "ConvTranspose":
            out = op_conv1d(x, y, inp(2), at.get("pads", [0, 0]),
                            at.get("strides", [1])[0],
                            at.get("dilations", [1])[0], at.get("group", 1),
                            transposed=True,
                            output_padding=at.get("output_padding", [0])[0])
        elif op == "STFT":
            out = op_stft(x, np.asarray(y).item(), inp(2),
                          np.asarray(inp(3)).item(), at.get("onesided", 1))
        elif op == "DynamicQuantizeLSTM":
            ys, yh, yc = op_lstm(
                x, y, inp(2), inp(3), inp(5), inp(6),
                inp(8), inp(9), inp(10), inp(11),
                at["hidden_size"], at.get("direction") == "bidirectional",
                quantize_activations=self.quantized_lstm)
            for name, v in zip(node.output, (ys, yh, yc)):
                if name:
                    env[name] = v
            return
        else:
            raise NotImplementedError(f"{op} ({node.name})")

        env[node.output[0]] = out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--fp16", action="store_true",
                    help="emulate ORT's fp16 node boundaries")
    ap.add_argument("--qlstm", action="store_true",
                    help="emulate ORT's activation-quantizing LSTM")
    args = ap.parse_args()

    ids = sample_tokens(args.tokens)
    style = load_voice(args.voice, len(ids))
    rt = RefRuntime(fp16_boundaries=args.fp16, quantized_lstm=args.qlstm)
    env = rt.run(ids, style, args.speed, strict=args.strict)
    wav = env["waveform"]
    print(f"numpy waveform {wav.shape} min={wav.min():.4f} max={wav.max():.4f}")

    tag = f"{args.voice}_t{args.tokens}_s{args.speed:g}"
    gp = os.path.join(ROOT, "tools", "_ref", f"golden_{tag}.npy")
    if os.path.exists(gp):
        g = np.load(gp)
        if g.shape != wav.shape:
            print(f"SHAPE MISMATCH golden={g.shape} ours={wav.shape}")
            return 1
        err = np.abs(g - wav)
        denom = np.sqrt((g.astype(np.float64) ** 2).mean())
        rmse = np.sqrt(((g - wav).astype(np.float64) ** 2).mean())
        print(f"vs golden: max|err|={err.max():.3e}  rmse={rmse:.3e}  "
              f"snr={20*np.log10(denom/max(rmse,1e-20)):.1f}dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
