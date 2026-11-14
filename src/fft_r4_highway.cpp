// Standalone Highway TU for AoS radix-4 butterflies on power-of-4 lengths.

#include "fft_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fft_r4_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel::detail {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename Cplx>
HWY_INLINE void digit_reverse_4(Cplx* buf, usize n) noexcept {
    const usize digits = ilog2_pow2(n) / 2;
    for (usize i = 0; i < n; ++i) {
        usize j = 0;
        usize x = i;
        for (usize d = 0; d < digits; ++d) {
            j = (j << 2) | (x & 3u);
            x >>= 2;
        }
        if (i < j) {
            keel::swap(buf[i], buf[j]);
        }
    }
}

template <typename T, typename Cplx>
HWY_NOINLINE void radix4_stage(Cplx* buf, usize n, usize len, const Cplx* w_fwd, usize n_half, T* sw1r, T* sw1i,
                               T* sw2r, T* sw2i, T* sw3r, T* sw3i) noexcept {
    const hn::ScalableTag<T> d;
    const usize lanes = hn::Lanes(d);
    const usize m = len >> 2;
    const usize step = n / len;

    for (usize k = 0; k < m; ++k) {
        const Cplx w1 = w_fwd[k * step];
        const Cplx w2 = w_fwd[2 * k * step];
        const usize i3 = 3 * k * step;
        const Cplx w3 = (i3 < n_half) ? w_fwd[i3] : -w_fwd[i3 - n_half];
        sw1r[k] = w1.re;
        sw1i[k] = w1.im;
        sw2r[k] = w2.re;
        sw2i[k] = w2.im;
        sw3r[k] = w3.re;
        sw3i[k] = w3.im;
    }

    for (usize i = 0; i < n; i += len) {
        usize j = 0;
        if (m >= lanes) {
            for (; j + lanes <= m; j += lanes) {
                T* pu = reinterpret_cast<T*>(buf + i + j);
                T* pv = reinterpret_cast<T*>(buf + i + j + m);
                T* pw = reinterpret_cast<T*>(buf + i + j + 2 * m);
                T* pz = reinterpret_cast<T*>(buf + i + j + 3 * m);

                hn::Vec<decltype(d)> ure, uim, vre, vim, wre, wim, zre, zim;
                hn::LoadInterleaved2(d, pu, ure, uim);
                hn::LoadInterleaved2(d, pv, vre, vim);
                hn::LoadInterleaved2(d, pw, wre, wim);
                hn::LoadInterleaved2(d, pz, zre, zim);

                const auto w1r = hn::LoadU(d, sw1r + j);
                const auto w1i = hn::LoadU(d, sw1i + j);
                const auto w2r = hn::LoadU(d, sw2r + j);
                const auto w2i = hn::LoadU(d, sw2i + j);
                const auto w3r = hn::LoadU(d, sw3r + j);
                const auto w3i = hn::LoadU(d, sw3i + j);

                const auto x1re = hn::NegMulAdd(vim, w1i, hn::Mul(vre, w1r));
                const auto x1im = hn::MulAdd(vim, w1r, hn::Mul(vre, w1i));
                const auto x2re = hn::NegMulAdd(wim, w2i, hn::Mul(wre, w2r));
                const auto x2im = hn::MulAdd(wim, w2r, hn::Mul(wre, w2i));
                const auto x3re = hn::NegMulAdd(zim, w3i, hn::Mul(zre, w3r));
                const auto x3im = hn::MulAdd(zim, w3r, hn::Mul(zre, w3i));

                const auto are = hn::Add(ure, x2re);
                const auto aim = hn::Add(uim, x2im);
                const auto bre = hn::Sub(ure, x2re);
                const auto bim = hn::Sub(uim, x2im);
                const auto cre = hn::Add(x1re, x3re);
                const auto cim = hn::Add(x1im, x3im);
                const auto dre = hn::Sub(x1re, x3re);
                const auto dim = hn::Sub(x1im, x3im);

                hn::StoreInterleaved2(hn::Add(are, cre), hn::Add(aim, cim), d, pu);
                hn::StoreInterleaved2(hn::Add(bre, dim), hn::Sub(bim, dre), d, pv);
                hn::StoreInterleaved2(hn::Sub(are, cre), hn::Sub(aim, cim), d, pw);
                hn::StoreInterleaved2(hn::Sub(bre, dim), hn::Add(bim, dre), d, pz);
            }
        }
        for (; j < m; ++j) {
            const Cplx x0 = buf[i + j];
            const Cplx x1 = buf[i + j + m] * Cplx(sw1r[j], sw1i[j]);
            const Cplx x2 = buf[i + j + 2 * m] * Cplx(sw2r[j], sw2i[j]);
            const Cplx x3 = buf[i + j + 3 * m] * Cplx(sw3r[j], sw3i[j]);
            const Cplx a = x0 + x2;
            const Cplx b = x0 - x2;
            const Cplx c = x1 + x3;
            const Cplx dc = x1 - x3;
            const Cplx id(-dc.im, dc.re);
            buf[i + j] = a + c;
            buf[i + j + m] = b - id;
            buf[i + j + 2 * m] = a - c;
            buf[i + j + 3 * m] = b + id;
        }
    }
}

template <typename T, typename Cplx>
void radix4_pow4_body(Cplx* buf, usize n, const Cplx* w_fwd, T* sw_base, usize stride) noexcept {
    if (n <= 1) return;
    digit_reverse_4(buf, n);

    T* sw1r = sw_base;
    T* sw1i = sw_base + stride;
    T* sw2r = sw_base + 2 * stride;
    T* sw2i = sw_base + 3 * stride;
    T* sw3r = sw_base + 4 * stride;
    T* sw3i = sw_base + 5 * stride;

    const usize n_half = n >> 1;
    for (usize len = 4; len <= n; len <<= 2) {
        radix4_stage<T, Cplx>(buf, n, len, w_fwd, n_half, sw1r, sw1i, sw2r, sw2i, sw3r, sw3i);
    }
}

void Radix4Pow4F64(Complex64* buf, usize n, const Complex64* w_fwd, f64* sw_base, usize stride) {
    radix4_pow4_body<f64, Complex64>(buf, n, w_fwd, sw_base, stride);
}

void Radix4Pow4F32(Complex32* buf, usize n, const Complex32* w_fwd, f32* sw_base, usize stride) {
    radix4_pow4_body<f32, Complex32>(buf, n, w_fwd, sw_base, stride);
}

} // namespace HWY_NAMESPACE
} // namespace keel::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace keel::detail {

HWY_EXPORT(Radix4Pow4F64);
HWY_EXPORT(Radix4Pow4F32);

void hwy_radix4_pow4_inplace(Complex64* buf, usize n, const Complex64* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    const usize stride = (n >> 2) + 8;
    f64* sw_base = scratch.alloc<f64>(6 * stride);
    HWY_DYNAMIC_DISPATCH(Radix4Pow4F64)(buf, n, w_fwd, sw_base, stride);
}

void hwy_radix4_pow4_inplace(Complex32* buf, usize n, const Complex32* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    const usize stride = (n >> 2) + 16;
    f32* sw_base = scratch.alloc<f32>(6 * stride);
    HWY_DYNAMIC_DISPATCH(Radix4Pow4F32)(buf, n, w_fwd, sw_base, stride);
}

} // namespace keel::detail

#endif // HWY_ONCE
