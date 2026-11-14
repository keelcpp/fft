// Standalone Highway TU for Structure-of-Arrays (split real/imag) radix-4 butterflies.

#include "fft_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fft_r4_soa_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel::detail {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_INLINE void digit_reverse_4_soa(T* re, T* im, usize n) noexcept {
    const usize digits = ilog2_pow2(n) / 2;
    for (usize i = 0; i < n; ++i) {
        usize j = 0;
        usize x = i;
        for (usize d = 0; d < digits; ++d) {
            j = (j << 2) | (x & 3u);
            x >>= 2;
        }
        if (i < j) {
            keel::swap(re[i], re[j]);
            keel::swap(im[i], im[j]);
        }
    }
}

template <typename T, typename Cplx>
HWY_NOINLINE void radix4_stage_soa(T* re, T* im, usize n, usize len, const Cplx* w_fwd, usize n_half, T* sw1r,
                                   T* sw1i, T* sw2r, T* sw2i, T* sw3r, T* sw3i) noexcept {
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
                const auto u_re = hn::LoadU(d, re + i + j);
                const auto u_im = hn::LoadU(d, im + i + j);
                const auto v_re = hn::LoadU(d, re + i + j + m);
                const auto v_im = hn::LoadU(d, im + i + j + m);
                const auto w_re_in = hn::LoadU(d, re + i + j + 2 * m);
                const auto w_im_in = hn::LoadU(d, im + i + j + 2 * m);
                const auto z_re = hn::LoadU(d, re + i + j + 3 * m);
                const auto z_im = hn::LoadU(d, im + i + j + 3 * m);

                const auto w1r = hn::LoadU(d, sw1r + j);
                const auto w1i = hn::LoadU(d, sw1i + j);
                const auto w2r = hn::LoadU(d, sw2r + j);
                const auto w2i = hn::LoadU(d, sw2i + j);
                const auto w3r = hn::LoadU(d, sw3r + j);
                const auto w3i = hn::LoadU(d, sw3i + j);

                const auto x1re = hn::NegMulAdd(v_im, w1i, hn::Mul(v_re, w1r));
                const auto x1im = hn::MulAdd(v_im, w1r, hn::Mul(v_re, w1i));
                const auto x2re = hn::NegMulAdd(w_im_in, w2i, hn::Mul(w_re_in, w2r));
                const auto x2im = hn::MulAdd(w_im_in, w2r, hn::Mul(w_re_in, w2i));
                const auto x3re = hn::NegMulAdd(z_im, w3i, hn::Mul(z_re, w3r));
                const auto x3im = hn::MulAdd(z_im, w3r, hn::Mul(z_re, w3i));

                const auto are = hn::Add(u_re, x2re);
                const auto aim = hn::Add(u_im, x2im);
                const auto bre = hn::Sub(u_re, x2re);
                const auto bim = hn::Sub(u_im, x2im);
                const auto cre = hn::Add(x1re, x3re);
                const auto cim = hn::Add(x1im, x3im);
                const auto dre = hn::Sub(x1re, x3re);
                const auto dim = hn::Sub(x1im, x3im);

                hn::StoreU(hn::Add(are, cre), d, re + i + j);
                hn::StoreU(hn::Add(aim, cim), d, im + i + j);
                hn::StoreU(hn::Add(bre, dim), d, re + i + j + m);
                hn::StoreU(hn::Sub(bim, dre), d, im + i + j + m);
                hn::StoreU(hn::Sub(are, cre), d, re + i + j + 2 * m);
                hn::StoreU(hn::Sub(aim, cim), d, im + i + j + 2 * m);
                hn::StoreU(hn::Sub(bre, dim), d, re + i + j + 3 * m);
                hn::StoreU(hn::Add(bim, dre), d, im + i + j + 3 * m);
            }
        }
        for (; j < m; ++j) {
            const T u_re = re[i + j];
            const T u_im = im[i + j];
            const T v_re = re[i + j + m];
            const T v_im = im[i + j + m];
            const T w_re = re[i + j + 2 * m];
            const T w_im = im[i + j + 2 * m];
            const T z_re = re[i + j + 3 * m];
            const T z_im = im[i + j + 3 * m];

            const T x1re = v_re * sw1r[j] - v_im * sw1i[j];
            const T x1im = v_re * sw1i[j] + v_im * sw1r[j];
            const T x2re = w_re * sw2r[j] - w_im * sw2i[j];
            const T x2im = w_re * sw2i[j] + w_im * sw2r[j];
            const T x3re = z_re * sw3r[j] - z_im * sw3i[j];
            const T x3im = z_re * sw3i[j] + z_im * sw3r[j];

            const T are = u_re + x2re;
            const T aim = u_im + x2im;
            const T bre = u_re - x2re;
            const T bim = u_im - x2im;
            const T cre = x1re + x3re;
            const T cim = x1im + x3im;
            const T dre = x1re - x3re;
            const T dim = x1im - x3im;

            re[i + j] = are + cre;
            im[i + j] = aim + cim;
            re[i + j + m] = bre + dim;
            im[i + j + m] = bim - dre;
            re[i + j + 2 * m] = are - cre;
            im[i + j + 2 * m] = aim - cim;
            re[i + j + 3 * m] = bre - dim;
            im[i + j + 3 * m] = bim + dre;
        }
    }
}

template <typename T, typename Cplx>
void radix4_pow4_soa_stages_body(T* re, T* im, usize n, const Cplx* w_fwd, T* sw_base, usize stride) noexcept {
    if (n <= 1) return;
    digit_reverse_4_soa(re, im, n);

    T* sw1r = sw_base;
    T* sw1i = sw_base + stride;
    T* sw2r = sw_base + 2 * stride;
    T* sw2i = sw_base + 3 * stride;
    T* sw3r = sw_base + 4 * stride;
    T* sw3i = sw_base + 5 * stride;

    const usize n_half = n >> 1;
    for (usize len = 4; len <= n; len <<= 2) {
        radix4_stage_soa<T, Cplx>(re, im, n, len, w_fwd, n_half, sw1r, sw1i, sw2r, sw2i, sw3r, sw3i);
    }
}

template <typename T, typename Cplx>
void radix4_pow4_soa_aos_body(Cplx* buf, usize n, const Cplx* w_fwd, T* re, T* im, T* sw_base, usize stride) noexcept {
    if (n <= 1) return;
    for (usize i = 0; i < n; ++i) {
        re[i] = buf[i].re;
        im[i] = buf[i].im;
    }
    radix4_pow4_soa_stages_body<T, Cplx>(re, im, n, w_fwd, sw_base, stride);
    for (usize i = 0; i < n; ++i) {
        buf[i] = Cplx(re[i], im[i]);
    }
}

void Radix4Pow4SoAStagesF64(f64* re, f64* im, usize n, const c64* w_fwd, f64* sw_base, usize stride) {
    radix4_pow4_soa_stages_body<f64, c64>(re, im, n, w_fwd, sw_base, stride);
}

void Radix4Pow4SoAF64(c64* buf, usize n, const c64* w_fwd, f64* re, f64* im, f64* sw_base,
                      usize stride) {
    radix4_pow4_soa_aos_body<f64, c64>(buf, n, w_fwd, re, im, sw_base, stride);
}

void Radix4Pow4SoAStagesF32(f32* re, f32* im, usize n, const c32* w_fwd, f32* sw_base, usize stride) {
    radix4_pow4_soa_stages_body<f32, c32>(re, im, n, w_fwd, sw_base, stride);
}

void Radix4Pow4SoAF32(c32* buf, usize n, const c32* w_fwd, f32* re, f32* im, f32* sw_base,
                      usize stride) {
    radix4_pow4_soa_aos_body<f32, c32>(buf, n, w_fwd, re, im, sw_base, stride);
}

} // namespace HWY_NAMESPACE
} // namespace keel::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace keel::detail {

HWY_EXPORT(Radix4Pow4SoAStagesF64);
HWY_EXPORT(Radix4Pow4SoAF64);
HWY_EXPORT(Radix4Pow4SoAStagesF32);
HWY_EXPORT(Radix4Pow4SoAF32);

void hwy_radix4_pow4_soa_stages_inplace(f64* re, f64* im, usize n, const c64* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    const usize stride = (n >> 2) + 8;
    f64* sw_base = scratch.alloc<f64>(6 * stride);
    HWY_DYNAMIC_DISPATCH(Radix4Pow4SoAStagesF64)(re, im, n, w_fwd, sw_base, stride);
}

void hwy_radix4_pow4_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f64* re = scratch.alloc<f64>(n);
    f64* im = scratch.alloc<f64>(n);
    const usize stride = (n >> 2) + 8;
    f64* sw_base = scratch.alloc<f64>(6 * stride);
    HWY_DYNAMIC_DISPATCH(Radix4Pow4SoAF64)(buf, n, w_fwd, re, im, sw_base, stride);
}

void hwy_radix4_pow4_soa_stages_inplace(f32* re, f32* im, usize n, const c32* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    const usize stride = (n >> 2) + 16;
    f32* sw_base = scratch.alloc<f32>(6 * stride);
    HWY_DYNAMIC_DISPATCH(Radix4Pow4SoAStagesF32)(re, im, n, w_fwd, sw_base, stride);
}

void hwy_radix4_pow4_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f32* re = scratch.alloc<f32>(n);
    f32* im = scratch.alloc<f32>(n);
    const usize stride = (n >> 2) + 16;
    f32* sw_base = scratch.alloc<f32>(6 * stride);
    HWY_DYNAMIC_DISPATCH(Radix4Pow4SoAF32)(buf, n, w_fwd, re, im, sw_base, stride);
}

} // namespace keel::detail

#endif // HWY_ONCE
