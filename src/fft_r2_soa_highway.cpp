// Standalone Highway TU for Structure-of-Arrays (split real/imag) radix-2 butterflies.
// Eliminates cross-lane LoadInterleaved2 / StoreInterleaved2 permutes on wide SIMD targets.

#include "fft_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fft_r2_soa_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel::detail {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_INLINE void bit_reverse_2_soa(T* re, T* im, usize n) noexcept {
    for (usize i = 1, j = 0; i < n; ++i) {
        usize bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            keel::swap(re[i], re[j]);
            keel::swap(im[i], im[j]);
        }
    }
}

template <typename T, typename Cplx>
void radix2_soa_stages_body(T* re, T* im, usize n, const Cplx* w_fwd, T* stage_wr, T* stage_wi) noexcept {
    if (n <= 1) return;
    bit_reverse_2_soa(re, im, n);

    const hn::ScalableTag<T> d;
    const usize lanes = hn::Lanes(d);

    for (usize len = 2; len <= n; len <<= 1) {
        const usize half = len >> 1;
        const usize step = n / len;
        for (usize k = 0; k < half; ++k) {
            stage_wr[k] = w_fwd[k * step].re;
            stage_wi[k] = w_fwd[k * step].im;
        }
        for (usize i = 0; i < n; i += len) {
            usize j = 0;
            if (half >= lanes) {
                for (; j + lanes <= half; j += lanes) {
                    const auto wre = hn::LoadU(d, stage_wr + j);
                    const auto wim = hn::LoadU(d, stage_wi + j);

                    const auto u_re = hn::LoadU(d, re + i + j);
                    const auto u_im = hn::LoadU(d, im + i + j);
                    const auto v_re = hn::LoadU(d, re + i + j + half);
                    const auto v_im = hn::LoadU(d, im + i + j + half);

                    const auto bv_re = hn::NegMulAdd(v_im, wim, hn::Mul(v_re, wre));
                    const auto bv_im = hn::MulAdd(v_im, wre, hn::Mul(v_re, wim));

                    hn::StoreU(hn::Add(u_re, bv_re), d, re + i + j);
                    hn::StoreU(hn::Add(u_im, bv_im), d, im + i + j);
                    hn::StoreU(hn::Sub(u_re, bv_re), d, re + i + j + half);
                    hn::StoreU(hn::Sub(u_im, bv_im), d, im + i + j + half);
                }
            }
            for (; j < half; ++j) {
                const T u_re = re[i + j];
                const T u_im = im[i + j];
                const T v_re = re[i + j + half];
                const T v_im = im[i + j + half];
                const T bv_re = v_re * stage_wr[j] - v_im * stage_wi[j];
                const T bv_im = v_re * stage_wi[j] + v_im * stage_wr[j];
                re[i + j] = u_re + bv_re;
                im[i + j] = u_im + bv_im;
                re[i + j + half] = u_re - bv_re;
                im[i + j + half] = u_im - bv_im;
            }
        }
    }
}

template <typename T, typename Cplx>
void radix2_soa_aos_body(Cplx* buf, usize n, const Cplx* w_fwd, T* re, T* im, T* stage_wr, T* stage_wi) noexcept {
    if (n <= 1) return;
    for (usize i = 0; i < n; ++i) {
        re[i] = buf[i].re;
        im[i] = buf[i].im;
    }
    radix2_soa_stages_body<T, Cplx>(re, im, n, w_fwd, stage_wr, stage_wi);
    for (usize i = 0; i < n; ++i) {
        buf[i] = Cplx(re[i], im[i]);
    }
}

void Radix2SoAStagesF64(f64* re, f64* im, usize n, const c64* w_fwd, f64* stage_wr, f64* stage_wi) {
    radix2_soa_stages_body<f64, c64>(re, im, n, w_fwd, stage_wr, stage_wi);
}

void Radix2SoAF64(c64* buf, usize n, const c64* w_fwd, f64* re, f64* im, f64* stage_wr,
                  f64* stage_wi) {
    radix2_soa_aos_body<f64, c64>(buf, n, w_fwd, re, im, stage_wr, stage_wi);
}

void Radix2SoAStagesF32(f32* re, f32* im, usize n, const c32* w_fwd, f32* stage_wr, f32* stage_wi) {
    radix2_soa_stages_body<f32, c32>(re, im, n, w_fwd, stage_wr, stage_wi);
}

void Radix2SoAF32(c32* buf, usize n, const c32* w_fwd, f32* re, f32* im, f32* stage_wr,
                  f32* stage_wi) {
    radix2_soa_aos_body<f32, c32>(buf, n, w_fwd, re, im, stage_wr, stage_wi);
}

} // namespace HWY_NAMESPACE
} // namespace keel::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace keel::detail {

HWY_EXPORT(Radix2SoAStagesF64);
HWY_EXPORT(Radix2SoAF64);
HWY_EXPORT(Radix2SoAStagesF32);
HWY_EXPORT(Radix2SoAF32);

void hwy_radix2_soa_stages_inplace(f64* re, f64* im, usize n, const c64* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f64* stage_wr = scratch.alloc<f64>(n >> 1);
    f64* stage_wi = scratch.alloc<f64>(n >> 1);
    HWY_DYNAMIC_DISPATCH(Radix2SoAStagesF64)(re, im, n, w_fwd, stage_wr, stage_wi);
}

void hwy_radix2_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f64* re = scratch.alloc<f64>(n);
    f64* im = scratch.alloc<f64>(n);
    f64* stage_wr = scratch.alloc<f64>(n >> 1);
    f64* stage_wi = scratch.alloc<f64>(n >> 1);
    HWY_DYNAMIC_DISPATCH(Radix2SoAF64)(buf, n, w_fwd, re, im, stage_wr, stage_wi);
}

void hwy_radix2_soa_stages_inplace(f32* re, f32* im, usize n, const c32* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f32* stage_wr = scratch.alloc<f32>(n >> 1);
    f32* stage_wi = scratch.alloc<f32>(n >> 1);
    HWY_DYNAMIC_DISPATCH(Radix2SoAStagesF32)(re, im, n, w_fwd, stage_wr, stage_wi);
}

void hwy_radix2_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f32* re = scratch.alloc<f32>(n);
    f32* im = scratch.alloc<f32>(n);
    f32* stage_wr = scratch.alloc<f32>(n >> 1);
    f32* stage_wi = scratch.alloc<f32>(n >> 1);
    HWY_DYNAMIC_DISPATCH(Radix2SoAF32)(buf, n, w_fwd, re, im, stage_wr, stage_wi);
}

} // namespace keel::detail

#endif // HWY_ONCE
