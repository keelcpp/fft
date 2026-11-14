// Standalone Highway TU for AoS iterative radix-2 Cooley-Tukey butterflies.
// Kept isolated in its own translation unit so per-TU compiler inlining budgets
// are not shared across distinct radix butterflies.

#include "fft_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fft_r2_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel::detail {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename Cplx>
HWY_INLINE void bit_reverse_2(Cplx* buf, usize n) noexcept {
    for (usize i = 1, j = 0; i < n; ++i) {
        usize bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            keel::swap(buf[i], buf[j]);
        }
    }
}

template <typename T, typename Cplx>
void radix2_body(Cplx* buf, usize n, const Cplx* w_fwd, T* stage_wr, T* stage_wi) noexcept {
    if (n <= 1) return;
    bit_reverse_2(buf, n);

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
                    T* pu = reinterpret_cast<T*>(buf + i + j);
                    T* pv = reinterpret_cast<T*>(buf + i + j + half);
                    hn::Vec<decltype(d)> ure, uim, vlre, vlim;
                    hn::LoadInterleaved2(d, pu, ure, uim);
                    hn::LoadInterleaved2(d, pv, vlre, vlim);
                    const auto vre = hn::NegMulAdd(vlim, wim, hn::Mul(vlre, wre));
                    const auto vim = hn::MulAdd(vlim, wre, hn::Mul(vlre, wim));
                    hn::StoreInterleaved2(hn::Add(ure, vre), hn::Add(uim, vim), d, pu);
                    hn::StoreInterleaved2(hn::Sub(ure, vre), hn::Sub(uim, vim), d, pv);
                }
            }
            for (; j < half; ++j) {
                const Cplx u = buf[i + j];
                const Cplx v = buf[i + j + half] * Cplx(stage_wr[j], stage_wi[j]);
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
            }
        }
    }
}

void Radix2F64(c64* buf, usize n, const c64* w_fwd, f64* stage_wr, f64* stage_wi) {
    radix2_body<f64, c64>(buf, n, w_fwd, stage_wr, stage_wi);
}

void Radix2F32(c32* buf, usize n, const c32* w_fwd, f32* stage_wr, f32* stage_wi) {
    radix2_body<f32, c32>(buf, n, w_fwd, stage_wr, stage_wi);
}

} // namespace HWY_NAMESPACE
} // namespace keel::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace keel::detail {

HWY_EXPORT(Radix2F64);
HWY_EXPORT(Radix2F32);

void hwy_radix2_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f64* stage_wr = scratch.alloc<f64>(n >> 1);
    f64* stage_wi = scratch.alloc<f64>(n >> 1);
    HWY_DYNAMIC_DISPATCH(Radix2F64)(buf, n, w_fwd, stage_wr, stage_wi);
}

void hwy_radix2_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    f32* stage_wr = scratch.alloc<f32>(n >> 1);
    f32* stage_wi = scratch.alloc<f32>(n >> 1);
    HWY_DYNAMIC_DISPATCH(Radix2F32)(buf, n, w_fwd, stage_wr, stage_wi);
}

} // namespace keel::detail

#endif // HWY_ONCE
