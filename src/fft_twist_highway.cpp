// Standalone Highway TU for packed half-length RFFT / IRFFT SIMD twist kernels.

#include "fft_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fft_twist_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel::detail {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T, typename Cplx, bool WriteConjugateTail>
void rfft_twist_body(const T* z_re, const T* z_im, usize n, const Cplx* w_full, T scale, Cplx* dst) noexcept {
    const usize half = n >> 1;
    const T z0_re = z_re[0];
    const T z0_im = z_im[0];
    dst[0] = Cplx((z0_re + z0_im) * scale, T(0));
    dst[half] = Cplx((z0_re - z0_im) * scale, T(0));

    const hn::ScalableTag<T> d;
    const usize lanes = hn::Lanes(d);
    const T half_scale = T(0.5) * scale;
    const auto v_half_scale = hn::Set(d, half_scale);

    usize k = 1;
    if (half >= 1 + lanes) {
        for (; k + lanes <= half; k += lanes) {
            const auto zk_re = hn::LoadU(d, z_re + k);
            const auto zk_im = hn::LoadU(d, z_im + k);

            const auto zj_re_raw = hn::LoadU(d, z_re + (half - k - lanes + 1));
            const auto zj_im_raw = hn::LoadU(d, z_im + (half - k - lanes + 1));
            const auto zj_re = hn::Reverse(d, zj_re_raw);
            const auto zj_im_neg = hn::Neg(hn::Reverse(d, zj_im_raw));

            const auto e_re = hn::Mul(v_half_scale, hn::Add(zk_re, zj_re));
            const auto e_im = hn::Mul(v_half_scale, hn::Add(zk_im, zj_im_neg));
            const auto d_re = hn::Mul(v_half_scale, hn::Sub(zk_re, zj_re));
            const auto d_im = hn::Mul(v_half_scale, hn::Sub(zk_im, zj_im_neg));

            const auto o_re = d_im;
            const auto o_im = hn::Neg(d_re);

            hn::Vec<decltype(d)> wk_re, wk_im;
            hn::LoadInterleaved2(d, reinterpret_cast<const T*>(w_full + k), wk_re, wk_im);

            const auto wo_re = hn::NegMulAdd(wk_im, o_im, hn::Mul(wk_re, o_re));
            const auto wo_im = hn::MulAdd(wk_im, o_re, hn::Mul(wk_re, o_im));

            const auto xk_re = hn::Add(e_re, wo_re);
            const auto xk_im = hn::Add(e_im, wo_im);

            hn::StoreInterleaved2(xk_re, xk_im, d, reinterpret_cast<T*>(dst + k));

            if constexpr (WriteConjugateTail) {
                const auto xk_re_rev = hn::Reverse(d, xk_re);
                const auto xk_im_rev_conj = hn::Neg(hn::Reverse(d, xk_im));
                hn::StoreInterleaved2(xk_re_rev, xk_im_rev_conj, d, reinterpret_cast<T*>(dst + (n - k - lanes + 1)));
            }
        }
    }

    for (; k < half; ++k) {
        const T zk_re = z_re[k];
        const T zk_im = z_im[k];
        const T zj_re = z_re[half - k];
        const T zj_im = -z_im[half - k];

        const T e_re = half_scale * (zk_re + zj_re);
        const T e_im = half_scale * (zk_im + zj_im);
        const T d_re = half_scale * (zk_re - zj_re);
        const T d_im = half_scale * (zk_im - zj_im);

        const T o_re = d_im;
        const T o_im = -d_re;

        const T wk_re = w_full[k].re;
        const T wk_im = w_full[k].im;
        const T wo_re = wk_re * o_re - wk_im * o_im;
        const T wo_im = wk_re * o_im + wk_im * o_re;

        const Cplx xk(e_re + wo_re, e_im + wo_im);
        dst[k] = xk;
        if constexpr (WriteConjugateTail) {
            dst[n - k] = conj(xk);
        }
    }
}

template <typename T, typename Cplx>
void irfft_untwist_conj_body(const Cplx* half_src, usize n, const Cplx* w_full, T* z_re, T* z_im) noexcept {
    const usize half = n >> 1;
    const T x0 = half_src[0].re;
    const T xh = half_src[half].re;
    z_re[0] = T(0.5) * (x0 + xh);
    z_im[0] = -T(0.5) * (x0 - xh);

    const hn::ScalableTag<T> d;
    const usize lanes = hn::Lanes(d);
    const auto v_half = hn::Set(d, T(0.5));

    usize k = 1;
    if (half >= 1 + lanes) {
        for (; k + lanes <= half; k += lanes) {
            hn::Vec<decltype(d)> xk_re, xk_im;
            hn::LoadInterleaved2(d, reinterpret_cast<const T*>(half_src + k), xk_re, xk_im);

            hn::Vec<decltype(d)> xj_re_raw, xj_im_raw;
            hn::LoadInterleaved2(d, reinterpret_cast<const T*>(half_src + (half - k - lanes + 1)), xj_re_raw,
                                 xj_im_raw);
            const auto xj_re = hn::Reverse(d, xj_re_raw);
            const auto xj_im_conj = hn::Neg(hn::Reverse(d, xj_im_raw));

            const auto e_re = hn::Mul(v_half, hn::Add(xk_re, xj_re));
            const auto e_im = hn::Mul(v_half, hn::Add(xk_im, xj_im_conj));
            const auto d_re = hn::Mul(v_half, hn::Sub(xk_re, xj_re));
            const auto d_im = hn::Mul(v_half, hn::Sub(xk_im, xj_im_conj));

            hn::Vec<decltype(d)> w_re, w_im;
            hn::LoadInterleaved2(d, reinterpret_cast<const T*>(w_full + k), w_re, w_im);

            const auto io_re = hn::NegMulAdd(w_re, d_im, hn::Mul(w_im, d_re));
            const auto io_im = hn::MulAdd(w_re, d_re, hn::Mul(w_im, d_im));

            hn::StoreU(hn::Add(e_re, io_re), d, z_re + k);
            hn::StoreU(hn::Neg(hn::Add(e_im, io_im)), d, z_im + k);
        }
    }

    for (; k < half; ++k) {
        const Cplx xk = half_src[k];
        const Cplx xj = half_src[half - k];
        const T xj_re = xj.re;
        const T xj_im_conj = -xj.im;

        const T e_re = T(0.5) * (xk.re + xj_re);
        const T e_im = T(0.5) * (xk.im + xj_im_conj);
        const T d_re = T(0.5) * (xk.re - xj_re);
        const T d_im = T(0.5) * (xk.im - xj_im_conj);

        const T w_re = w_full[k].re;
        const T w_im = w_full[k].im;
        const T io_re = w_im * d_re - w_re * d_im;
        const T io_im = w_re * d_re + w_im * d_im;

        z_re[k] = e_re + io_re;
        z_im[k] = -(e_im + io_im);
    }
}

template <typename T>
void irfft_store_interleaved_body(const T* z_re, const T* z_im, usize half, T post_scale, T* real_dst) noexcept {
    const hn::ScalableTag<T> d;
    const usize lanes = hn::Lanes(d);
    const auto v_pos = hn::Set(d, post_scale);
    const auto v_neg = hn::Set(d, -post_scale);

    usize m = 0;
    if (half >= lanes) {
        for (; m + lanes <= half; m += lanes) {
            const auto even_v = hn::Mul(hn::LoadU(d, z_re + m), v_pos);
            const auto odd_v = hn::Mul(hn::LoadU(d, z_im + m), v_neg);
            hn::StoreInterleaved2(even_v, odd_v, d, real_dst + 2 * m);
        }
    }
    const T neg_scale = -post_scale;
    for (; m < half; ++m) {
        real_dst[2 * m] = z_re[m] * post_scale;
        real_dst[2 * m + 1] = z_im[m] * neg_scale;
    }
}

void RfftTwistFullF64(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale, c64* dst) {
    rfft_twist_body<f64, c64, true>(z_re, z_im, n, w_full, scale, dst);
}

void RfftTwistFullF32(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale, c32* dst) {
    rfft_twist_body<f32, c32, true>(z_re, z_im, n, w_full, scale, dst);
}

void RfftTwistHalfF64(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale,
                      c64* half_dst) {
    rfft_twist_body<f64, c64, false>(z_re, z_im, n, w_full, scale, half_dst);
}

void RfftTwistHalfF32(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale,
                      c32* half_dst) {
    rfft_twist_body<f32, c32, false>(z_re, z_im, n, w_full, scale, half_dst);
}

void IrfftUntwistConjF64(const c64* half_src, usize n, const c64* w_full, f64* z_re, f64* z_im) {
    irfft_untwist_conj_body<f64, c64>(half_src, n, w_full, z_re, z_im);
}

void IrfftUntwistConjF32(const c32* half_src, usize n, const c32* w_full, f32* z_re, f32* z_im) {
    irfft_untwist_conj_body<f32, c32>(half_src, n, w_full, z_re, z_im);
}

void IrfftStoreInterleavedF64(const f64* z_re, const f64* z_im, usize half, f64 post_scale, f64* real_dst) {
    irfft_store_interleaved_body<f64>(z_re, z_im, half, post_scale, real_dst);
}

void IrfftStoreInterleavedF32(const f32* z_re, const f32* z_im, usize half, f32 post_scale, f32* real_dst) {
    irfft_store_interleaved_body<f32>(z_re, z_im, half, post_scale, real_dst);
}

} // namespace HWY_NAMESPACE
} // namespace keel::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace keel::detail {

HWY_EXPORT(RfftTwistFullF64);
HWY_EXPORT(RfftTwistFullF32);
HWY_EXPORT(RfftTwistHalfF64);
HWY_EXPORT(RfftTwistHalfF32);
HWY_EXPORT(IrfftUntwistConjF64);
HWY_EXPORT(IrfftUntwistConjF32);
HWY_EXPORT(IrfftStoreInterleavedF64);
HWY_EXPORT(IrfftStoreInterleavedF32);

void rfft_twist_full_unit_stride(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale,
                                 c64* dst) noexcept {
    HWY_DYNAMIC_DISPATCH(RfftTwistFullF64)(z_re, z_im, n, w_full, scale, dst);
}

void rfft_twist_full_unit_stride(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale,
                                 c32* dst) noexcept {
    HWY_DYNAMIC_DISPATCH(RfftTwistFullF32)(z_re, z_im, n, w_full, scale, dst);
}

void rfft_twist_half_unit_stride(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale,
                                 c64* half_dst) noexcept {
    HWY_DYNAMIC_DISPATCH(RfftTwistHalfF64)(z_re, z_im, n, w_full, scale, half_dst);
}

void rfft_twist_half_unit_stride(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale,
                                 c32* half_dst) noexcept {
    HWY_DYNAMIC_DISPATCH(RfftTwistHalfF32)(z_re, z_im, n, w_full, scale, half_dst);
}

void irfft_untwist_conj_unit_stride(const c64* half_src, usize n, const c64* w_full, f64* z_re,
                                    f64* z_im) noexcept {
    HWY_DYNAMIC_DISPATCH(IrfftUntwistConjF64)(half_src, n, w_full, z_re, z_im);
}

void irfft_untwist_conj_unit_stride(const c32* half_src, usize n, const c32* w_full, f32* z_re,
                                    f32* z_im) noexcept {
    HWY_DYNAMIC_DISPATCH(IrfftUntwistConjF32)(half_src, n, w_full, z_re, z_im);
}

void irfft_store_interleaved_unit_stride(const f64* z_re, const f64* z_im, usize half, f64 post_scale,
                                         f64* real_dst) noexcept {
    HWY_DYNAMIC_DISPATCH(IrfftStoreInterleavedF64)(z_re, z_im, half, post_scale, real_dst);
}

void irfft_store_interleaved_unit_stride(const f32* z_re, const f32* z_im, usize half, f32 post_scale,
                                         f32* real_dst) noexcept {
    HWY_DYNAMIC_DISPATCH(IrfftStoreInterleavedF32)(z_re, z_im, half, post_scale, real_dst);
}

} // namespace keel::detail

#endif // HWY_ONCE
