// Standalone Highway TU for Stockham auto-sort out-of-place radix-2 butterflies.

#include "fft_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "fft_stockham_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace keel::fft::detail {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T, typename Cplx>
void stockham_body(Cplx* x, Cplx* work, usize n, const Cplx* w_fwd, T* stage_wr, T* stage_wi) noexcept {
    if (n <= 1) return;

    const hn::ScalableTag<T> d;
    const usize lanes = hn::Lanes(d);

    Cplx* src = x;
    Cplx* dst = work;
    const usize stages = ilog2_pow2(n);

    for (usize stage = 0; stage < stages; ++stage) {
        const usize m = usize{1} << (stage + 1);
        const usize mh = m >> 1;
        const usize n_sub = n / m;
        const usize step = n_sub;

        for (usize k = 0; k < mh; ++k) {
            stage_wr[k] = w_fwd[k * step].re;
            stage_wi[k] = w_fwd[k * step].im;
        }

        for (usize pair_idx = 0; pair_idx < n_sub; ++pair_idx) {
            Cplx* a_in = src + pair_idx * mh;
            Cplx* b_in = src + (pair_idx + n_sub) * mh;
            Cplx* a_out = dst + pair_idx * m;
            Cplx* b_out = a_out + mh;

            usize k = 0;
            if (mh >= lanes) {
                for (; k + lanes <= mh; k += lanes) {
                    const auto wre = hn::LoadU(d, stage_wr + k);
                    const auto wim = hn::LoadU(d, stage_wi + k);

                    T* pa = reinterpret_cast<T*>(a_in + k);
                    T* pb = reinterpret_cast<T*>(b_in + k);
                    hn::Vec<decltype(d)> are, aim, bre, bim;
                    hn::LoadInterleaved2(d, pa, are, aim);
                    hn::LoadInterleaved2(d, pb, bre, bim);

                    const auto bvre = hn::NegMulAdd(bim, wim, hn::Mul(bre, wre));
                    const auto bvim = hn::MulAdd(bim, wre, hn::Mul(bre, wim));

                    T* pa_out = reinterpret_cast<T*>(a_out + k);
                    T* pb_out = reinterpret_cast<T*>(b_out + k);
                    hn::StoreInterleaved2(hn::Add(are, bvre), hn::Add(aim, bvim), d, pa_out);
                    hn::StoreInterleaved2(hn::Sub(are, bvre), hn::Sub(aim, bvim), d, pb_out);
                }
            }
            for (; k < mh; ++k) {
                const Cplx a = a_in[k];
                const Cplx b = b_in[k] * Cplx(stage_wr[k], stage_wi[k]);
                a_out[k] = a + b;
                b_out[k] = a - b;
            }
        }

        keel::swap(src, dst);
    }

    if (src != x) {
        mem_copy(x, src, n * sizeof(Cplx));
    }
}

void StockhamF64(c64* x, c64* work, usize n, const c64* w_fwd, f64* stage_wr, f64* stage_wi) {
    stockham_body<f64, c64>(x, work, n, w_fwd, stage_wr, stage_wi);
}

void StockhamF32(c32* x, c32* work, usize n, const c32* w_fwd, f32* stage_wr, f32* stage_wi) {
    stockham_body<f32, c32>(x, work, n, w_fwd, stage_wr, stage_wi);
}

} // namespace HWY_NAMESPACE
} // namespace keel::fft::detail
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace keel::fft::detail {

HWY_EXPORT(StockhamF64);
HWY_EXPORT(StockhamF32);

void hwy_stockham_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    c64* work = scratch.alloc<c64>(n);
    f64* stage_wr = scratch.alloc<f64>(n >> 1);
    f64* stage_wi = scratch.alloc<f64>(n >> 1);
    HWY_DYNAMIC_DISPATCH(StockhamF64)(buf, work, n, w_fwd, stage_wr, stage_wi);
}

void hwy_stockham_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    if (n <= 1) return;
    ScratchScope scratch;
    c32* work = scratch.alloc<c32>(n);
    f32* stage_wr = scratch.alloc<f32>(n >> 1);
    f32* stage_wi = scratch.alloc<f32>(n >> 1);
    HWY_DYNAMIC_DISPATCH(StockhamF32)(buf, work, n, w_fwd, stage_wr, stage_wi);
}

} // namespace keel::fft::detail

#endif // HWY_ONCE
