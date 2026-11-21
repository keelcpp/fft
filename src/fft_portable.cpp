#include "fft_portable_impl.h"

namespace keel::fft::detail {

void rfft_twist_full_unit_stride(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale,
                                 c64* dst) noexcept {
    portable_rfft_twist_full(z_re, z_im, n, w_full, scale, dst, 1);
}

void rfft_twist_full_unit_stride(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale,
                                 c32* dst) noexcept {
    portable_rfft_twist_full(z_re, z_im, n, w_full, scale, dst, 1);
}

void rfft_twist_half_unit_stride(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale,
                                 c64* half_dst) noexcept {
    portable_rfft_twist_half(z_re, z_im, n, w_full, scale, half_dst, 1);
}

void rfft_twist_half_unit_stride(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale,
                                 c32* half_dst) noexcept {
    portable_rfft_twist_half(z_re, z_im, n, w_full, scale, half_dst, 1);
}

void irfft_untwist_conj_unit_stride(const c64* half_src, usize n, const c64* w_full, f64* z_re,
                                    f64* z_im) noexcept {
    portable_irfft_untwist_conj(half_src, 1, n, w_full, z_re, z_im);
}

void irfft_untwist_conj_unit_stride(const c32* half_src, usize n, const c32* w_full, f32* z_re,
                                    f32* z_im) noexcept {
    portable_irfft_untwist_conj(half_src, 1, n, w_full, z_re, z_im);
}

void irfft_store_interleaved_unit_stride(const f64* z_re, const f64* z_im, usize half, f64 post_scale,
                                         f64* real_dst) noexcept {
    portable_irfft_store_interleaved(z_re, z_im, half, post_scale, real_dst, 1);
}

void irfft_store_interleaved_unit_stride(const f32* z_re, const f32* z_im, usize half, f32 post_scale,
                                         f32* real_dst) noexcept {
    portable_irfft_store_interleaved(z_re, z_im, half, post_scale, real_dst, 1);
}

} // namespace keel::fft::detail

namespace keel::fft::fft_kernels {

namespace {

template <typename T, typename Cplx>
void soa_wrap(Cplx* buf, usize n, const Cplx* w_fwd,
              void (*stages_fn)(T*, T*, usize, const Cplx*) noexcept) noexcept {
    if (n <= 1) return;
    detail::ScratchScope scratch;
    T* re = scratch.alloc<T>(n);
    T* im = scratch.alloc<T>(n);
    for (usize i = 0; i < n; ++i) {
        re[i] = buf[i].re;
        im[i] = buf[i].im;
    }
    stages_fn(re, im, n, w_fwd);
    for (usize i = 0; i < n; ++i) {
        buf[i] = Cplx(re[i], im[i]);
    }
}

} // namespace

void fft_radix2_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    if (detail::is_pow4(n) && n >= 64) {
        detail::portable_radix4_pow4_inplace(buf, n, w_fwd);
    } else {
        detail::portable_radix2_inplace(buf, n, w_fwd);
    }
}

void fft_radix2_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    if (detail::is_pow4(n) && n >= 64) {
        detail::portable_radix4_pow4_inplace(buf, n, w_fwd);
    } else {
        detail::portable_radix2_inplace(buf, n, w_fwd);
    }
}

void fft_radix4_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    detail::portable_radix4_pow4_inplace(buf, n, w_fwd);
}

void fft_radix4_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    detail::portable_radix4_pow4_inplace(buf, n, w_fwd);
}

void fft_stockham_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    detail::portable_stockham_inplace(buf, n, w_fwd);
}

void fft_stockham_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    detail::portable_stockham_inplace(buf, n, w_fwd);
}

void fft_radix2_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    soa_wrap<f64, c64>(buf, n, w_fwd, &detail::portable_radix2_soa_stages_inplace<f64, c64>);
}

void fft_radix2_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    soa_wrap<f32, c32>(buf, n, w_fwd, &detail::portable_radix2_soa_stages_inplace<f32, c32>);
}

void fft_radix4_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept {
    soa_wrap<f64, c64>(buf, n, w_fwd, &detail::portable_radix4_pow4_soa_stages_inplace<f64, c64>);
}

void fft_radix4_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept {
    soa_wrap<f32, c32>(buf, n, w_fwd, &detail::portable_radix4_pow4_soa_stages_inplace<f32, c32>);
}

void fft_soa_stages_inplace(f64* re, f64* im, usize n, const c64* w_fwd) noexcept {
    if (detail::is_pow4(n) && n >= 64) {
        detail::portable_radix4_pow4_soa_stages_inplace(re, im, n, w_fwd);
    } else {
        detail::portable_radix2_soa_stages_inplace(re, im, n, w_fwd);
    }
}

void fft_soa_stages_inplace(f32* re, f32* im, usize n, const c32* w_fwd) noexcept {
    if (detail::is_pow4(n) && n >= 64) {
        detail::portable_radix4_pow4_soa_stages_inplace(re, im, n, w_fwd);
    } else {
        detail::portable_radix2_soa_stages_inplace(re, im, n, w_fwd);
    }
}

} // namespace keel::fft::fft_kernels
