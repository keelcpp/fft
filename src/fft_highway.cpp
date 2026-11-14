#include "fft_internal.h"

namespace keel::fft_kernels {

namespace {

#if defined(__EMSCRIPTEN__)
// On WASM SIMD128, 128-bit LoadInterleaved2 is cheap, so AoS radix-2 avoids SoA conversion overhead.
constexpr usize kRadix2SoaThresholdF64 = ~usize{0};
constexpr usize kRadix4SoaThresholdF64 = ~usize{0};
constexpr usize kRadix2SoaThresholdF32 = ~usize{0};
constexpr usize kRadix4SoaThresholdF32 = ~usize{0};
#else
constexpr usize kRadix2SoaThresholdF64 = usize{1} << 11; // 2048
constexpr usize kRadix4SoaThresholdF64 = usize{1} << 14; // 16384
constexpr usize kRadix2SoaThresholdF32 = usize{1} << 10; // 1024
constexpr usize kRadix4SoaThresholdF32 = usize{1} << 13; // 8192
#endif

} // namespace

void fft_radix2_inplace(Complex64* buf, usize n, const Complex64* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    if (detail::is_pow4(n) && n >= kRadix4SoaThresholdF64) {
        detail::hwy_radix4_pow4_soa_inplace(buf, n, w_fwd);
    } else if (n >= kRadix2SoaThresholdF64) {
        detail::hwy_radix2_soa_inplace(buf, n, w_fwd);
    } else {
        detail::hwy_radix2_inplace(buf, n, w_fwd);
    }
}

void fft_radix2_inplace(Complex32* buf, usize n, const Complex32* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    if (detail::is_pow4(n) && n >= kRadix4SoaThresholdF32) {
        detail::hwy_radix4_pow4_soa_inplace(buf, n, w_fwd);
    } else if (n >= kRadix2SoaThresholdF32) {
        detail::hwy_radix2_soa_inplace(buf, n, w_fwd);
    } else {
        detail::hwy_radix2_inplace(buf, n, w_fwd);
    }
}

void fft_radix4_inplace(Complex64* buf, usize n, const Complex64* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    if (detail::is_pow4(n)) {
        detail::hwy_radix4_pow4_inplace(buf, n, w_fwd);
    } else {
        detail::hwy_radix2_inplace(buf, n, w_fwd);
    }
}

void fft_radix4_inplace(Complex32* buf, usize n, const Complex32* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    if (detail::is_pow4(n)) {
        detail::hwy_radix4_pow4_inplace(buf, n, w_fwd);
    } else {
        detail::hwy_radix2_inplace(buf, n, w_fwd);
    }
}

void fft_stockham_inplace(Complex64* buf, usize n, const Complex64* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    detail::hwy_stockham_inplace(buf, n, w_fwd);
}

void fft_stockham_inplace(Complex32* buf, usize n, const Complex32* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    detail::hwy_stockham_inplace(buf, n, w_fwd);
}

void fft_radix2_soa_inplace(Complex64* buf, usize n, const Complex64* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    detail::hwy_radix2_soa_inplace(buf, n, w_fwd);
}

void fft_radix2_soa_inplace(Complex32* buf, usize n, const Complex32* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    detail::hwy_radix2_soa_inplace(buf, n, w_fwd);
}

void fft_radix4_soa_inplace(Complex64* buf, usize n, const Complex64* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    if (detail::is_pow4(n)) {
        detail::hwy_radix4_pow4_soa_inplace(buf, n, w_fwd);
    } else {
        detail::hwy_radix2_soa_inplace(buf, n, w_fwd);
    }
}

void fft_radix4_soa_inplace(Complex32* buf, usize n, const Complex32* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
    if (detail::is_pow4(n)) {
        detail::hwy_radix4_pow4_soa_inplace(buf, n, w_fwd);
    } else {
        detail::hwy_radix2_soa_inplace(buf, n, w_fwd);
    }
}

void fft_soa_stages_inplace(f64* re, f64* im, usize n, const Complex64* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
#if !defined(__EMSCRIPTEN__)
    if (detail::is_pow4(n) && n >= kRadix4SoaThresholdF64) {
        detail::hwy_radix4_pow4_soa_stages_inplace(re, im, n, w_fwd);
        return;
    }
#endif
    detail::hwy_radix2_soa_stages_inplace(re, im, n, w_fwd);
}

void fft_soa_stages_inplace(f32* re, f32* im, usize n, const Complex32* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(detail::is_pow2(n));
#if !defined(__EMSCRIPTEN__)
    if (detail::is_pow4(n) && n >= kRadix4SoaThresholdF32) {
        detail::hwy_radix4_pow4_soa_stages_inplace(re, im, n, w_fwd);
        return;
    }
#endif
    detail::hwy_radix2_soa_stages_inplace(re, im, n, w_fwd);
}

} // namespace keel::fft_kernels
