#include "keel/fft.h"
#include "keel/fft_interop_std.h"

#include "keel/arena.h"

#include <cmath>
#include <cstdio>

using namespace keel;

namespace {

constexpr f64 kPi = 3.141592653589793238462643383279502884;

int g_tests_passed = 0;
int g_tests_failed = 0;

#define TEST_CHECK(cond)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);                                     \
            ++g_tests_failed;                                                                                          \
        } else {                                                                                                       \
            ++g_tests_passed;                                                                                          \
        }                                                                                                              \
    } while (0)

struct Rng {
    u64 state = 0x853C49E6748FEA9BULL;

    u32 next_u32() noexcept {
        const u64 old = state;
        state = old * 6364136223846793005ULL + 1442695040888963407ULL;
        const u32 xorshifted = static_cast<u32>(((old >> 18u) ^ old) >> 27u);
        const u32 rot = static_cast<u32>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
    }

    f64 next_f64() noexcept {
        // Uniform in [-1.0, 1.0]
        return (static_cast<f64>(next_u32()) / 2147483647.5) - 1.0;
    }

    f32 next_f32() noexcept { return static_cast<f32>(next_f64()); }
};

f64 max_abs_diff_c64(ArrayRef<Complex64> a, ArrayRef<Complex64> b) noexcept {
    if (a.size() != b.size()) return 1e100;
    f64 m = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        const f64 dr = std::fabs(a[i].re - b[i].re);
        const f64 di = std::fabs(a[i].im - b[i].im);
        if (dr > m) m = dr;
        if (di > m) m = di;
    }
    return m;
}

f64 max_abs_diff_c32_vs_c64(ArrayRef<Complex32> a, ArrayRef<Complex64> b) noexcept {
    if (a.size() != b.size()) return 1e100;
    f64 m = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        const f64 dr = std::fabs(static_cast<f64>(a[i].re) - b[i].re);
        const f64 di = std::fabs(static_cast<f64>(a[i].im) - b[i].im);
        if (dr > m) m = dr;
        if (di > m) m = di;
    }
    return m;
}

f64 max_abs_diff_c32(ArrayRef<Complex32> a, ArrayRef<Complex32> b) noexcept {
    if (a.size() != b.size()) return 1e100;
    f64 m = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        const f64 dr = std::fabs(static_cast<f64>(a[i].re) - static_cast<f64>(b[i].re));
        const f64 di = std::fabs(static_cast<f64>(a[i].im) - static_cast<f64>(b[i].im));
        if (dr > m) m = dr;
        if (di > m) m = di;
    }
    return m;
}

f64 max_abs_diff_f64(ArrayRef<f64> a, ArrayRef<f64> b) noexcept {
    if (a.size() != b.size()) return 1e100;
    f64 m = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        const f64 d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

f64 max_abs_diff_f32(ArrayRef<f32> a, ArrayRef<f32> b) noexcept {
    if (a.size() != b.size()) return 1e100;
    f64 m = 0.0;
    for (usize i = 0; i < a.size(); ++i) {
        const f64 d = std::fabs(static_cast<f64>(a[i]) - static_cast<f64>(b[i]));
        if (d > m) m = d;
    }
    return m;
}

// Exact O(N^2) double-precision DFT reference with modular angle reduction to avoid precision drift.
Array<Complex64> naive_dft(ArrayRef<Complex64> x, FftDirection dir,
                           FftNormalization norm = FftNormalization::Backward) {
    const usize n = x.size();
    Array<Complex64> out(n);
    if (n == 0) return out;

    const f64 sign = static_cast<f64>(static_cast<i8>(dir));
    const f64 two_pi_over_n = 2.0 * kPi / static_cast<f64>(n);
    f64 scale = 1.0;
    if (n > 1) {
        if (norm == FftNormalization::Backward && dir == FftDirection::Inverse) {
            scale = 1.0 / static_cast<f64>(n);
        } else if (norm == FftNormalization::Ortho) {
            scale = 1.0 / std::sqrt(static_cast<f64>(n));
        }
    }

    for (usize k = 0; k < n; ++k) {
        f64 sum_re = 0.0;
        f64 sum_im = 0.0;
        for (usize j = 0; j < n; ++j) {
            const u64 prod = (static_cast<u64>(k) * static_cast<u64>(j)) % static_cast<u64>(n);
            const f64 angle = sign * two_pi_over_n * static_cast<f64>(prod);
            const f64 wr = std::cos(angle);
            const f64 wi = std::sin(angle);
            sum_re += x[j].re * wr - x[j].im * wi;
            sum_im += x[j].re * wi + x[j].im * wr;
        }
        out[k] = Complex64(sum_re * scale, sum_im * scale);
    }
    return out;
}

void test_complex_and_shift() {
    constexpr Complex64 a(3.0, -4.0);
    constexpr Complex64 b(1.5, 2.0);
    static_assert(conj(a) == Complex64(3.0, 4.0));
    static_assert(norm_sq(a) == 25.0);
    static_assert((a + b) == Complex64(4.5, -2.0));
    static_assert((a - b) == Complex64(1.5, -6.0));
    static_assert((a * b) == Complex64(12.5, 0.0));
    static_assert((-a) == Complex64(-3.0, 4.0));

    Complex64 c = a;
    c += b;
    c -= b;
    TEST_CHECK(c == a);
    c *= 2.0;
    c /= 2.0;
    TEST_CHECK(c == a);

    constexpr Complex32 a32 = to_complex32(a);
    static_assert(a32 == Complex32(3.0f, -4.0f));
    static_assert(to_complex64(a32) == a);
    static_assert(conj(a32) == Complex32(3.0f, 4.0f));
    static_assert(norm_sq(a32) == 25.0f);

    // Even and odd fftshift / ifftshift round-trip
    const int even_in[6] = {0, 1, 2, 3, 4, 5};
    const int even_expected[6] = {3, 4, 5, 0, 1, 2};
    int even_shifted[6]{};
    int even_back[6]{};
    fftshift_1d<int>(even_in, even_shifted);
    ifftshift_1d<int>(even_shifted, even_back);
    for (usize i = 0; i < 6; ++i) {
        TEST_CHECK(even_shifted[i] == even_expected[i]);
        TEST_CHECK(even_back[i] == even_in[i]);
    }

    const int odd_in[5] = {0, 1, 2, 3, 4};
    const int odd_expected[5] = {3, 4, 0, 1, 2};
    int odd_shifted[5]{};
    int odd_back[5]{};
    fftshift_1d<int>(odd_in, odd_shifted);
    ifftshift_1d<int>(odd_shifted, odd_back);
    for (usize i = 0; i < 5; ++i) {
        TEST_CHECK(odd_shifted[i] == odd_expected[i]);
        TEST_CHECK(odd_back[i] == odd_in[i]);
    }

    // Empty no-op safety
    ArrayRef<Complex64> empty64{};
    MutArrayRef<Complex64> mut_empty64{};
    fft_1d_inplace(mut_empty64);
    ifft_1d_inplace(mut_empty64);
    fft_1d(empty64, mut_empty64);
    ifft_1d(empty64, mut_empty64);
    ArrayRef<f64> empty_r64{};
    MutArrayRef<f64> mut_empty_r64{};
    rfft_1d(empty_r64, mut_empty64);
    irfft_1d(empty64, mut_empty_r64, 0);
    TEST_CHECK(true);
}

void test_power_of_two_and_low_level_kernels() {
    Rng rng;
    const usize sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};

    for (usize n : sizes) {
        Array<Complex64> in64(n);
        Array<Complex32> in32(n);
        for (usize i = 0; i < n; ++i) {
            const f64 re = rng.next_f64();
            const f64 im = rng.next_f64();
            in64[i] = Complex64(re, im);
            in32[i] = Complex32(static_cast<f32>(re), static_cast<f32>(im));
        }

        // Exact reference computed from both exact f64 input and quantized f32 input
        Array<Complex64> in32_as_64(n);
        for (usize i = 0; i < n; ++i) {
            in32_as_64[i] = to_complex64(in32[i]);
        }
        const Array<Complex64> ref_fwd64 = naive_dft(in64, FftDirection::Forward, FftNormalization::Backward);
        const Array<Complex64> ref_inv64 = naive_dft(in64, FftDirection::Inverse, FftNormalization::Backward);
        const Array<Complex64> ref_fwd32 = naive_dft(in32_as_64, FftDirection::Forward, FftNormalization::Backward);

        // 1. High-level stateless f64 forward + inverse + round-trip
        Array<Complex64> work64 = in64.clone();
        fft_1d_inplace(work64);
        TEST_CHECK(max_abs_diff_c64(work64, ref_fwd64) < 1e-11);
        ifft_1d_inplace(work64);
        TEST_CHECK(max_abs_diff_c64(work64, in64) < 1e-11);

        Array<Complex64> inv_direct64(n);
        ifft_1d(in64, inv_direct64);
        TEST_CHECK(max_abs_diff_c64(inv_direct64, ref_inv64) < 1e-11);

        // Ortho normalization round-trip
        Array<Complex64> ortho64(n);
        fft_1d(in64, ortho64, FftNormalization::Ortho);
        const Array<Complex64> ref_ortho = naive_dft(in64, FftDirection::Forward, FftNormalization::Ortho);
        TEST_CHECK(max_abs_diff_c64(ortho64, ref_ortho) < 1e-11);
        ifft_1d_inplace(ortho64, FftNormalization::Ortho);
        TEST_CHECK(max_abs_diff_c64(ortho64, in64) < 1e-11);

        // 2. High-level stateless f32 forward + inverse round-trip
        // Relative to DFT magnitude (which grows like sqrt(N)), keep strict < 1e-4 bound
        const f64 tol32 = (n <= 1024) ? 5e-5 : 1e-4;
        Array<Complex32> work32 = in32.clone();
        fft_1d_inplace(work32);
        TEST_CHECK(max_abs_diff_c32_vs_c64(work32, ref_fwd32) < tol32);
        ifft_1d_inplace(work32);
        TEST_CHECK(max_abs_diff_c32(work32, in32) < 1e-5);

        // 3. Low-level kernel entry points (f64 & f32)
        Array<Complex64> w64(n >> 1);
        fill_twiddles_f64(w64, n, FftDirection::Forward);
        Array<Complex32> w32(n >> 1);
        fill_twiddles_f32(w32, n, FftDirection::Forward);

        auto check_kernel64 = [&](void (*fn)(Complex64*, usize, const Complex64*) noexcept) {
            Array<Complex64> b = in64.clone();
            fn(b.data(), n, w64.data());
            TEST_CHECK(max_abs_diff_c64(b, ref_fwd64) < 1e-11);
        };
        check_kernel64(&fft_radix2_inplace);
        check_kernel64(&fft_radix4_inplace);
        check_kernel64(&fft_stockham_inplace);
        check_kernel64(&fft_radix2_soa_inplace);
        check_kernel64(&fft_radix4_soa_inplace);

        {
            Array<f64> re(n), im(n);
            for (usize i = 0; i < n; ++i) {
                re[i] = in64[i].re;
                im[i] = in64[i].im;
            }
            fft_soa_stages_inplace(re.data(), im.data(), n, w64.data());
            Array<Complex64> soa_out(n);
            for (usize i = 0; i < n; ++i) {
                soa_out[i] = Complex64(re[i], im[i]);
            }
            TEST_CHECK(max_abs_diff_c64(soa_out, ref_fwd64) < 1e-11);
        }

        auto check_kernel32 = [&](void (*fn)(Complex32*, usize, const Complex32*) noexcept) {
            Array<Complex32> b = in32.clone();
            fn(b.data(), n, w32.data());
            TEST_CHECK(max_abs_diff_c32_vs_c64(b, ref_fwd32) < tol32);
        };
        check_kernel32(&fft_radix2_inplace);
        check_kernel32(&fft_radix4_inplace);
        check_kernel32(&fft_stockham_inplace);
        check_kernel32(&fft_radix2_soa_inplace);
        check_kernel32(&fft_radix4_soa_inplace);

        {
            Array<f32> re(n), im(n);
            for (usize i = 0; i < n; ++i) {
                re[i] = in32[i].re;
                im[i] = in32[i].im;
            }
            fft_soa_stages_inplace(re.data(), im.data(), n, w32.data());
            Array<Complex32> soa_out(n);
            for (usize i = 0; i < n; ++i) {
                soa_out[i] = Complex32(re[i], im[i]);
            }
            TEST_CHECK(max_abs_diff_c32_vs_c64(soa_out, ref_fwd32) < tol32);
        }
    }

    // Large threshold check: N = 16384 exercises Radix-4 SoA dispatcher in both f64 and f32
    {
        const usize n = 16384;
        Array<Complex64> x64(n);
        Array<Complex32> x32(n);
        for (usize i = 0; i < n; ++i) {
            const f64 re = rng.next_f64();
            const f64 im = rng.next_f64();
            x64[i] = Complex64(re, im);
            x32[i] = Complex32(static_cast<f32>(re), static_cast<f32>(im));
        }

        Array<Complex64> a64 = x64.clone();
        Array<Complex64> b64 = x64.clone();
        Array<Complex64> w64(n >> 1);
        fill_twiddles_f64(w64, n, FftDirection::Forward);
        fft_radix2_inplace(a64.data(), n, w64.data());
        fft_stockham_inplace(b64.data(), n, w64.data());
        TEST_CHECK(max_abs_diff_c64(a64, b64) < 1e-10);
        ifft_1d_inplace(a64);
        TEST_CHECK(max_abs_diff_c64(a64, x64) < 1e-11);

        Array<Complex32> a32 = x32.clone();
        fft_1d_inplace(a32);
        ifft_1d_inplace(a32);
        TEST_CHECK(max_abs_diff_c32(a32, x32) < 1e-5);
    }
}

void test_bluestein_arbitrary_lengths_and_padding() {
    Rng rng;
    const usize sizes[] = {3, 5, 6, 7, 15, 17, 100, 127, 480, 1000};

    for (usize n : sizes) {
        Array<Complex64> in64(n);
        Array<Complex32> in32(n);
        Array<Complex64> in32_as_64(n);
        for (usize i = 0; i < n; ++i) {
            const f64 re = rng.next_f64();
            const f64 im = rng.next_f64();
            in64[i] = Complex64(re, im);
            in32[i] = Complex32(static_cast<f32>(re), static_cast<f32>(im));
            in32_as_64[i] = to_complex64(in32[i]);
        }

        const Array<Complex64> ref_fwd64 = naive_dft(in64, FftDirection::Forward);
        const Array<Complex64> ref_inv64 = naive_dft(in64, FftDirection::Inverse);
        const Array<Complex64> ref_fwd32 = naive_dft(in32_as_64, FftDirection::Forward);

        Array<Complex64> out64(n);
        fft_1d(in64, out64);
        TEST_CHECK(max_abs_diff_c64(out64, ref_fwd64) < 1e-11);

        Array<Complex64> back64(n);
        ifft_1d(out64, back64);
        TEST_CHECK(max_abs_diff_c64(back64, in64) < 1e-11);

        Array<Complex64> inv64(n);
        ifft_1d(in64, inv64);
        TEST_CHECK(max_abs_diff_c64(inv64, ref_inv64) < 1e-11);

        // Ortho round-trip on non-pow2
        Array<Complex64> ortho64 = in64.clone();
        fft_1d_inplace(ortho64, FftNormalization::Ortho);
        ifft_1d_inplace(ortho64, FftNormalization::Ortho);
        TEST_CHECK(max_abs_diff_c64(ortho64, in64) < 1e-11);

        // f32 Bluestein forward + inverse round-trip
        Array<Complex32> out32(n);
        fft_1d(in32, out32);
        TEST_CHECK(max_abs_diff_c32_vs_c64(out32, ref_fwd32) < 1e-4);

        Array<Complex32> back32(n);
        ifft_1d(out32, back32);
        TEST_CHECK(max_abs_diff_c32(back32, in32) < 2e-5);
    }

    // Zero-padding and truncation in fft_1d / ifft_1d (both pow2 and non-pow2 target sizes)
    Array<Complex64> short_in(11);
    for (usize i = 0; i < 11; ++i) {
        short_in[i] = Complex64(rng.next_f64(), rng.next_f64());
    }
    for (usize dst_n : {usize{5}, usize{16}, usize{23}}) {
        Array<Complex64> padded(dst_n, Complex64(0.0, 0.0));
        const usize copy_n = (short_in.size() < dst_n) ? short_in.size() : dst_n;
        for (usize i = 0; i < copy_n; ++i) padded[i] = short_in[i];

        const Array<Complex64> ref_f = naive_dft(padded, FftDirection::Forward);
        const Array<Complex64> ref_i = naive_dft(padded, FftDirection::Inverse);

        Array<Complex64> got_f(dst_n), got_i(dst_n);
        fft_1d(short_in, got_f);
        ifft_1d(short_in, got_i);
        TEST_CHECK(max_abs_diff_c64(got_f, ref_f) < 1e-11);
        TEST_CHECK(max_abs_diff_c64(got_i, ref_i) < 1e-11);
    }
}

void test_real_fft_and_irfft() {
    Rng rng;
    const usize sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 64, 100, 127, 256, 480, 1000, 1024};

    for (usize n : sizes) {
        Array<f64> r64(n);
        Array<f32> r32(n);
        Array<Complex64> c_ref_in64(n);
        Array<Complex64> c_ref_in32(n);
        for (usize i = 0; i < n; ++i) {
            const f64 v = rng.next_f64();
            r64[i] = v;
            r32[i] = static_cast<f32>(v);
            c_ref_in64[i] = Complex64(v, 0.0);
            c_ref_in32[i] = Complex64(static_cast<f64>(r32[i]), 0.0);
        }

        const Array<Complex64> ref64 = naive_dft(c_ref_in64, FftDirection::Forward);
        const Array<Complex64> ref32 = naive_dft(c_ref_in32, FftDirection::Forward);
        const usize half_bins = (n / 2) + 1;

        // f64 fft_1d_real (full N) + rfft_1d (n/2 + 1) + irfft_1d + ifft_1d_symmetric
        Array<Complex64> full64(n);
        fft_1d_real(r64, full64);
        TEST_CHECK(max_abs_diff_c64(full64, ref64) < 1e-11);
        TEST_CHECK(full64[0].im == 0.0);
        if ((n & 1u) == 0u) {
            TEST_CHECK(full64[n / 2].im == 0.0);
        }

        Array<Complex64> half64(half_bins);
        rfft_1d(r64, half64);
        TEST_CHECK(max_abs_diff_c64(half64, ArrayRef<Complex64>(ref64).slice(0, half_bins)) < 1e-11);

        Array<f64> rec64(n);
        irfft_1d(half64, rec64, n);
        TEST_CHECK(max_abs_diff_f64(rec64, r64) < 1e-11);

        // Perturb upper conjugate half & DC/Nyquist imag before symmetric ifft: lower half must win
        Array<Complex64> noisy_sym64 = full64.clone();
        noisy_sym64[0].im = 99.0;
        if ((n & 1u) == 0u) noisy_sym64[n / 2].im = -77.0;
        for (usize k = half_bins; k < n; ++k) {
            noisy_sym64[k] = Complex64(123.0, -456.0);
        }
        Array<f64> sym_rec64(n);
        ifft_1d_symmetric(noisy_sym64, sym_rec64);
        TEST_CHECK(max_abs_diff_f64(sym_rec64, r64) < 1e-11);

        // Ortho rfft/irfft round-trip
        Array<Complex64> half_ortho64(half_bins);
        rfft_1d(r64, half_ortho64, FftNormalization::Ortho);
        Array<f64> rec_ortho64(n);
        irfft_1d(half_ortho64, rec_ortho64, n, FftNormalization::Ortho);
        TEST_CHECK(max_abs_diff_f64(rec_ortho64, r64) < 1e-11);

        // f32 fft_1d_real + rfft_1d + irfft_1d + ifft_1d_symmetric
        Array<Complex32> full32(n);
        fft_1d_real(r32, full32);
        TEST_CHECK(max_abs_diff_c32_vs_c64(full32, ref32) < 1e-4);

        Array<Complex32> half32(half_bins);
        rfft_1d(r32, half32);
        TEST_CHECK(max_abs_diff_c32_vs_c64(half32, ArrayRef<Complex64>(ref32).slice(0, half_bins)) < 1e-4);

        Array<f32> rec32(n);
        irfft_1d(half32, rec32, n);
        TEST_CHECK(max_abs_diff_f32(rec32, r32) < 2e-5);

        Array<f32> sym_rec32(n);
        ifft_1d_symmetric(full32, sym_rec32);
        TEST_CHECK(max_abs_diff_f32(sym_rec32, r32) < 2e-5);
    }

    // Zero-padded odd/even real input into pow2 fft_1d_real (exercises packed half-length zero-pad path)
    for (usize src_len : {usize{7}, usize{8}, usize{9}}) {
        Array<f64> r_short(src_len);
        Array<Complex64> c_padded(32, Complex64(0.0, 0.0));
        for (usize i = 0; i < src_len; ++i) {
            r_short[i] = rng.next_f64();
            c_padded[i] = Complex64(r_short[i], 0.0);
        }
        const Array<Complex64> ref = naive_dft(c_padded, FftDirection::Forward);
        Array<Complex64> got(32);
        fft_1d_real(r_short, got);
        TEST_CHECK(max_abs_diff_c64(got, ref) < 1e-11);
    }
}

struct CountingCtx {
    Allocator* upstream = nullptr;
    usize alloc_calls = 0;
};

void* counting_alloc(void* ctx, usize size, usize align) noexcept {
    auto* c = static_cast<CountingCtx*>(ctx);
    ++c->alloc_calls;
    return c->upstream->alloc(c->upstream->ctx, size, align);
}

void* counting_realloc(void* ctx, void* ptr, usize old_size, usize new_size, usize align) noexcept {
    auto* c = static_cast<CountingCtx*>(ctx);
    ++c->alloc_calls;
    return c->upstream->realloc(c->upstream->ctx, ptr, old_size, new_size, align);
}

void counting_free(void* ctx, void* ptr, usize size, usize align) noexcept {
    auto* c = static_cast<CountingCtx*>(ctx);
    c->upstream->free(c->upstream->ctx, ptr, size, align);
}

void test_plans_arena_and_zero_exec_allocations() {
    Rng rng;

    // Invalid n == 0 returns InvalidArgument
    auto bad64 = FftPlan1D::create(0);
    TEST_CHECK(bad64.is_err() && bad64.error().code == ErrorCode::InvalidArgument);
    auto bad32 = FftPlan1DF32::create(0);
    TEST_CHECK(bad32.is_err() && bad32.error().code == ErrorCode::InvalidArgument);

    // Custom Arena allocator routing + move + clone + multi-call reuse
    Arena custom_arena(2 * 1024 * 1024);
    const usize used_before = custom_arena.used();
    auto p_pow2_res = FftPlan1D::create(512, custom_arena.allocator());
    auto p_blue_res = FftPlan1D::create(127, custom_arena.allocator());
    auto p_f32_res = FftPlan1DF32::create(256, custom_arena.allocator());
    TEST_CHECK(p_pow2_res.is_ok() && p_blue_res.is_ok() && p_f32_res.is_ok());
    TEST_CHECK(custom_arena.used() > used_before);

    FftPlan1D plan_pow2 = keel::move(*p_pow2_res);
    TEST_CHECK(p_pow2_res->empty() && p_pow2_res->size() == 0);
    TEST_CHECK(plan_pow2.size() == 512 && plan_pow2.is_power_of_two());
    TEST_CHECK(plan_pow2.allocator() == custom_arena.allocator());

    FftPlan1D plan_blue = keel::move(*p_blue_res);
    TEST_CHECK(plan_blue.size() == 127 && !plan_blue.is_power_of_two());

    FftPlan1D cloned_blue = plan_blue.clone(custom_arena.allocator());
    TEST_CHECK(cloned_blue.size() == 127 && cloned_blue.allocator() == custom_arena.allocator());

    // Pre-allocate work buffers before installing the counting default allocator
    Array<Complex64> c512(512), c512_out(512);
    Array<f64> r512(512), r512_out(512);
    Array<Complex64> h512(257);
    for (usize i = 0; i < 512; ++i) {
        r512[i] = rng.next_f64();
        c512[i] = Complex64(r512[i], rng.next_f64());
    }

    Array<Complex64> c127(127), c127_orig(127);
    Array<f64> r127(127), r127_out(127);
    Array<Complex64> h127(64);
    for (usize i = 0; i < 127; ++i) {
        r127[i] = rng.next_f64();
        c127[i] = Complex64(r127[i], rng.next_f64());
        c127_orig[i] = c127[i];
    }

    Array<Complex32> c256(256), c256_orig(256);
    for (usize i = 0; i < 256; ++i) {
        c256[i] = Complex32(rng.next_f32(), rng.next_f32());
        c256_orig[i] = c256[i];
    }

    // Warm up thread-local scratch arena once, then verify ZERO heap allocations across repeated plan executions!
    (void)get_scratch();

    Allocator* prev_default = default_allocator();
    CountingCtx count_ctx{.upstream = prev_default, .alloc_calls = 0};
    Allocator counting{
        .alloc = &counting_alloc,
        .realloc = &counting_realloc,
        .free = &counting_free,
        .ctx = &count_ctx,
    };
    const usize arena_used_before_exec = custom_arena.used();
    set_default_allocator(&counting);

    for (int iter = 0; iter < 4; ++iter) {
        plan_pow2.exec(c512, c512_out, FftDirection::Forward);
        plan_pow2.exec_inplace(c512_out, FftDirection::Inverse);
        plan_pow2.exec_rfft(r512, h512);
        plan_pow2.exec_irfft(h512, r512_out);

        cloned_blue.exec_inplace(c127, FftDirection::Forward);
        cloned_blue.exec_inplace(c127, FftDirection::Inverse);
        cloned_blue.exec_rfft(r127, h127);
        cloned_blue.exec_irfft(h127, r127_out);

        p_f32_res->exec_inplace(c256, FftDirection::Forward);
        p_f32_res->exec_inplace(c256, FftDirection::Inverse);
    }

    set_default_allocator(prev_default);

    TEST_CHECK(count_ctx.alloc_calls == 0);
    TEST_CHECK(custom_arena.used() == arena_used_before_exec);
    TEST_CHECK(max_abs_diff_c64(c512_out, c512) < 1e-11);
    TEST_CHECK(max_abs_diff_f64(r512_out, r512) < 1e-11);
    TEST_CHECK(max_abs_diff_c64(c127, c127_orig) < 1e-11);
    TEST_CHECK(max_abs_diff_f64(r127_out, r127) < 1e-11);
    TEST_CHECK(max_abs_diff_c32(c256, c256_orig) < 1e-5);
}

void test_strided_nd_batches() {
    Rng rng;

    // 3D tensor shape [4, 6, 5]: covers pow2 (4), even non-pow2 (6), and odd prime (5) axes!
    const usize dims[3] = {4, 6, 5};
    const usize total = dims[0] * dims[1] * dims[2];

    Array<Complex64> tensor64(total);
    Array<Complex32> tensor32(total);
    Array<f64> rtensor64(total);
    Array<f32> rtensor32(total);
    for (usize i = 0; i < total; ++i) {
        const f64 re = rng.next_f64();
        const f64 im = rng.next_f64();
        tensor64[i] = Complex64(re, im);
        tensor32[i] = Complex32(static_cast<f32>(re), static_cast<f32>(im));
        rtensor64[i] = re;
        rtensor32[i] = static_cast<f32>(re);
    }

    // Test both Column-Major and Row-Major across axes 0, 1, 2, with exact, zero-padded (+2 -> 6,8,7 or 8), and truncated (-1) lengths
    for (bool col_major : {true, false}) {
        for (usize axis = 0; axis < 3; ++axis) {
            const usize axis_len = dims[axis];
            const usize target_lens[3] = {axis_len, usize{8}, axis_len - 1};

            for (usize target_n : target_lens) {
                const FftBatchLayout layout = col_major
                                                  ? FftBatchLayout::from_col_major(dims, axis, target_n)
                                                  : FftBatchLayout::from_row_major(dims, axis, target_n);

                usize out_dims[3] = {dims[0], dims[1], dims[2]};
                out_dims[axis] = target_n;
                const usize out_total = out_dims[0] * out_dims[1] * out_dims[2];

                Array<Complex64> batch_out64(out_total);
                fft_batch(tensor64, batch_out64, layout, FftDirection::Forward);

                Array<Complex32> batch_out32(out_total);
                fft_batch(tensor32, batch_out32, layout, FftDirection::Forward);

                Array<Complex64> batch_real_out64(out_total);
                fft_batch_real(rtensor64, batch_real_out64, layout);

                // Verify every fiber against 1D fft_1d / fft_1d_real
                const usize use_len = (axis_len < target_n) ? axis_len : target_n;
                Array<Complex64> slice_in(axis_len);
                Array<Complex64> slice_exp(target_n);
                Array<Complex64> slice_got(target_n);
                Array<f64> rslice_in(axis_len);
                Array<Complex64> rslice_exp(target_n);
                Array<Complex64> rslice_got(target_n);

                for (usize o = 0; o < layout.outer_count; ++o) {
                    for (usize i = 0; i < layout.inner_count; ++i) {
                        const usize in_base = o * layout.outer_src_step + i * layout.inner_step;
                        const usize out_base = o * layout.outer_dst_step + i * layout.inner_step;
                        for (usize k = 0; k < axis_len; ++k) {
                            slice_in[k] = tensor64[in_base + k * layout.elem_stride];
                            rslice_in[k] = rtensor64[in_base + k * layout.elem_stride];
                        }
                        (void)use_len;
                        fft_1d(slice_in, slice_exp);
                        fft_1d_real(rslice_in, rslice_exp);

                        for (usize k = 0; k < target_n; ++k) {
                            slice_got[k] = batch_out64[out_base + k * layout.effective_dst_stride()];
                            rslice_got[k] = batch_real_out64[out_base + k * layout.effective_dst_stride()];
                        }
                        TEST_CHECK(max_abs_diff_c64(slice_got, slice_exp) < 1e-11);
                        TEST_CHECK(max_abs_diff_c64(rslice_got, rslice_exp) < 1e-11);
                    }
                }

                // When target_n == axis_len, verify full C2C, RFFT/IRFFT, and Symmetric IFFT round-trips!
                if (target_n == axis_len) {
                    Array<Complex64> inv_out64(total);
                    ifft_batch(batch_out64, inv_out64, layout);
                    TEST_CHECK(max_abs_diff_c64(inv_out64, tensor64) < 1e-11);

                    Array<Complex32> inv_out32(total);
                    ifft_batch(batch_out32, inv_out32, layout);
                    TEST_CHECK(max_abs_diff_c32(inv_out32, tensor32) < 2e-5);

                    Array<f64> sym_rec64(total);
                    ifft_batch_symmetric(batch_real_out64, sym_rec64, layout);
                    TEST_CHECK(max_abs_diff_f64(sym_rec64, rtensor64) < 1e-11);

                    // Half-spectrum rfft_batch -> irfft_batch round-trip (f64 & f32)
                    const usize half_bins = (axis_len / 2) + 1;
                    usize half_dims[3] = {dims[0], dims[1], dims[2]};
                    half_dims[axis] = half_bins;
                    const usize half_total = half_dims[0] * half_dims[1] * half_dims[2];

                    const FftBatchLayout rfft_layout =
                        col_major ? FftBatchLayout::from_col_major(dims, axis, axis_len, half_bins)
                                  : FftBatchLayout::from_row_major(dims, axis, axis_len, half_bins);
                    const FftBatchLayout irfft_layout =
                        col_major ? FftBatchLayout::from_col_major(half_dims, axis, axis_len, axis_len)
                                  : FftBatchLayout::from_row_major(half_dims, axis, axis_len, axis_len);

                    Array<Complex64> half_spec64(half_total);
                    rfft_batch(rtensor64, half_spec64, rfft_layout);
                    Array<f64> rfft_rec64(total);
                    irfft_batch(half_spec64, rfft_rec64, irfft_layout);
                    TEST_CHECK(max_abs_diff_f64(rfft_rec64, rtensor64) < 1e-11);

                    Array<Complex32> half_spec32(half_total);
                    rfft_batch(rtensor32, half_spec32, rfft_layout);
                    Array<f32> rfft_rec32(total);
                    irfft_batch(half_spec32, rfft_rec32, irfft_layout);
                    TEST_CHECK(max_abs_diff_f32(rfft_rec32, rtensor32) < 2e-5);
                }
            }
        }
    }
}

void test_std_complex_interop() {
    Rng rng;
    constexpr usize kLen = 64;
    std::complex<double> buf64[kLen]{};
    std::complex<double> orig64[kLen]{};
    std::complex<float> buf32[kLen]{};
    std::complex<float> orig32[kLen]{};

    for (usize i = 0; i < kLen; ++i) {
        const double re = rng.next_f64();
        const double im = rng.next_f64();
        buf64[i] = std::complex<double>(re, im);
        orig64[i] = buf64[i];
        buf32[i] = std::complex<float>(static_cast<float>(re), static_cast<float>(im));
        orig32[i] = buf32[i];
    }

    const Complex64 converted = from_std_complex(buf64[0]);
    TEST_CHECK(to_std_complex(converted) == buf64[0]);

    fft_1d_inplace(std::span<std::complex<double>>(buf64, kLen));
    ifft_1d_inplace(std::span<std::complex<double>>(buf64, kLen));
    TEST_CHECK(max_abs_diff_c64(as_complex_ref(buf64, kLen), as_complex_ref(orig64, kLen)) < 1e-11);

    std::complex<float> spec32[kLen]{};
    std::complex<float> rec32[kLen]{};
    fft_1d(std::span<const std::complex<float>>(buf32, kLen), std::span<std::complex<float>>(spec32, kLen));
    ifft_1d(std::span<const std::complex<float>>(spec32, kLen), std::span<std::complex<float>>(rec32, kLen));
    TEST_CHECK(max_abs_diff_c32(as_complex_ref(rec32, kLen), as_complex_ref(orig32, kLen)) < 1e-5);

    Array<Complex64> keel_arr(8, Complex64(1.25, -0.5));
    auto std_span = to_std_complex_span(MutArrayRef<Complex64>(keel_arr));
    TEST_CHECK(std_span.size() == 8 && std_span[0] == std::complex<double>(1.25, -0.5));
}

} // namespace

int main() {
    test_complex_and_shift();
    test_power_of_two_and_low_level_kernels();
    test_bluestein_arbitrary_lengths_and_padding();
    test_real_fft_and_irfft();
    test_plans_arena_and_zero_exec_allocations();
    test_strided_nd_batches();
    test_std_complex_interop();

    if (g_tests_failed != 0) {
        std::fprintf(stderr, "%d test check(s) FAILED (%d passed)\n", g_tests_failed, g_tests_passed);
        return 1;
    }
    std::printf("All %d keel::fft test checks passed.\n", g_tests_passed);
    return 0;
}
