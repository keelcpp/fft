#include "keel/fft.h"

#include "fft_portable_impl.h"
#include "keel/hash_map.h"

#include <cmath>
#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace keel {

namespace fft_kernels {

void fill_twiddles_f64(MutArrayRef<Complex64> w, usize n, FftDirection dir) noexcept {
    const usize half = n >> 1;
    if (half == 0) return;
    KEEL_ASSERT(w.size() >= half);
    const f64 sign = static_cast<f64>(static_cast<i8>(dir));
    const f64 base = sign * 2.0 * detail::kPi / static_cast<f64>(n);
    for (usize k = 0; k < half; ++k) {
        const f64 angle = base * static_cast<f64>(k);
        w[k] = Complex64(std::cos(angle), std::sin(angle));
    }
}

void fill_twiddles_f32(MutArrayRef<Complex32> w, usize n, FftDirection dir) noexcept {
    const usize half = n >> 1;
    if (half == 0) return;
    KEEL_ASSERT(w.size() >= half);
    const f64 sign = static_cast<f64>(static_cast<i8>(dir));
    const f64 base = sign * 2.0 * detail::kPi / static_cast<f64>(n);
    for (usize k = 0; k < half; ++k) {
        const f64 angle = base * static_cast<f64>(k);
        w[k] = Complex32(static_cast<f32>(std::cos(angle)), static_cast<f32>(std::sin(angle)));
    }
}

} // namespace fft_kernels

namespace {

// Dedicated process-heap allocator for thread-local FFT plan caches so temporary
// `keel::set_default_allocator(...)` overrides never dangle cached twiddle/Bluestein tables.
void* cache_sys_alloc(void*, usize size, usize align) noexcept {
    if (size == 0) return nullptr;
    const usize u_align = (align < kDefaultAlign) ? kDefaultAlign : align;
#if defined(_WIN32)
    return _aligned_malloc(size, u_align);
#else
    if (u_align <= alignof(std::max_align_t)) {
        return std::malloc(size);
    }
    void* ptr = nullptr;
    if (posix_memalign(&ptr, u_align, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void* cache_sys_realloc(void* ctx, void* ptr, usize old_size, usize new_size, usize align) noexcept {
    if (!ptr) return cache_sys_alloc(ctx, new_size, align);
    if (new_size == 0) {
#if defined(_WIN32)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
        return nullptr;
    }
    const usize u_align = (align < kDefaultAlign) ? kDefaultAlign : align;
#if defined(_WIN32)
    (void)old_size;
    return _aligned_realloc(ptr, new_size, u_align);
#else
    if (u_align <= alignof(std::max_align_t)) {
        return std::realloc(ptr, new_size);
    }
    void* new_ptr = cache_sys_alloc(ctx, new_size, align);
    if (new_ptr) {
        if (old_size > 0) {
            mem_copy(new_ptr, ptr, keel::min(old_size, new_size));
        }
        std::free(ptr);
    }
    return new_ptr;
#endif
}

void cache_sys_free(void*, void* ptr, usize, usize) noexcept {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

constinit Allocator g_cache_allocator{
    .alloc = &cache_sys_alloc,
    .realloc = &cache_sys_realloc,
    .free = &cache_sys_free,
    .ctx = nullptr,
};

template <typename T>
[[nodiscard]] T norm_scale(usize n, FftDirection dir, FftNormalization norm) noexcept {
    if (n <= 1) return T(1);
    switch (norm) {
        case FftNormalization::None:
            return T(1);
        case FftNormalization::Backward:
            return (dir == FftDirection::Inverse) ? static_cast<T>(1.0 / static_cast<f64>(n)) : T(1);
        case FftNormalization::Ortho:
            return static_cast<T>(1.0 / std::sqrt(static_cast<f64>(n)));
    }
    return T(1);
}

template <typename T>
[[nodiscard]] T irfft_packed_post_scale(usize n, FftNormalization norm) noexcept {
    switch (norm) {
        case FftNormalization::None:
            return T(2);
        case FftNormalization::Backward:
            return static_cast<T>(2.0 / static_cast<f64>(n));
        case FftNormalization::Ortho:
            return static_cast<T>(2.0 / std::sqrt(static_cast<f64>(n)));
    }
    return static_cast<T>(2.0 / static_cast<f64>(n));
}

template <typename Cplx>
void fill_twiddles_generic(MutArrayRef<Cplx> w, usize n, FftDirection dir) noexcept {
    if constexpr (IsSame<Cplx, Complex64>) {
        fft_kernels::fill_twiddles_f64(w, n, dir);
    } else {
        fft_kernels::fill_twiddles_f32(w, n, dir);
    }
}

template <typename T, typename Cplx>
struct PlanView {
    usize n = 0;
    usize conv_m = 0;
    bool is_pow2 = false;
    const Cplx* w_fwd = nullptr;
    const Cplx* w_half = nullptr;
    const Cplx* chirp = nullptr;
    const Cplx* v_freq = nullptr;
};

template <typename T, typename Cplx>
void init_plan_arrays(usize n, Allocator* alloc, usize& out_conv_m, bool& out_is_pow2, Array<Cplx>& out_twiddles,
                      Array<Cplx>& out_half_twiddles, Array<Cplx>& out_chirp, Array<Cplx>& out_v_freq) {
    Allocator* dst_alloc = alloc ? alloc : default_allocator();
    out_is_pow2 = detail::is_pow2(n);

    if (out_is_pow2) {
        out_conv_m = 0;
        const usize half = n >> 1;
        out_twiddles = Array<Cplx>(half, dst_alloc);
        if (half > 0) {
            fill_twiddles_generic<Cplx>(out_twiddles, n, FftDirection::Forward);
        }
        if (n >= 4) {
            const usize quarter = n >> 2;
            out_half_twiddles = Array<Cplx>(quarter, dst_alloc);
            for (usize k = 0; k < quarter; ++k) {
                out_half_twiddles[k] = out_twiddles[2 * k];
            }
        } else {
            out_half_twiddles = Array<Cplx>(dst_alloc);
        }
        out_chirp = Array<Cplx>(dst_alloc);
        out_v_freq = Array<Cplx>(dst_alloc);
        return;
    }

    // Bluestein chirp-z precomputation for arbitrary non-power-of-2 length `n`.
    const usize m = detail::next_pow2(2 * n - 1);
    out_conv_m = m;

    out_twiddles = Array<Cplx>(m >> 1, dst_alloc);
    fill_twiddles_generic<Cplx>(out_twiddles, m, FftDirection::Forward);
    out_half_twiddles = Array<Cplx>(dst_alloc);

    out_chirp = Array<Cplx>(n, dst_alloc);
    const f64 inv_n = 1.0 / static_cast<f64>(n);
    const u64 two_n = 2ULL * static_cast<u64>(n);
    for (usize k = 0; k < n; ++k) {
        const u64 mod = (static_cast<u64>(k) * static_cast<u64>(k)) % two_n;
        const f64 angle = -detail::kPi * static_cast<f64>(mod) * inv_n;
        out_chirp[k] = Cplx(static_cast<T>(std::cos(angle)), static_cast<T>(std::sin(angle)));
    }

    out_v_freq = Array<Cplx>(m, Cplx(T(0), T(0)), dst_alloc);
    out_v_freq[0] = conj(out_chirp[0]);
    for (usize k = 1; k < n; ++k) {
        const Cplx c = conj(out_chirp[k]);
        out_v_freq[k] = c;
        out_v_freq[m - k] = c;
    }
    fft_kernels::fft_radix2_inplace(out_v_freq.data(), m, out_twiddles.data());
}

template <typename T, typename Cplx>
void exec_c2c_core(const PlanView<T, Cplx>& p, Cplx* buf, FftDirection dir, FftNormalization norm,
                   Cplx* bluestein_work) noexcept {
    if (p.n <= 1) return;

    const T scale = norm_scale<T>(p.n, dir, norm);

    if (p.is_pow2) {
        if (dir == FftDirection::Forward) {
            fft_kernels::fft_radix2_inplace(buf, p.n, p.w_fwd);
            if (scale != T(1)) {
                for (usize k = 0; k < p.n; ++k) {
                    buf[k] *= scale;
                }
            }
        } else {
            for (usize k = 0; k < p.n; ++k) {
                buf[k].im = -buf[k].im;
            }
            fft_kernels::fft_radix2_inplace(buf, p.n, p.w_fwd);
            if (scale != T(1)) {
                const T neg_scale = -scale;
                for (usize k = 0; k < p.n; ++k) {
                    buf[k] = Cplx(buf[k].re * scale, buf[k].im * neg_scale);
                }
            } else {
                for (usize k = 0; k < p.n; ++k) {
                    buf[k].im = -buf[k].im;
                }
            }
        }
        return;
    }

    // Bluestein chirp-z convolution of power-of-2 length `m = p.conv_m`.
    const usize n = p.n;
    const usize m = p.conv_m;
    const Cplx* chirp = p.chirp;
    const Cplx* v_freq = p.v_freq;

    auto run_with_work = [&](Cplx* work) noexcept {
        if (dir == FftDirection::Forward) {
            for (usize k = 0; k < n; ++k) {
                work[k] = buf[k] * chirp[k];
            }
        } else {
            for (usize k = 0; k < n; ++k) {
                work[k] = conj(buf[k]) * chirp[k];
            }
        }
        mem_zero(work + n, (m - n) * sizeof(Cplx));

        fft_kernels::fft_radix2_inplace(work, m, p.w_fwd);
        for (usize i = 0; i < m; ++i) {
            work[i] *= v_freq[i];
        }
        for (usize i = 0; i < m; ++i) {
            work[i].im = -work[i].im;
        }
        fft_kernels::fft_radix2_inplace(work, m, p.w_fwd);

        const T combined_scale = static_cast<T>(static_cast<f64>(scale) / static_cast<f64>(m));
        const T neg_combined = -combined_scale;
        if (dir == FftDirection::Forward) {
            for (usize k = 0; k < n; ++k) {
                const Cplx ck(work[k].re * combined_scale, work[k].im * neg_combined);
                buf[k] = ck * chirp[k];
            }
        } else {
            for (usize k = 0; k < n; ++k) {
                const Cplx ck(work[k].re * combined_scale, work[k].im * neg_combined);
                buf[k] = conj(ck * chirp[k]);
            }
        }
    };

    if (bluestein_work) {
        run_with_work(bluestein_work);
    } else {
        detail::ScratchScope scratch;
        Cplx* work = scratch.alloc<Cplx>(m);
        run_with_work(work);
    }
}

template <typename T, typename Cplx>
void exec_rfft_slice(const PlanView<T, Cplx>& p, const T* src, usize use_len, usize src_stride, Cplx* dst,
                     usize dst_stride, bool full_spectrum, FftNormalization norm, T* rfft_re, T* rfft_im,
                     Cplx* cbuf, Cplx* bluestein_work) noexcept {
    const usize n = p.n;
    if (p.is_pow2 && n >= 4) {
        const usize half = n >> 1;
        const usize full_pairs = use_len >> 1;
        if (src_stride == 1) {
            for (usize m = 0; m < full_pairs; ++m) {
                rfft_re[m] = src[2 * m];
                rfft_im[m] = src[2 * m + 1];
            }
        } else {
            for (usize m = 0; m < full_pairs; ++m) {
                rfft_re[m] = src[(2 * m) * src_stride];
                rfft_im[m] = src[(2 * m + 1) * src_stride];
            }
        }
        usize zero_from = full_pairs;
        if ((use_len & 1u) != 0u) {
            rfft_re[full_pairs] = src[(2 * full_pairs) * src_stride];
            rfft_im[full_pairs] = T(0);
            zero_from = full_pairs + 1;
        }
        if (zero_from < half) {
            mem_zero(rfft_re + zero_from, (half - zero_from) * sizeof(T));
            mem_zero(rfft_im + zero_from, (half - zero_from) * sizeof(T));
        }

        fft_kernels::fft_soa_stages_inplace(rfft_re, rfft_im, half, p.w_half);
        const T scale = norm_scale<T>(n, FftDirection::Forward, norm);

        if (full_spectrum) {
            if (dst_stride == 1) {
                detail::rfft_twist_full_unit_stride(rfft_re, rfft_im, n, p.w_fwd, scale, dst);
            } else {
                detail::portable_rfft_twist_full(rfft_re, rfft_im, n, p.w_fwd, scale, dst, dst_stride);
            }
        } else {
            if (dst_stride == 1) {
                detail::rfft_twist_half_unit_stride(rfft_re, rfft_im, n, p.w_fwd, scale, dst);
            } else {
                detail::portable_rfft_twist_half(rfft_re, rfft_im, n, p.w_fwd, scale, dst, dst_stride);
            }
        }
        return;
    }

    // General path (handles odd/non-pow2 N, tiny N < 4, and zero-padding/truncation).
    if (src_stride == 1) {
        for (usize k = 0; k < use_len; ++k) {
            cbuf[k] = Cplx(src[k], T(0));
        }
    } else {
        for (usize k = 0; k < use_len; ++k) {
            cbuf[k] = Cplx(src[k * src_stride], T(0));
        }
    }
    if (use_len < n) {
        mem_zero(cbuf + use_len, (n - use_len) * sizeof(Cplx));
    }

    exec_c2c_core(p, cbuf, FftDirection::Forward, norm, bluestein_work);

    // Enforce exact Hermitian real-signal invariants on DC, Nyquist, and conjugate pairs.
    cbuf[0].im = T(0);
    const usize half = n >> 1;
    if ((n & 1u) == 0u) {
        cbuf[half].im = T(0);
    }

    if (full_spectrum) {
        const usize mirror_hi = (n - 1) >> 1;
        for (usize k = 1; k <= mirror_hi; ++k) {
            cbuf[n - k] = conj(cbuf[k]);
        }
        if (dst_stride == 1) {
            mem_copy(dst, cbuf, n * sizeof(Cplx));
        } else {
            for (usize k = 0; k < n; ++k) {
                dst[k * dst_stride] = cbuf[k];
            }
        }
    } else {
        const usize out_bins = half + 1;
        if (dst_stride == 1) {
            mem_copy(dst, cbuf, out_bins * sizeof(Cplx));
        } else {
            for (usize k = 0; k < out_bins; ++k) {
                dst[k * dst_stride] = cbuf[k];
            }
        }
    }
}

template <typename T, typename Cplx>
void exec_irfft_slice(const PlanView<T, Cplx>& p, const Cplx* half_src, usize src_stride, T* real_dst,
                      usize dst_stride, FftNormalization norm, T* z_re, T* z_im, Cplx* cbuf,
                      Cplx* bluestein_work) noexcept {
    const usize n = p.n;
    if (n == 1) {
        real_dst[0] = half_src[0].re * norm_scale<T>(1, FftDirection::Inverse, norm);
        return;
    }
    if (n == 2) {
        const T x0 = half_src[0].re;
        const T x1 = half_src[src_stride].re;
        const T s = norm_scale<T>(2, FftDirection::Inverse, norm);
        real_dst[0] = (x0 + x1) * s;
        real_dst[dst_stride] = (x0 - x1) * s;
        return;
    }

    if (p.is_pow2 && n >= 4) {
        const usize half = n >> 1;
        if (src_stride == 1) {
            detail::irfft_untwist_conj_unit_stride(half_src, n, p.w_fwd, z_re, z_im);
        } else {
            detail::portable_irfft_untwist_conj(half_src, src_stride, n, p.w_fwd, z_re, z_im);
        }

        fft_kernels::fft_soa_stages_inplace(z_re, z_im, half, p.w_half);

        const T post_scale = irfft_packed_post_scale<T>(n, norm);
        if (dst_stride == 1) {
            detail::irfft_store_interleaved_unit_stride(z_re, z_im, half, post_scale, real_dst);
        } else {
            detail::portable_irfft_store_interleaved(z_re, z_im, half, post_scale, real_dst, dst_stride);
        }
        return;
    }

    // General non-power-of-2 IRFFT via Hermitian completion + Bluestein inverse FFT.
    cbuf[0] = Cplx(half_src[0].re, T(0));
    const usize limit = (n - 1) >> 1;
    for (usize k = 1; k <= limit; ++k) {
        const Cplx v = half_src[k * src_stride];
        cbuf[k] = v;
        cbuf[n - k] = conj(v);
    }
    if ((n & 1u) == 0u) {
        cbuf[n >> 1] = Cplx(half_src[(n >> 1) * src_stride].re, T(0));
    }

    exec_c2c_core(p, cbuf, FftDirection::Inverse, norm, bluestein_work);

    if (dst_stride == 1) {
        for (usize k = 0; k < n; ++k) {
            real_dst[k] = cbuf[k].re;
        }
    } else {
        for (usize k = 0; k < n; ++k) {
            real_dst[k * dst_stride] = cbuf[k].re;
        }
    }
}

template <typename T, typename Cplx>
void exec_plan_1d(const PlanView<T, Cplx>& p, ArrayRef<Cplx> src, MutArrayRef<Cplx> dst, FftDirection dir,
                  FftNormalization norm) noexcept {
    KEEL_ASSERT(p.n > 0 && dst.size() == p.n);
    const usize use_len = keel::min(src.size(), p.n);
    Cplx* out = dst.data();
    if (use_len > 0 && out != src.data()) {
        mem_move(out, src.data(), use_len * sizeof(Cplx));
    }
    if (use_len < p.n) {
        mem_zero(out + use_len, (p.n - use_len) * sizeof(Cplx));
    }
    exec_c2c_core<T, Cplx>(p, out, dir, norm, nullptr);
}

template <typename T, typename Cplx>
void exec_plan_rfft_1d(const PlanView<T, Cplx>& p, ArrayRef<T> real_src, MutArrayRef<Cplx> dst, bool full_spectrum,
                       FftNormalization norm) noexcept {
    const usize n = p.n;
    const usize expected_out = full_spectrum ? n : ((n >> 1) + 1);
    KEEL_ASSERT(n > 0 && dst.size() == expected_out);

    const usize use_len = keel::min(real_src.size(), n);
    detail::ScratchScope scratch;
    if (p.is_pow2 && n >= 4) {
        const usize half = n >> 1;
        T* rfft_re = scratch.alloc<T>(half);
        T* rfft_im = scratch.alloc<T>(half);
        exec_rfft_slice<T, Cplx>(p, real_src.data(), use_len, 1, dst.data(), 1, full_spectrum, norm, rfft_re, rfft_im,
                                 nullptr, nullptr);
    } else {
        Cplx* cbuf = scratch.alloc<Cplx>(n);
        Cplx* bwork = p.is_pow2 ? nullptr : scratch.alloc<Cplx>(p.conv_m);
        exec_rfft_slice<T, Cplx>(p, real_src.data(), use_len, 1, dst.data(), 1, full_spectrum, norm, nullptr, nullptr,
                                 cbuf, bwork);
    }
}

template <typename T, typename Cplx>
void exec_plan_irfft_1d(const PlanView<T, Cplx>& p, ArrayRef<Cplx> half_src, MutArrayRef<T> real_dst,
                        FftNormalization norm) noexcept {
    const usize n = p.n;
    KEEL_ASSERT(n > 0 && half_src.size() == ((n >> 1) + 1) && real_dst.size() == n);

    if (n <= 2) {
        exec_irfft_slice<T, Cplx>(p, half_src.data(), 1, real_dst.data(), 1, norm, nullptr, nullptr, nullptr, nullptr);
        return;
    }

    detail::ScratchScope scratch;
    if (p.is_pow2) {
        const usize half = n >> 1;
        T* z_re = scratch.alloc<T>(half);
        T* z_im = scratch.alloc<T>(half);
        exec_irfft_slice<T, Cplx>(p, half_src.data(), 1, real_dst.data(), 1, norm, z_re, z_im, nullptr, nullptr);
    } else {
        Cplx* cbuf = scratch.alloc<Cplx>(n);
        Cplx* bwork = scratch.alloc<Cplx>(p.conv_m);
        exec_irfft_slice<T, Cplx>(p, half_src.data(), 1, real_dst.data(), 1, norm, nullptr, nullptr, cbuf, bwork);
    }
}

template <typename T, typename Cplx>
void exec_plan_symmetric_ifft_1d(const PlanView<T, Cplx>& p, ArrayRef<Cplx> src, MutArrayRef<T> real_dst,
                                 FftNormalization norm) noexcept {
    const usize n = p.n;
    KEEL_ASSERT(n > 0 && real_dst.size() == n);
    const usize half_bins = (n >> 1) + 1;

    if (src.size() >= half_bins) {
        exec_plan_irfft_1d(p, src.slice(0, half_bins), real_dst, norm);
        return;
    }

    detail::ScratchScope scratch;
    Cplx* half_buf = scratch.alloc<Cplx>(half_bins);
    if (!src.empty()) {
        mem_copy(half_buf, src.data(), src.size() * sizeof(Cplx));
    }
    mem_zero(half_buf + src.size(), (half_bins - src.size()) * sizeof(Cplx));
    exec_plan_irfft_1d(p, ArrayRef<Cplx>(half_buf, half_bins), real_dst, norm);
}

[[maybe_unused]] void validate_batch_bounds(usize src_size, usize dst_size, usize in_slice_len, usize out_slice_len,
                                            const FftBatchLayout& layout) noexcept {
    KEEL_ASSERT(layout.elem_stride > 0);
    const usize dst_stride = layout.effective_dst_stride();
    KEEL_ASSERT(dst_stride > 0);
    if (in_slice_len > 0) {
        const usize max_src = (layout.outer_count - 1) * layout.outer_src_step +
                              (layout.inner_count - 1) * layout.inner_step +
                              (in_slice_len - 1) * layout.elem_stride;
        KEEL_ASSERT(max_src < src_size);
    }
    if (out_slice_len > 0) {
        const usize max_dst = (layout.outer_count - 1) * layout.outer_dst_step +
                              (layout.inner_count - 1) * layout.inner_step +
                              (out_slice_len - 1) * dst_stride;
        KEEL_ASSERT(max_dst < dst_size);
    }
}

template <typename T, typename Cplx>
void exec_plan_batch_c2c(const PlanView<T, Cplx>& p, ArrayRef<Cplx> src, MutArrayRef<Cplx> dst,
                         const FftBatchLayout& layout, FftDirection dir, FftNormalization norm) noexcept {
    if (layout.outer_count == 0 || layout.inner_count == 0) return;
    const usize n = p.n;
    KEEL_ASSERT(n > 0 && layout.effective_fft_len() == n);

    const usize use_len = keel::min(layout.input_len, n);
    validate_batch_bounds(src.size(), dst.size(), use_len, n, layout);

    const usize src_stride = layout.elem_stride;
    const usize dst_stride = layout.effective_dst_stride();
    const Cplx* src_ptr = src.data();
    Cplx* dst_ptr = dst.data();

    detail::ScratchScope scratch;
    Cplx* slice_buf = scratch.alloc<Cplx>(n);
    Cplx* bwork = p.is_pow2 ? nullptr : scratch.alloc<Cplx>(p.conv_m);

    for (usize o = 0; o < layout.outer_count; ++o) {
        const usize outer_in = o * layout.outer_src_step;
        const usize outer_out = o * layout.outer_dst_step;
        for (usize i = 0; i < layout.inner_count; ++i) {
            const Cplx* in_s = src_ptr + outer_in + i * layout.inner_step;
            Cplx* out_s = dst_ptr + outer_out + i * layout.inner_step;

            if (src_stride == 1) {
                if (use_len > 0) {
                    mem_copy(slice_buf, in_s, use_len * sizeof(Cplx));
                }
            } else {
                for (usize k = 0; k < use_len; ++k) {
                    slice_buf[k] = in_s[k * src_stride];
                }
            }
            if (use_len < n) {
                mem_zero(slice_buf + use_len, (n - use_len) * sizeof(Cplx));
            }

            exec_c2c_core(p, slice_buf, dir, norm, bwork);

            if (dst_stride == 1) {
                mem_copy(out_s, slice_buf, n * sizeof(Cplx));
            } else {
                for (usize k = 0; k < n; ++k) {
                    out_s[k * dst_stride] = slice_buf[k];
                }
            }
        }
    }
}

template <typename T, typename Cplx>
void exec_plan_batch_rfft(const PlanView<T, Cplx>& p, ArrayRef<T> src, MutArrayRef<Cplx> dst,
                          const FftBatchLayout& layout, bool full_spectrum, FftNormalization norm) noexcept {
    if (layout.outer_count == 0 || layout.inner_count == 0) return;
    const usize n = p.n;
    KEEL_ASSERT(n > 0 && layout.effective_fft_len() == n);

    const usize use_len = keel::min(layout.input_len, n);
    const usize out_len = full_spectrum ? n : ((n >> 1) + 1);
    validate_batch_bounds(src.size(), dst.size(), use_len, out_len, layout);

    const usize src_stride = layout.elem_stride;
    const usize dst_stride = layout.effective_dst_stride();
    const T* src_ptr = src.data();
    Cplx* dst_ptr = dst.data();

    detail::ScratchScope scratch;
    const bool packed_fast = p.is_pow2 && n >= 4;
    T* rfft_re = packed_fast ? scratch.alloc<T>(n >> 1) : nullptr;
    T* rfft_im = packed_fast ? scratch.alloc<T>(n >> 1) : nullptr;
    Cplx* cbuf = packed_fast ? nullptr : scratch.alloc<Cplx>(n);
    Cplx* bwork = (!packed_fast && !p.is_pow2) ? scratch.alloc<Cplx>(p.conv_m) : nullptr;

    for (usize o = 0; o < layout.outer_count; ++o) {
        const usize outer_in = o * layout.outer_src_step;
        const usize outer_out = o * layout.outer_dst_step;
        for (usize i = 0; i < layout.inner_count; ++i) {
            exec_rfft_slice(p, src_ptr + outer_in + i * layout.inner_step, use_len, src_stride,
                            dst_ptr + outer_out + i * layout.inner_step, dst_stride, full_spectrum, norm, rfft_re,
                            rfft_im, cbuf, bwork);
        }
    }
}

template <typename T, typename Cplx>
void exec_plan_batch_irfft(const PlanView<T, Cplx>& p, ArrayRef<Cplx> half_src, MutArrayRef<T> dst,
                           const FftBatchLayout& layout, FftNormalization norm) noexcept {
    if (layout.outer_count == 0 || layout.inner_count == 0) return;
    const usize n = p.n;
    const usize half_bins = (n >> 1) + 1;
    const usize req_fft = (layout.fft_len > 0) ? layout.fft_len : n;
    KEEL_ASSERT(n > 0 && req_fft == n);
    KEEL_ASSERT(layout.input_len == 0 || layout.input_len == half_bins);

    validate_batch_bounds(half_src.size(), dst.size(), half_bins, n, layout);

    const usize src_stride = layout.elem_stride;
    const usize dst_stride = layout.effective_dst_stride();
    const Cplx* src_ptr = half_src.data();
    T* dst_ptr = dst.data();

    detail::ScratchScope scratch;
    const bool packed_fast = p.is_pow2 && n >= 4;
    T* z_re = packed_fast ? scratch.alloc<T>(n >> 1) : nullptr;
    T* z_im = packed_fast ? scratch.alloc<T>(n >> 1) : nullptr;
    Cplx* cbuf = (!packed_fast && n > 2) ? scratch.alloc<Cplx>(n) : nullptr;
    Cplx* bwork = (!packed_fast && n > 2) ? scratch.alloc<Cplx>(p.conv_m) : nullptr;

    for (usize o = 0; o < layout.outer_count; ++o) {
        const usize outer_in = o * layout.outer_src_step;
        const usize outer_out = o * layout.outer_dst_step;
        for (usize i = 0; i < layout.inner_count; ++i) {
            exec_irfft_slice(p, src_ptr + outer_in + i * layout.inner_step, src_stride,
                             dst_ptr + outer_out + i * layout.inner_step, dst_stride, norm, z_re, z_im, cbuf, bwork);
        }
    }
}

template <typename T, typename Cplx>
void exec_plan_batch_symmetric_ifft(const PlanView<T, Cplx>& p, ArrayRef<Cplx> src, MutArrayRef<T> dst,
                                    const FftBatchLayout& layout, FftNormalization norm) noexcept {
    if (layout.outer_count == 0 || layout.inner_count == 0) return;
    const usize n = p.n;
    KEEL_ASSERT(n > 0 && layout.effective_fft_len() == n);

    const usize half_bins = (n >> 1) + 1;
    const usize copy_bins = keel::min(layout.input_len, half_bins);
    validate_batch_bounds(src.size(), dst.size(), copy_bins, n, layout);

    const usize src_stride = layout.elem_stride;
    const usize dst_stride = layout.effective_dst_stride();
    const Cplx* src_ptr = src.data();
    T* dst_ptr = dst.data();

    detail::ScratchScope scratch;
    Cplx* half_buf = (copy_bins < half_bins) ? scratch.alloc<Cplx>(half_bins) : nullptr;
    const bool packed_fast = p.is_pow2 && n >= 4;
    T* z_re = packed_fast ? scratch.alloc<T>(n >> 1) : nullptr;
    T* z_im = packed_fast ? scratch.alloc<T>(n >> 1) : nullptr;
    Cplx* cbuf = (!packed_fast && n > 2) ? scratch.alloc<Cplx>(n) : nullptr;
    Cplx* bwork = (!packed_fast && n > 2) ? scratch.alloc<Cplx>(p.conv_m) : nullptr;

    for (usize o = 0; o < layout.outer_count; ++o) {
        const usize outer_in = o * layout.outer_src_step;
        const usize outer_out = o * layout.outer_dst_step;
        for (usize i = 0; i < layout.inner_count; ++i) {
            const Cplx* in_s = src_ptr + outer_in + i * layout.inner_step;
            T* out_s = dst_ptr + outer_out + i * layout.inner_step;

            if (copy_bins >= half_bins) {
                exec_irfft_slice(p, in_s, src_stride, out_s, dst_stride, norm, z_re, z_im, cbuf, bwork);
            } else {
                if (src_stride == 1) {
                    if (copy_bins > 0) {
                        mem_copy(half_buf, in_s, copy_bins * sizeof(Cplx));
                    }
                } else {
                    for (usize k = 0; k < copy_bins; ++k) {
                        half_buf[k] = in_s[k * src_stride];
                    }
                }
                mem_zero(half_buf + copy_bins, (half_bins - copy_bins) * sizeof(Cplx));
                exec_irfft_slice(p, half_buf, 1, out_s, dst_stride, norm, z_re, z_im, cbuf, bwork);
            }
        }
    }
}

[[nodiscard]] PlanView<f64, Complex64> make_view(const FftPlan1D& plan, usize conv_m, const Complex64* w_half,
                                                 const Complex64* chirp, const Complex64* v_freq) noexcept {
    return PlanView<f64, Complex64>{
        .n = plan.size(),
        .conv_m = conv_m,
        .is_pow2 = plan.is_power_of_two(),
        .w_fwd = plan.twiddles().data(),
        .w_half = w_half,
        .chirp = chirp,
        .v_freq = v_freq,
    };
}

[[nodiscard]] PlanView<f32, Complex32> make_view(const FftPlan1DF32& plan, usize conv_m, const Complex32* w_half,
                                                 const Complex32* chirp, const Complex32* v_freq) noexcept {
    return PlanView<f32, Complex32>{
        .n = plan.size(),
        .conv_m = conv_m,
        .is_pow2 = plan.is_power_of_two(),
        .w_fwd = plan.twiddles().data(),
        .w_half = w_half,
        .chirp = chirp,
        .v_freq = v_freq,
    };
}

[[nodiscard]] const FftPlan1D& cached_plan_f64(usize n) noexcept {
    thread_local HashMap<usize, FftPlan1D> cache(&g_cache_allocator);
    if (const FftPlan1D* existing = cache.get(n)) {
        return *existing;
    }
    Result<FftPlan1D> created = FftPlan1D::create(n, &g_cache_allocator);
    KEEL_ASSERT(created.is_ok());
    return *cache.try_insert(n, keel::move(*created)).value;
}

[[nodiscard]] const FftPlan1DF32& cached_plan_f32(usize n) noexcept {
    thread_local HashMap<usize, FftPlan1DF32> cache(&g_cache_allocator);
    if (const FftPlan1DF32* existing = cache.get(n)) {
        return *existing;
    }
    Result<FftPlan1DF32> created = FftPlan1DF32::create(n, &g_cache_allocator);
    KEEL_ASSERT(created.is_ok());
    return *cache.try_insert(n, keel::move(*created)).value;
}

[[nodiscard]] usize infer_irfft_batch_len(const FftBatchLayout& layout) noexcept {
    if (layout.fft_len > 0) {
        return layout.fft_len;
    }
    if (layout.input_len == 0) {
        return 0;
    }
    return (layout.input_len > 1) ? (2 * (layout.input_len - 1)) : 1;
}

} // namespace

// ============================================================================
// FftPlan1D (f64)
// ============================================================================

Result<FftPlan1D> FftPlan1D::create(usize n, Allocator* alloc) {
    if (n == 0) {
        return ErrorCode::InvalidArgument;
    }
    FftPlan1D plan;
    plan.m_n = n;
    init_plan_arrays<f64, Complex64>(n, alloc, plan.m_conv_m, plan.m_is_pow2, plan.m_twiddles, plan.m_half_twiddles,
                                     plan.m_chirp, plan.m_v_freq);
    return plan;
}

FftPlan1D FftPlan1D::clone(Allocator* alloc) const {
    Allocator* dst_alloc = alloc ? alloc : allocator();
    FftPlan1D copy;
    copy.m_n = m_n;
    copy.m_conv_m = m_conv_m;
    copy.m_is_pow2 = m_is_pow2;
    copy.m_twiddles = m_twiddles.clone(dst_alloc);
    copy.m_half_twiddles = m_half_twiddles.clone(dst_alloc);
    copy.m_chirp = m_chirp.clone(dst_alloc);
    copy.m_v_freq = m_v_freq.clone(dst_alloc);
    return copy;
}

void FftPlan1D::exec_inplace(MutArrayRef<Complex64> buf, FftDirection dir, FftNormalization norm) const noexcept {
    KEEL_ASSERT(m_n > 0 && buf.size() == m_n);
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_c2c_core<f64, Complex64>(view, buf.data(), dir, norm, nullptr);
}

void FftPlan1D::exec(ArrayRef<Complex64> src, MutArrayRef<Complex64> dst, FftDirection dir,
                     FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_1d(view, src, dst, dir, norm);
}

void FftPlan1D::exec_rfft(ArrayRef<f64> real_src, MutArrayRef<Complex64> half_spectrum_dst,
                          FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_rfft_1d(view, real_src, half_spectrum_dst, false, norm);
}

void FftPlan1D::exec_real_to_complex(ArrayRef<f64> real_src, MutArrayRef<Complex64> full_dst,
                                     FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_rfft_1d(view, real_src, full_dst, true, norm);
}

void FftPlan1D::exec_irfft(ArrayRef<Complex64> half_spectrum_src, MutArrayRef<f64> real_dst,
                           FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_irfft_1d(view, half_spectrum_src, real_dst, norm);
}

void FftPlan1D::exec_symmetric_ifft(ArrayRef<Complex64> src, MutArrayRef<f64> real_dst,
                                    FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_symmetric_ifft_1d(view, src, real_dst, norm);
}

void FftPlan1D::exec_batch(ArrayRef<Complex64> src, MutArrayRef<Complex64> dst, const FftBatchLayout& layout,
                           FftDirection dir, FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_c2c(view, src, dst, layout, dir, norm);
}

void FftPlan1D::exec_batch_real(ArrayRef<f64> src, MutArrayRef<Complex64> dst, const FftBatchLayout& layout,
                                FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_rfft(view, src, dst, layout, true, norm);
}

void FftPlan1D::exec_batch_rfft(ArrayRef<f64> src, MutArrayRef<Complex64> half_dst, const FftBatchLayout& layout,
                                FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_rfft(view, src, half_dst, layout, false, norm);
}

void FftPlan1D::exec_batch_irfft(ArrayRef<Complex64> half_src, MutArrayRef<f64> dst, const FftBatchLayout& layout,
                                 FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_irfft(view, half_src, dst, layout, norm);
}

void FftPlan1D::exec_batch_symmetric_ifft(ArrayRef<Complex64> src, MutArrayRef<f64> dst, const FftBatchLayout& layout,
                                          FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_symmetric_ifft(view, src, dst, layout, norm);
}

// ============================================================================
// FftPlan1DF32 (f32)
// ============================================================================

Result<FftPlan1DF32> FftPlan1DF32::create(usize n, Allocator* alloc) {
    if (n == 0) {
        return ErrorCode::InvalidArgument;
    }
    FftPlan1DF32 plan;
    plan.m_n = n;
    init_plan_arrays<f32, Complex32>(n, alloc, plan.m_conv_m, plan.m_is_pow2, plan.m_twiddles, plan.m_half_twiddles,
                                     plan.m_chirp, plan.m_v_freq);
    return plan;
}

FftPlan1DF32 FftPlan1DF32::clone(Allocator* alloc) const {
    Allocator* dst_alloc = alloc ? alloc : allocator();
    FftPlan1DF32 copy;
    copy.m_n = m_n;
    copy.m_conv_m = m_conv_m;
    copy.m_is_pow2 = m_is_pow2;
    copy.m_twiddles = m_twiddles.clone(dst_alloc);
    copy.m_half_twiddles = m_half_twiddles.clone(dst_alloc);
    copy.m_chirp = m_chirp.clone(dst_alloc);
    copy.m_v_freq = m_v_freq.clone(dst_alloc);
    return copy;
}

void FftPlan1DF32::exec_inplace(MutArrayRef<Complex32> buf, FftDirection dir, FftNormalization norm) const noexcept {
    KEEL_ASSERT(m_n > 0 && buf.size() == m_n);
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_c2c_core<f32, Complex32>(view, buf.data(), dir, norm, nullptr);
}

void FftPlan1DF32::exec(ArrayRef<Complex32> src, MutArrayRef<Complex32> dst, FftDirection dir,
                        FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_1d(view, src, dst, dir, norm);
}

void FftPlan1DF32::exec_rfft(ArrayRef<f32> real_src, MutArrayRef<Complex32> half_spectrum_dst,
                             FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_rfft_1d(view, real_src, half_spectrum_dst, false, norm);
}

void FftPlan1DF32::exec_real_to_complex(ArrayRef<f32> real_src, MutArrayRef<Complex32> full_dst,
                                        FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_rfft_1d(view, real_src, full_dst, true, norm);
}

void FftPlan1DF32::exec_irfft(ArrayRef<Complex32> half_spectrum_src, MutArrayRef<f32> real_dst,
                              FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_irfft_1d(view, half_spectrum_src, real_dst, norm);
}

void FftPlan1DF32::exec_symmetric_ifft(ArrayRef<Complex32> src, MutArrayRef<f32> real_dst,
                                       FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_symmetric_ifft_1d(view, src, real_dst, norm);
}

void FftPlan1DF32::exec_batch(ArrayRef<Complex32> src, MutArrayRef<Complex32> dst, const FftBatchLayout& layout,
                              FftDirection dir, FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_c2c(view, src, dst, layout, dir, norm);
}

void FftPlan1DF32::exec_batch_real(ArrayRef<f32> src, MutArrayRef<Complex32> dst, const FftBatchLayout& layout,
                                   FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_rfft(view, src, dst, layout, true, norm);
}

void FftPlan1DF32::exec_batch_rfft(ArrayRef<f32> src, MutArrayRef<Complex32> half_dst, const FftBatchLayout& layout,
                                   FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_rfft(view, src, half_dst, layout, false, norm);
}

void FftPlan1DF32::exec_batch_irfft(ArrayRef<Complex32> half_src, MutArrayRef<f32> dst, const FftBatchLayout& layout,
                                    FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_irfft(view, half_src, dst, layout, norm);
}

void FftPlan1DF32::exec_batch_symmetric_ifft(ArrayRef<Complex32> src, MutArrayRef<f32> dst,
                                             const FftBatchLayout& layout, FftNormalization norm) const noexcept {
    const auto view = make_view(*this, m_conv_m, m_half_twiddles.data(), m_chirp.data(), m_v_freq.data());
    exec_plan_batch_symmetric_ifft(view, src, dst, layout, norm);
}

// ============================================================================
// High-Level Stateless / Thread-Cached 1-D Transforms (f64 & f32)
// ============================================================================

void fft_1d_inplace(MutArrayRef<Complex64> buf, FftNormalization norm) noexcept {
    if (buf.size() <= 1) return;
    cached_plan_f64(buf.size()).exec_inplace(buf, FftDirection::Forward, norm);
}

void ifft_1d_inplace(MutArrayRef<Complex64> buf, FftNormalization norm) noexcept {
    if (buf.size() <= 1) return;
    cached_plan_f64(buf.size()).exec_inplace(buf, FftDirection::Inverse, norm);
}

void fft_1d(ArrayRef<Complex64> src, MutArrayRef<Complex64> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f64(dst.size()).exec(src, dst, FftDirection::Forward, norm);
}

void ifft_1d(ArrayRef<Complex64> src, MutArrayRef<Complex64> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f64(dst.size()).exec(src, dst, FftDirection::Inverse, norm);
}

void fft_1d_real(ArrayRef<f64> src, MutArrayRef<Complex64> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f64(dst.size()).exec_real_to_complex(src, dst, norm);
}

void rfft_1d(ArrayRef<f64> src, MutArrayRef<Complex64> half_dst, FftNormalization norm) noexcept {
    if (src.empty()) {
        KEEL_ASSERT(half_dst.empty());
        return;
    }
    cached_plan_f64(src.size()).exec_rfft(src, half_dst, norm);
}

void irfft_1d(ArrayRef<Complex64> half_src, MutArrayRef<f64> dst, usize n, FftNormalization norm) noexcept {
    if (n == 0) {
        KEEL_ASSERT(dst.empty());
        return;
    }
    KEEL_ASSERT(dst.size() == n);
    cached_plan_f64(n).exec_irfft(half_src, dst, norm);
}

void ifft_1d_symmetric(ArrayRef<Complex64> src, MutArrayRef<f64> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f64(dst.size()).exec_symmetric_ifft(src, dst, norm);
}

void fft_1d_inplace(MutArrayRef<Complex32> buf, FftNormalization norm) noexcept {
    if (buf.size() <= 1) return;
    cached_plan_f32(buf.size()).exec_inplace(buf, FftDirection::Forward, norm);
}

void ifft_1d_inplace(MutArrayRef<Complex32> buf, FftNormalization norm) noexcept {
    if (buf.size() <= 1) return;
    cached_plan_f32(buf.size()).exec_inplace(buf, FftDirection::Inverse, norm);
}

void fft_1d(ArrayRef<Complex32> src, MutArrayRef<Complex32> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f32(dst.size()).exec(src, dst, FftDirection::Forward, norm);
}

void ifft_1d(ArrayRef<Complex32> src, MutArrayRef<Complex32> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f32(dst.size()).exec(src, dst, FftDirection::Inverse, norm);
}

void fft_1d_real(ArrayRef<f32> src, MutArrayRef<Complex32> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f32(dst.size()).exec_real_to_complex(src, dst, norm);
}

void rfft_1d(ArrayRef<f32> src, MutArrayRef<Complex32> half_dst, FftNormalization norm) noexcept {
    if (src.empty()) {
        KEEL_ASSERT(half_dst.empty());
        return;
    }
    cached_plan_f32(src.size()).exec_rfft(src, half_dst, norm);
}

void irfft_1d(ArrayRef<Complex32> half_src, MutArrayRef<f32> dst, usize n, FftNormalization norm) noexcept {
    if (n == 0) {
        KEEL_ASSERT(dst.empty());
        return;
    }
    KEEL_ASSERT(dst.size() == n);
    cached_plan_f32(n).exec_irfft(half_src, dst, norm);
}

void ifft_1d_symmetric(ArrayRef<Complex32> src, MutArrayRef<f32> dst, FftNormalization norm) noexcept {
    if (dst.empty()) return;
    cached_plan_f32(dst.size()).exec_symmetric_ifft(src, dst, norm);
}

// ============================================================================
// High-Level Batched / Strided Multi-Dimensional Transforms (f64 & f32)
// ============================================================================

void fft_batch(ArrayRef<Complex64> src, MutArrayRef<Complex64> dst, const FftBatchLayout& layout, FftDirection dir,
               FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f64(n).exec_batch(src, dst, layout, dir, norm);
}

void fft_batch_real(ArrayRef<f64> src, MutArrayRef<Complex64> dst, const FftBatchLayout& layout,
                    FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f64(n).exec_batch_real(src, dst, layout, norm);
}

void rfft_batch(ArrayRef<f64> src, MutArrayRef<Complex64> half_dst, const FftBatchLayout& layout,
                FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f64(n).exec_batch_rfft(src, half_dst, layout, norm);
}

void irfft_batch(ArrayRef<Complex64> half_src, MutArrayRef<f64> dst, const FftBatchLayout& layout,
                 FftNormalization norm) noexcept {
    if (layout.outer_count == 0 || layout.inner_count == 0) return;
    const usize n = infer_irfft_batch_len(layout);
    if (n == 0) return;
    cached_plan_f64(n).exec_batch_irfft(half_src, dst, layout, norm);
}

void ifft_batch_symmetric(ArrayRef<Complex64> src, MutArrayRef<f64> dst, const FftBatchLayout& layout,
                          FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f64(n).exec_batch_symmetric_ifft(src, dst, layout, norm);
}

void fft_batch(ArrayRef<Complex32> src, MutArrayRef<Complex32> dst, const FftBatchLayout& layout, FftDirection dir,
               FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f32(n).exec_batch(src, dst, layout, dir, norm);
}

void fft_batch_real(ArrayRef<f32> src, MutArrayRef<Complex32> dst, const FftBatchLayout& layout,
                    FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f32(n).exec_batch_real(src, dst, layout, norm);
}

void rfft_batch(ArrayRef<f32> src, MutArrayRef<Complex32> half_dst, const FftBatchLayout& layout,
                FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f32(n).exec_batch_rfft(src, half_dst, layout, norm);
}

void irfft_batch(ArrayRef<Complex32> half_src, MutArrayRef<f32> dst, const FftBatchLayout& layout,
                 FftNormalization norm) noexcept {
    if (layout.outer_count == 0 || layout.inner_count == 0) return;
    const usize n = infer_irfft_batch_len(layout);
    if (n == 0) return;
    cached_plan_f32(n).exec_batch_irfft(half_src, dst, layout, norm);
}

void ifft_batch_symmetric(ArrayRef<Complex32> src, MutArrayRef<f32> dst, const FftBatchLayout& layout,
                          FftNormalization norm) noexcept {
    const usize n = layout.effective_fft_len();
    if (n == 0 || layout.outer_count == 0 || layout.inner_count == 0) return;
    cached_plan_f32(n).exec_batch_symmetric_ifft(src, dst, layout, norm);
}

} // namespace keel
