#pragma once

#include "keel/array.h"
#include "keel/complex.h"
#include "keel/result.h"

namespace keel::fft {

enum class FftDirection : i8 {
    Forward = -1,
    Inverse = 1,
};

enum class FftNormalization : u8 {
    None = 0,
    Backward,
    Ortho,
};

// General strided batch layout for transforming N-D tensors along an arbitrary axis in a single call.
// For each outer index `o` in `[0, outer_count)` and inner index `i` in `[0, inner_count)`:
//   src_base = o * outer_src_step + i * inner_step
//   dst_base = o * outer_dst_step + i * inner_step
//   src_elem(k) = src[src_base + k * elem_stride]             for k in [0, input_len)
//   dst_elem(k) = dst[dst_base + k * effective_dst_stride()]  for k in [0, out_len)
struct FftBatchLayout {
    usize input_len = 0;
    usize fft_len = 0;
    usize elem_stride = 1;
    usize dst_elem_stride = 0;
    usize inner_count = 1;
    usize inner_step = 1;
    usize outer_count = 1;
    usize outer_src_step = 0;
    usize outer_dst_step = 0;

    [[nodiscard]] constexpr usize effective_fft_len() const noexcept {
        return (fft_len > 0) ? fft_len : input_len;
    }

    [[nodiscard]] constexpr usize effective_dst_stride() const noexcept {
        return (dst_elem_stride > 0) ? dst_elem_stride : elem_stride;
    }

    // Contiguous batch of `batch_count` 1-D signals, each of length `in_len`.
    [[nodiscard]] static constexpr FftBatchLayout contiguous(
        usize batch_count, usize in_len, usize target_fft_len = 0, usize dst_slice_len = 0) noexcept {
        const usize n = (target_fft_len > 0) ? target_fft_len : in_len;
        const usize out_step = (dst_slice_len > 0) ? dst_slice_len : n;
        return FftBatchLayout{
            .input_len = in_len,
            .fft_len = n,
            .elem_stride = 1,
            .dst_elem_stride = 1,
            .inner_count = 1,
            .inner_step = 1,
            .outer_count = batch_count,
            .outer_src_step = in_len,
            .outer_dst_step = out_step,
        };
    }

    // Column-major N-D tensor with shape `dims` transforming along `axis` (0-indexed).
    // Pass `dst_axis_len = actual_fft_len / 2 + 1` when building a layout for `rfft_batch`,
    // or `in_axis_len_override = actual_fft_len / 2 + 1` for `irfft_batch`.
    [[nodiscard]] static constexpr FftBatchLayout from_col_major(
        ArrayRef<usize> dims, usize axis, usize target_fft_len = 0, usize dst_axis_len = 0) noexcept {
        KEEL_ASSERT(axis < dims.size());
        usize inner = 1;
        for (usize d = 0; d < axis; ++d) {
            inner *= dims[d];
        }
        usize outer = 1;
        for (usize d = axis + 1; d < dims.size(); ++d) {
            outer *= dims[d];
        }
        const usize in_len = dims[axis];
        const usize n = (target_fft_len > 0) ? target_fft_len : in_len;
        const usize out_len = (dst_axis_len > 0) ? dst_axis_len : n;
        if (inner == 1) {
            return FftBatchLayout{
                .input_len = in_len,
                .fft_len = n,
                .elem_stride = 1,
                .dst_elem_stride = 1,
                .inner_count = 1,
                .inner_step = 1,
                .outer_count = outer,
                .outer_src_step = in_len,
                .outer_dst_step = out_len,
            };
        }
        return FftBatchLayout{
            .input_len = in_len,
            .fft_len = n,
            .elem_stride = inner,
            .dst_elem_stride = inner,
            .inner_count = inner,
            .inner_step = 1,
            .outer_count = outer,
            .outer_src_step = inner * in_len,
            .outer_dst_step = inner * out_len,
        };
    }

    // Row-major N-D tensor with shape `dims` transforming along `axis` (0-indexed).
    [[nodiscard]] static constexpr FftBatchLayout from_row_major(
        ArrayRef<usize> dims, usize axis, usize target_fft_len = 0, usize dst_axis_len = 0) noexcept {
        KEEL_ASSERT(axis < dims.size());
        usize outer = 1;
        for (usize d = 0; d < axis; ++d) {
            outer *= dims[d];
        }
        usize inner = 1;
        for (usize d = axis + 1; d < dims.size(); ++d) {
            inner *= dims[d];
        }
        const usize in_len = dims[axis];
        const usize n = (target_fft_len > 0) ? target_fft_len : in_len;
        const usize out_len = (dst_axis_len > 0) ? dst_axis_len : n;
        if (inner == 1) {
            return FftBatchLayout{
                .input_len = in_len,
                .fft_len = n,
                .elem_stride = 1,
                .dst_elem_stride = 1,
                .inner_count = 1,
                .inner_step = 1,
                .outer_count = outer,
                .outer_src_step = in_len,
                .outer_dst_step = out_len,
            };
        }
        return FftBatchLayout{
            .input_len = in_len,
            .fft_len = n,
            .elem_stride = inner,
            .dst_elem_stride = inner,
            .inner_count = inner,
            .inner_step = 1,
            .outer_count = outer,
            .outer_src_step = inner * in_len,
            .outer_dst_step = inner * out_len,
        };
    }
};

// Reusable 1-D FFT plan for double-precision (`f64` / `c64`) transforms.
// Supports arbitrary `n >= 1`: power-of-2 sizes use radix-2 / radix-4 / SoA butterflies,
// while non-power-of-2 sizes use Bluestein's exact chirp-z convolution.
// All `exec*` methods acquire temporary workspace from `keel::get_scratch()` (zero heap allocations per call).
class FftPlan1D {
public:
    FftPlan1D() noexcept = default;
    FftPlan1D(const FftPlan1D&) = delete;
    FftPlan1D& operator=(const FftPlan1D&) = delete;

    FftPlan1D(FftPlan1D&& other) noexcept
        : m_n(other.m_n), m_conv_m(other.m_conv_m), m_is_pow2(other.m_is_pow2),
          m_twiddles(keel::move(other.m_twiddles)), m_half_twiddles(keel::move(other.m_half_twiddles)),
          m_chirp(keel::move(other.m_chirp)), m_v_freq(keel::move(other.m_v_freq)) {
        other.m_n = 0;
        other.m_conv_m = 0;
        other.m_is_pow2 = false;
    }

    FftPlan1D& operator=(FftPlan1D&& other) noexcept {
        if (this != &other) {
            m_n = other.m_n;
            m_conv_m = other.m_conv_m;
            m_is_pow2 = other.m_is_pow2;
            m_twiddles = keel::move(other.m_twiddles);
            m_half_twiddles = keel::move(other.m_half_twiddles);
            m_chirp = keel::move(other.m_chirp);
            m_v_freq = keel::move(other.m_v_freq);
            other.m_n = 0;
            other.m_conv_m = 0;
            other.m_is_pow2 = false;
        }
        return *this;
    }

    ~FftPlan1D() noexcept = default;

    // Creates a plan for transform length `n`. Returns `ErrorCode::InvalidArgument` if `n == 0`.
    [[nodiscard]] static Result<FftPlan1D> create(usize n, Allocator* alloc = nullptr);

    [[nodiscard]] FftPlan1D clone(Allocator* alloc = nullptr) const;

    [[nodiscard]] usize size() const noexcept { return m_n; }
    [[nodiscard]] bool empty() const noexcept { return m_n == 0; }
    [[nodiscard]] bool is_power_of_two() const noexcept { return m_is_pow2; }
    [[nodiscard]] Allocator* allocator() const noexcept { return m_twiddles.allocator(); }
    [[nodiscard]] ArrayRef<c64> twiddles() const noexcept { return m_twiddles; }

    void exec_inplace(MutArrayRef<c64> buf,
                      FftDirection dir = FftDirection::Forward,
                      FftNormalization norm = FftNormalization::Backward) const noexcept;

    // Out-of-place (or aliased) transform of length `size()`.
    // If `src.size() < size()`, zero-pads; if `src.size() > size()`, truncates.
    void exec(ArrayRef<c64> src,
              MutArrayRef<c64> dst,
              FftDirection dir = FftDirection::Forward,
              FftNormalization norm = FftNormalization::Backward) const noexcept;

    // Real-to-half-spectrum forward FFT (`half_spectrum_dst.size() == size() / 2 + 1`).
    void exec_rfft(ArrayRef<f64> real_src,
                   MutArrayRef<c64> half_spectrum_dst,
                   FftNormalization norm = FftNormalization::Backward) const noexcept;

    // Real-to-full-spectrum forward FFT (`full_dst.size() == size()`), filling conjugate-symmetric upper bins.
    void exec_real_to_complex(ArrayRef<f64> real_src,
                              MutArrayRef<c64> full_dst,
                              FftNormalization norm = FftNormalization::Backward) const noexcept;

    // Half-spectrum-to-real inverse FFT (`half_spectrum_src.size() == size() / 2 + 1`, `real_dst.size() == size()`).
    void exec_irfft(ArrayRef<c64> half_spectrum_src,
                    MutArrayRef<f64> real_dst,
                    FftNormalization norm = FftNormalization::Backward) const noexcept;

    // Conjugate-symmetric complex-to-real inverse FFT (`real_dst.size() == size()`).
    // Treats lower half `0..size()/2` of padded/truncated `src` as authoritative (DC and even Nyquist forced real).
    void exec_symmetric_ifft(ArrayRef<c64> src,
                             MutArrayRef<f64> real_dst,
                             FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch(ArrayRef<c64> src,
                    MutArrayRef<c64> dst,
                    const FftBatchLayout& layout,
                    FftDirection dir = FftDirection::Forward,
                    FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_real(ArrayRef<f64> src,
                         MutArrayRef<c64> dst,
                         const FftBatchLayout& layout,
                         FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_rfft(ArrayRef<f64> src,
                         MutArrayRef<c64> half_dst,
                         const FftBatchLayout& layout,
                         FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_irfft(ArrayRef<c64> half_src,
                          MutArrayRef<f64> dst,
                          const FftBatchLayout& layout,
                          FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_symmetric_ifft(ArrayRef<c64> src,
                                   MutArrayRef<f64> dst,
                                   const FftBatchLayout& layout,
                                   FftNormalization norm = FftNormalization::Backward) const noexcept;

private:
    usize m_n = 0;
    usize m_conv_m = 0;
    bool m_is_pow2 = false;
    Array<c64> m_twiddles;
    Array<c64> m_half_twiddles;
    Array<c64> m_chirp;
    Array<c64> m_v_freq;
};

// Reusable 1-D FFT plan for single-precision (`f32` / `c32`) transforms.
class FftPlan1DF32 {
public:
    FftPlan1DF32() noexcept = default;
    FftPlan1DF32(const FftPlan1DF32&) = delete;
    FftPlan1DF32& operator=(const FftPlan1DF32&) = delete;

    FftPlan1DF32(FftPlan1DF32&& other) noexcept
        : m_n(other.m_n), m_conv_m(other.m_conv_m), m_is_pow2(other.m_is_pow2),
          m_twiddles(keel::move(other.m_twiddles)), m_half_twiddles(keel::move(other.m_half_twiddles)),
          m_chirp(keel::move(other.m_chirp)), m_v_freq(keel::move(other.m_v_freq)) {
        other.m_n = 0;
        other.m_conv_m = 0;
        other.m_is_pow2 = false;
    }

    FftPlan1DF32& operator=(FftPlan1DF32&& other) noexcept {
        if (this != &other) {
            m_n = other.m_n;
            m_conv_m = other.m_conv_m;
            m_is_pow2 = other.m_is_pow2;
            m_twiddles = keel::move(other.m_twiddles);
            m_half_twiddles = keel::move(other.m_half_twiddles);
            m_chirp = keel::move(other.m_chirp);
            m_v_freq = keel::move(other.m_v_freq);
            other.m_n = 0;
            other.m_conv_m = 0;
            other.m_is_pow2 = false;
        }
        return *this;
    }

    ~FftPlan1DF32() noexcept = default;

    // Creates a plan for transform length `n`. Returns `ErrorCode::InvalidArgument` if `n == 0`.
    [[nodiscard]] static Result<FftPlan1DF32> create(usize n, Allocator* alloc = nullptr);

    [[nodiscard]] FftPlan1DF32 clone(Allocator* alloc = nullptr) const;

    [[nodiscard]] usize size() const noexcept { return m_n; }
    [[nodiscard]] bool empty() const noexcept { return m_n == 0; }
    [[nodiscard]] bool is_power_of_two() const noexcept { return m_is_pow2; }
    [[nodiscard]] Allocator* allocator() const noexcept { return m_twiddles.allocator(); }
    [[nodiscard]] ArrayRef<c32> twiddles() const noexcept { return m_twiddles; }

    void exec_inplace(MutArrayRef<c32> buf,
                      FftDirection dir = FftDirection::Forward,
                      FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec(ArrayRef<c32> src,
              MutArrayRef<c32> dst,
              FftDirection dir = FftDirection::Forward,
              FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_rfft(ArrayRef<f32> real_src,
                   MutArrayRef<c32> half_spectrum_dst,
                   FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_real_to_complex(ArrayRef<f32> real_src,
                              MutArrayRef<c32> full_dst,
                              FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_irfft(ArrayRef<c32> half_spectrum_src,
                    MutArrayRef<f32> real_dst,
                    FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_symmetric_ifft(ArrayRef<c32> src,
                             MutArrayRef<f32> real_dst,
                             FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch(ArrayRef<c32> src,
                    MutArrayRef<c32> dst,
                    const FftBatchLayout& layout,
                    FftDirection dir = FftDirection::Forward,
                    FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_real(ArrayRef<f32> src,
                         MutArrayRef<c32> dst,
                         const FftBatchLayout& layout,
                         FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_rfft(ArrayRef<f32> src,
                         MutArrayRef<c32> half_dst,
                         const FftBatchLayout& layout,
                         FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_irfft(ArrayRef<c32> half_src,
                          MutArrayRef<f32> dst,
                          const FftBatchLayout& layout,
                          FftNormalization norm = FftNormalization::Backward) const noexcept;

    void exec_batch_symmetric_ifft(ArrayRef<c32> src,
                                   MutArrayRef<f32> dst,
                                   const FftBatchLayout& layout,
                                   FftNormalization norm = FftNormalization::Backward) const noexcept;

private:
    usize m_n = 0;
    usize m_conv_m = 0;
    bool m_is_pow2 = false;
    Array<c32> m_twiddles;
    Array<c32> m_half_twiddles;
    Array<c32> m_chirp;
    Array<c32> m_v_freq;
};

static_assert(IsNothrowMoveConstructible<FftPlan1D> && IsNothrowDestructible<FftPlan1D>);
static_assert(IsNothrowMoveConstructible<FftPlan1DF32> && IsNothrowDestructible<FftPlan1DF32>);

// ============================================================================
// High-Level Stateless / Thread-Cached 1-D Transforms (f64 & f32)
// ============================================================================

void fft_1d_inplace(MutArrayRef<c64> buf, FftNormalization norm = FftNormalization::Backward) noexcept;
void ifft_1d_inplace(MutArrayRef<c64> buf, FftNormalization norm = FftNormalization::Backward) noexcept;

void fft_1d(ArrayRef<c64> src, MutArrayRef<c64> dst,
            FftNormalization norm = FftNormalization::Backward) noexcept;
void ifft_1d(ArrayRef<c64> src, MutArrayRef<c64> dst,
             FftNormalization norm = FftNormalization::Backward) noexcept;

void fft_1d_real(ArrayRef<f64> src, MutArrayRef<c64> dst,
                 FftNormalization norm = FftNormalization::Backward) noexcept;
void rfft_1d(ArrayRef<f64> src, MutArrayRef<c64> half_dst,
             FftNormalization norm = FftNormalization::Backward) noexcept;
void irfft_1d(ArrayRef<c64> half_src, MutArrayRef<f64> dst, usize n,
              FftNormalization norm = FftNormalization::Backward) noexcept;
inline void irfft_1d(ArrayRef<c64> half_src, MutArrayRef<f64> dst,
                     FftNormalization norm = FftNormalization::Backward) noexcept {
    irfft_1d(half_src, dst, dst.size(), norm);
}
void ifft_1d_symmetric(ArrayRef<c64> src, MutArrayRef<f64> dst,
                       FftNormalization norm = FftNormalization::Backward) noexcept;

void fft_1d_inplace(MutArrayRef<c32> buf, FftNormalization norm = FftNormalization::Backward) noexcept;
void ifft_1d_inplace(MutArrayRef<c32> buf, FftNormalization norm = FftNormalization::Backward) noexcept;

void fft_1d(ArrayRef<c32> src, MutArrayRef<c32> dst,
            FftNormalization norm = FftNormalization::Backward) noexcept;
void ifft_1d(ArrayRef<c32> src, MutArrayRef<c32> dst,
             FftNormalization norm = FftNormalization::Backward) noexcept;

void fft_1d_real(ArrayRef<f32> src, MutArrayRef<c32> dst,
                 FftNormalization norm = FftNormalization::Backward) noexcept;
void rfft_1d(ArrayRef<f32> src, MutArrayRef<c32> half_dst,
             FftNormalization norm = FftNormalization::Backward) noexcept;
void irfft_1d(ArrayRef<c32> half_src, MutArrayRef<f32> dst, usize n,
              FftNormalization norm = FftNormalization::Backward) noexcept;
inline void irfft_1d(ArrayRef<c32> half_src, MutArrayRef<f32> dst,
                     FftNormalization norm = FftNormalization::Backward) noexcept {
    irfft_1d(half_src, dst, dst.size(), norm);
}
void ifft_1d_symmetric(ArrayRef<c32> src, MutArrayRef<f32> dst,
                       FftNormalization norm = FftNormalization::Backward) noexcept;

// ============================================================================
// High-Level Batched / Strided Multi-Dimensional Transforms (f64 & f32)
// ============================================================================

void fft_batch(ArrayRef<c64> src, MutArrayRef<c64> dst, const FftBatchLayout& layout,
               FftDirection dir = FftDirection::Forward,
               FftNormalization norm = FftNormalization::Backward) noexcept;
inline void ifft_batch(ArrayRef<c64> src, MutArrayRef<c64> dst, const FftBatchLayout& layout,
                       FftNormalization norm = FftNormalization::Backward) noexcept {
    fft_batch(src, dst, layout, FftDirection::Inverse, norm);
}

void fft_batch_real(ArrayRef<f64> src, MutArrayRef<c64> dst, const FftBatchLayout& layout,
                    FftNormalization norm = FftNormalization::Backward) noexcept;
void rfft_batch(ArrayRef<f64> src, MutArrayRef<c64> half_dst, const FftBatchLayout& layout,
                FftNormalization norm = FftNormalization::Backward) noexcept;
void irfft_batch(ArrayRef<c64> half_src, MutArrayRef<f64> dst, const FftBatchLayout& layout,
                 FftNormalization norm = FftNormalization::Backward) noexcept;
void ifft_batch_symmetric(ArrayRef<c64> src, MutArrayRef<f64> dst, const FftBatchLayout& layout,
                          FftNormalization norm = FftNormalization::Backward) noexcept;

void fft_batch(ArrayRef<c32> src, MutArrayRef<c32> dst, const FftBatchLayout& layout,
               FftDirection dir = FftDirection::Forward,
               FftNormalization norm = FftNormalization::Backward) noexcept;
inline void ifft_batch(ArrayRef<c32> src, MutArrayRef<c32> dst, const FftBatchLayout& layout,
                       FftNormalization norm = FftNormalization::Backward) noexcept {
    fft_batch(src, dst, layout, FftDirection::Inverse, norm);
}

void fft_batch_real(ArrayRef<f32> src, MutArrayRef<c32> dst, const FftBatchLayout& layout,
                    FftNormalization norm = FftNormalization::Backward) noexcept;
void rfft_batch(ArrayRef<f32> src, MutArrayRef<c32> half_dst, const FftBatchLayout& layout,
                FftNormalization norm = FftNormalization::Backward) noexcept;
void irfft_batch(ArrayRef<c32> half_src, MutArrayRef<f32> dst, const FftBatchLayout& layout,
                 FftNormalization norm = FftNormalization::Backward) noexcept;
void ifft_batch_symmetric(ArrayRef<c32> src, MutArrayRef<f32> dst, const FftBatchLayout& layout,
                          FftNormalization norm = FftNormalization::Backward) noexcept;

// ============================================================================
// Low-Level Power-of-2 Kernel Entry Points (SIMD & Portable)
// ============================================================================

namespace fft_kernels {

void fill_twiddles_f64(MutArrayRef<c64> w, usize n, FftDirection dir = FftDirection::Forward) noexcept;
void fill_twiddles_f32(MutArrayRef<c32> w, usize n, FftDirection dir = FftDirection::Forward) noexcept;

void fft_radix2_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void fft_radix2_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void fft_radix4_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void fft_radix4_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void fft_stockham_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void fft_stockham_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void fft_radix2_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void fft_radix2_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void fft_radix4_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void fft_radix4_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void fft_soa_stages_inplace(f64* re, f64* im, usize n, const c64* w_fwd) noexcept;
void fft_soa_stages_inplace(f32* re, f32* im, usize n, const c32* w_fwd) noexcept;

} // namespace fft_kernels

using fft_kernels::fft_radix2_inplace;
using fft_kernels::fft_radix2_soa_inplace;
using fft_kernels::fft_radix4_inplace;
using fft_kernels::fft_radix4_soa_inplace;
using fft_kernels::fft_soa_stages_inplace;
using fft_kernels::fft_stockham_inplace;
using fft_kernels::fill_twiddles_f32;
using fft_kernels::fill_twiddles_f64;

// ============================================================================
// Spectrum Shift Helpers (DC <-> Center)
// ============================================================================

template <typename T>
constexpr void fftshift_1d(ArrayRef<T> src, MutArrayRef<T> dst) noexcept {
    const usize n = src.size();
    KEEL_ASSERT(dst.size() == n);
    if (n == 0) return;
    const usize pivot = (n + 1) / 2;
    const usize tail = n - pivot;
    for (usize i = 0; i < tail; ++i) {
        dst[i] = src[pivot + i];
    }
    for (usize i = 0; i < pivot; ++i) {
        dst[tail + i] = src[i];
    }
}

template <typename T>
constexpr void ifftshift_1d(ArrayRef<T> src, MutArrayRef<T> dst) noexcept {
    const usize n = src.size();
    KEEL_ASSERT(dst.size() == n);
    if (n == 0) return;
    const usize pivot = n / 2;
    const usize tail = n - pivot;
    for (usize i = 0; i < tail; ++i) {
        dst[i] = src[pivot + i];
    }
    for (usize i = 0; i < pivot; ++i) {
        dst[tail + i] = src[i];
    }
}

} // namespace keel::fft
