#pragma once

// Opt-in zero-copy `std::span<std::complex<T>>` overloads for `keel::fft` (builds on `keel/interop_std.h`).

#include "keel/fft.h"
#include "keel/interop_std.h"

namespace keel::fft {

inline void fft_1d_inplace(std::span<std::complex<double>> buf,
                           FftNormalization norm = FftNormalization::Backward) noexcept {
    fft_1d_inplace(as_mut_complex_ref(buf), norm);
}

inline void ifft_1d_inplace(std::span<std::complex<double>> buf,
                            FftNormalization norm = FftNormalization::Backward) noexcept {
    ifft_1d_inplace(as_mut_complex_ref(buf), norm);
}

inline void fft_1d(std::span<const std::complex<double>> src, std::span<std::complex<double>> dst,
                   FftNormalization norm = FftNormalization::Backward) noexcept {
    fft_1d(as_complex_ref(src), as_mut_complex_ref(dst), norm);
}

inline void ifft_1d(std::span<const std::complex<double>> src, std::span<std::complex<double>> dst,
                    FftNormalization norm = FftNormalization::Backward) noexcept {
    ifft_1d(as_complex_ref(src), as_mut_complex_ref(dst), norm);
}

inline void fft_1d_inplace(std::span<std::complex<float>> buf,
                           FftNormalization norm = FftNormalization::Backward) noexcept {
    fft_1d_inplace(as_mut_complex_ref(buf), norm);
}

inline void ifft_1d_inplace(std::span<std::complex<float>> buf,
                            FftNormalization norm = FftNormalization::Backward) noexcept {
    ifft_1d_inplace(as_mut_complex_ref(buf), norm);
}

inline void fft_1d(std::span<const std::complex<float>> src, std::span<std::complex<float>> dst,
                   FftNormalization norm = FftNormalization::Backward) noexcept {
    fft_1d(as_complex_ref(src), as_mut_complex_ref(dst), norm);
}

inline void ifft_1d(std::span<const std::complex<float>> src, std::span<std::complex<float>> dst,
                    FftNormalization norm = FftNormalization::Backward) noexcept {
    ifft_1d(as_complex_ref(src), as_mut_complex_ref(dst), norm);
}

} // namespace keel::fft
