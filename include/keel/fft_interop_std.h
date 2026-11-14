#pragma once

// Opt-in zero-copy interoperability with `<complex>` and `<span>`.
// ISO C++ ([complex.numbers.general]) guarantees `std::complex<T>` has the exact array layout `T[2]`
// (`real` at offset 0, `imag` at offset `sizeof(T)`), matching `keel::Complex64` and `keel::Complex32`.

#include "keel/fft.h"

#include <complex>
#include <span>

namespace keel {

static_assert(sizeof(std::complex<double>) == sizeof(Complex64) && alignof(std::complex<double>) == alignof(Complex64));
static_assert(sizeof(std::complex<float>) == sizeof(Complex32) && alignof(std::complex<float>) == alignof(Complex32));

[[nodiscard]] constexpr std::complex<double> to_std_complex(Complex64 z) noexcept {
    return std::complex<double>(z.re, z.im);
}

[[nodiscard]] constexpr std::complex<float> to_std_complex(Complex32 z) noexcept {
    return std::complex<float>(z.re, z.im);
}

[[nodiscard]] constexpr Complex64 from_std_complex(std::complex<double> z) noexcept {
    return Complex64(z.real(), z.imag());
}

[[nodiscard]] constexpr Complex32 from_std_complex(std::complex<float> z) noexcept {
    return Complex32(z.real(), z.imag());
}

[[nodiscard]] inline ArrayRef<Complex64> as_complex_ref(const std::complex<double>* data, usize size) noexcept {
    return ArrayRef<Complex64>(reinterpret_cast<const Complex64*>(data), size);
}

[[nodiscard]] inline ArrayRef<Complex32> as_complex_ref(const std::complex<float>* data, usize size) noexcept {
    return ArrayRef<Complex32>(reinterpret_cast<const Complex32*>(data), size);
}

[[nodiscard]] inline ArrayRef<Complex64> as_complex_ref(std::span<const std::complex<double>> s) noexcept {
    return ArrayRef<Complex64>(reinterpret_cast<const Complex64*>(s.data()), s.size());
}

[[nodiscard]] inline ArrayRef<Complex32> as_complex_ref(std::span<const std::complex<float>> s) noexcept {
    return ArrayRef<Complex32>(reinterpret_cast<const Complex32*>(s.data()), s.size());
}

[[nodiscard]] inline MutArrayRef<Complex64> as_mut_complex_ref(std::complex<double>* data, usize size) noexcept {
    return MutArrayRef<Complex64>(reinterpret_cast<Complex64*>(data), size);
}

[[nodiscard]] inline MutArrayRef<Complex32> as_mut_complex_ref(std::complex<float>* data, usize size) noexcept {
    return MutArrayRef<Complex32>(reinterpret_cast<Complex32*>(data), size);
}

[[nodiscard]] inline MutArrayRef<Complex64> as_mut_complex_ref(std::span<std::complex<double>> s) noexcept {
    return MutArrayRef<Complex64>(reinterpret_cast<Complex64*>(s.data()), s.size());
}

[[nodiscard]] inline MutArrayRef<Complex32> as_mut_complex_ref(std::span<std::complex<float>> s) noexcept {
    return MutArrayRef<Complex32>(reinterpret_cast<Complex32*>(s.data()), s.size());
}

[[nodiscard]] inline std::span<const std::complex<double>> to_std_complex_span(ArrayRef<Complex64> s) noexcept {
    return std::span<const std::complex<double>>(reinterpret_cast<const std::complex<double>*>(s.data()), s.size());
}

[[nodiscard]] inline std::span<std::complex<double>> to_std_complex_span(MutArrayRef<Complex64> s) noexcept {
    return std::span<std::complex<double>>(reinterpret_cast<std::complex<double>*>(s.data()), s.size());
}

[[nodiscard]] inline std::span<const std::complex<float>> to_std_complex_span(ArrayRef<Complex32> s) noexcept {
    return std::span<const std::complex<float>>(reinterpret_cast<const std::complex<float>*>(s.data()), s.size());
}

[[nodiscard]] inline std::span<std::complex<float>> to_std_complex_span(MutArrayRef<Complex32> s) noexcept {
    return std::span<std::complex<float>>(reinterpret_cast<std::complex<float>*>(s.data()), s.size());
}

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

} // namespace keel
