# keelcpp/fft

Lightweight, high-performance C++20 Fast Fourier Transform library built on [`keelcpp/keel`](https://github.com/keelcpp/keel) under the `0BSD` license.

## Features

- **Arbitrary Transform Lengths (`N >= 1`):** Power-of-2 butterflies (AoS & SoA Radix-2, Pow-4 Radix-4, Stockham auto-sort) plus exact Bluestein chirp-z convolution for non-power-of-2 and prime lengths (`f64` and `f32`).
- **SIMD + Portable Fallback:** Optional Google Highway (`KEEL_FFT_WITH_HIGHWAY=ON` by default) runtime-dispatched SIMD kernels (AVX2, AVX-512, NEON, WASM SIMD128) with zero-dependency portable C++20 fallback (`KEEL_FFT_WITH_HIGHWAY=OFF`).
- **Real-Signal Acceleration:** Packed half-length `N/2` `rfft` / `irfft`, full conjugate-symmetric `fft_1d_real`, and `ifft_1d_symmetric` with SIMD twist/untwist kernels.
- **Reusable Plans & N-D Strided Batches:** Move-only `FftPlan1D` / `FftPlan1DF32` (`keel::get_scratch()` zero-allocation execution) and `FftBatchLayout` (`from_col_major` / `from_row_major`) for transforming arbitrary N-D tensor axes in a single call.
- **Pure Headers & Exception-Neutral:** `include/keel/fft.h` pulls zero STL bloat and compiles cleanly with `-fno-exceptions -fno-rtti`; opt-in `include/keel/fft_interop_std.h` provides zero-copy `std::complex` / `std::span` views.

## CMake Integration

Links as `keel::fft` (`find_package(keel_fft)` or `FetchContent`).
