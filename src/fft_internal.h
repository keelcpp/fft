#pragma once

#include "keel/arena.h"
#include "keel/bits.h"
#include "keel/fft.h"

namespace keel::fft::detail {

inline constexpr f64 kPi = 3.141592653589793238462643383279502884;

[[nodiscard]] constexpr bool is_pow2(usize n) noexcept {
    return n != 0 && (n & (n - 1)) == 0;
}

[[nodiscard]] constexpr usize ilog2_pow2(usize n) noexcept {
    KEEL_ASSERT(is_pow2(n));
    return static_cast<usize>(countr_zero(static_cast<u64>(n)));
}

[[nodiscard]] constexpr bool is_pow4(usize n) noexcept {
    return is_pow2(n) && (ilog2_pow2(n) & 1u) == 0u;
}

[[nodiscard]] constexpr usize next_pow2(usize n) noexcept {
    if (n <= 1) return 1;
    return static_cast<usize>(bit_ceil(static_cast<u64>(n)));
}

// Fast RAII scratch allocator backed by `keel::get_scratch()` (64-byte SIMD aligned),
// with automatic heap fallback only if a gigantic transform exceeds remaining arena space.
class ScratchScope {
public:
    ScratchScope() noexcept : m_temp(get_scratch()) {}

    ~ScratchScope() noexcept {
        for (usize i = m_heap_count; i > 0; --i) {
            const HeapBlock& b = m_heap[i - 1];
            default_allocator()->free_bytes(b.ptr, b.bytes, 64);
        }
    }

    ScratchScope(const ScratchScope&) = delete;
    ScratchScope& operator=(const ScratchScope&) = delete;

    template <typename T>
    [[nodiscard]] T* alloc(usize count) noexcept {
        if (count == 0) return nullptr;
        if (count > (~usize{0}) / sizeof(T)) [[unlikely]] {
            handle_oom(~usize{0}, 64);
        }
        const usize bytes = count * sizeof(T);
        Arena& a = m_temp.arena();
        const usize used = a.used();
        const usize res = a.reserved();
        if (res >= used && (res - used) >= bytes + 128) [[likely]] {
            return static_cast<T*>(a.alloc(bytes, 64));
        }
        void* p = default_allocator()->alloc_bytes(bytes, 64);
        KEEL_ASSERT(m_heap_count < 8);
        m_heap[m_heap_count++] = HeapBlock{p, bytes};
        return static_cast<T*>(p);
    }

private:
    struct HeapBlock {
        void* ptr = nullptr;
        usize bytes = 0;
    };

    ArenaTemp m_temp;
    HeapBlock m_heap[8]{};
    usize m_heap_count = 0;
};

// Unit-stride RFFT/IRFFT twist helpers (dispatched to Highway SIMD when enabled, scalar portable otherwise).
void rfft_twist_full_unit_stride(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale,
                                 c64* dst) noexcept;
void rfft_twist_full_unit_stride(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale,
                                 c32* dst) noexcept;

void rfft_twist_half_unit_stride(const f64* z_re, const f64* z_im, usize n, const c64* w_full, f64 scale,
                                 c64* half_dst) noexcept;
void rfft_twist_half_unit_stride(const f32* z_re, const f32* z_im, usize n, const c32* w_full, f32 scale,
                                 c32* half_dst) noexcept;

void irfft_untwist_conj_unit_stride(const c64* half_src, usize n, const c64* w_full, f64* z_re,
                                    f64* z_im) noexcept;
void irfft_untwist_conj_unit_stride(const c32* half_src, usize n, const c32* w_full, f32* z_re,
                                    f32* z_im) noexcept;

void irfft_store_interleaved_unit_stride(const f64* z_re, const f64* z_im, usize half, f64 post_scale,
                                         f64* real_dst) noexcept;
void irfft_store_interleaved_unit_stride(const f32* z_re, const f32* z_im, usize half, f32 post_scale,
                                         f32* real_dst) noexcept;

#if defined(KEEL_FFT_WITH_HIGHWAY)
void hwy_radix2_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void hwy_radix2_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void hwy_radix4_pow4_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void hwy_radix4_pow4_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void hwy_stockham_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void hwy_stockham_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void hwy_radix2_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void hwy_radix2_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void hwy_radix2_soa_stages_inplace(f64* re, f64* im, usize n, const c64* w_fwd) noexcept;
void hwy_radix2_soa_stages_inplace(f32* re, f32* im, usize n, const c32* w_fwd) noexcept;

void hwy_radix4_pow4_soa_inplace(c64* buf, usize n, const c64* w_fwd) noexcept;
void hwy_radix4_pow4_soa_inplace(c32* buf, usize n, const c32* w_fwd) noexcept;

void hwy_radix4_pow4_soa_stages_inplace(f64* re, f64* im, usize n, const c64* w_fwd) noexcept;
void hwy_radix4_pow4_soa_stages_inplace(f32* re, f32* im, usize n, const c32* w_fwd) noexcept;
#endif

} // namespace keel::fft::detail
