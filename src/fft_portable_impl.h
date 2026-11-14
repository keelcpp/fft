#pragma once

#include "fft_internal.h"

namespace keel::detail {

template <typename Cplx>
inline void portable_radix2_inplace(Cplx* buf, usize n, const Cplx* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(is_pow2(n));

    for (usize i = 1, j = 0; i < n; ++i) {
        usize bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            keel::swap(buf[i], buf[j]);
        }
    }

    for (usize len = 2; len <= n; len <<= 1) {
        const usize half = len >> 1;
        const usize step = n / len;
        for (usize i = 0; i < n; i += len) {
            for (usize j = 0; j < half; ++j) {
                const Cplx w = w_fwd[j * step];
                const Cplx u = buf[i + j];
                const Cplx v = buf[i + j + half] * w;
                buf[i + j] = u + v;
                buf[i + j + half] = u - v;
            }
        }
    }
}

template <typename Cplx>
inline void portable_radix4_pow4_inplace(Cplx* buf, usize n, const Cplx* w_fwd) noexcept {
    if (n <= 1) return;
    if (!is_pow4(n)) {
        portable_radix2_inplace(buf, n, w_fwd);
        return;
    }

    const usize digits = ilog2_pow2(n) / 2;
    for (usize i = 0; i < n; ++i) {
        usize j = 0;
        usize x = i;
        for (usize d = 0; d < digits; ++d) {
            j = (j << 2) | (x & 3u);
            x >>= 2;
        }
        if (i < j) {
            keel::swap(buf[i], buf[j]);
        }
    }

    const usize n_half = n >> 1;
    for (usize len = 4; len <= n; len <<= 2) {
        const usize m = len >> 2;
        const usize step = n / len;
        for (usize i = 0; i < n; i += len) {
            for (usize j = 0; j < m; ++j) {
                const Cplx w1 = w_fwd[j * step];
                const Cplx w2 = w_fwd[2 * j * step];
                const usize i3 = 3 * j * step;
                const Cplx w3 = (i3 < n_half) ? w_fwd[i3] : -w_fwd[i3 - n_half];

                const Cplx x0 = buf[i + j];
                const Cplx x1 = buf[i + j + m] * w1;
                const Cplx x2 = buf[i + j + 2 * m] * w2;
                const Cplx x3 = buf[i + j + 3 * m] * w3;

                const Cplx a = x0 + x2;
                const Cplx b = x0 - x2;
                const Cplx c = x1 + x3;
                const Cplx dc = x1 - x3;
                const Cplx id(-dc.im, dc.re);

                buf[i + j] = a + c;
                buf[i + j + m] = b - id;
                buf[i + j + 2 * m] = a - c;
                buf[i + j + 3 * m] = b + id;
            }
        }
    }
}

template <typename Cplx>
inline void portable_stockham_inplace(Cplx* buf, usize n, const Cplx* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(is_pow2(n));

    ScratchScope scratch;
    Cplx* tmp = scratch.alloc<Cplx>(n);

    Cplx* src = buf;
    Cplx* dst = tmp;
    const usize stages = ilog2_pow2(n);

    for (usize stage = 0; stage < stages; ++stage) {
        const usize m = usize{1} << (stage + 1);
        const usize mh = m >> 1;
        const usize n_sub = n / m;
        const usize step = n_sub;

        for (usize pair_idx = 0; pair_idx < n_sub; ++pair_idx) {
            const Cplx* a_in = src + pair_idx * mh;
            const Cplx* b_in = src + (pair_idx + n_sub) * mh;
            Cplx* a_out = dst + pair_idx * m;
            Cplx* b_out = a_out + mh;

            for (usize k = 0; k < mh; ++k) {
                const Cplx a = a_in[k];
                const Cplx b = b_in[k] * w_fwd[k * step];
                a_out[k] = a + b;
                b_out[k] = a - b;
            }
        }
        keel::swap(src, dst);
    }

    if (src != buf) {
        mem_copy(buf, src, n * sizeof(Cplx));
    }
}

template <typename T, typename Cplx>
inline void portable_radix2_soa_stages_inplace(T* re, T* im, usize n, const Cplx* w_fwd) noexcept {
    if (n <= 1) return;
    KEEL_ASSERT(is_pow2(n));

    for (usize i = 1, j = 0; i < n; ++i) {
        usize bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            keel::swap(re[i], re[j]);
            keel::swap(im[i], im[j]);
        }
    }

    for (usize len = 2; len <= n; len <<= 1) {
        const usize half = len >> 1;
        const usize step = n / len;
        for (usize i = 0; i < n; i += len) {
            for (usize j = 0; j < half; ++j) {
                const Cplx w = w_fwd[j * step];
                const T u_re = re[i + j];
                const T u_im = im[i + j];
                const T v_re = re[i + j + half];
                const T v_im = im[i + j + half];
                const T bv_re = v_re * w.re - v_im * w.im;
                const T bv_im = v_re * w.im + v_im * w.re;
                re[i + j] = u_re + bv_re;
                im[i + j] = u_im + bv_im;
                re[i + j + half] = u_re - bv_re;
                im[i + j + half] = u_im - bv_im;
            }
        }
    }
}

template <typename T, typename Cplx>
inline void portable_radix4_pow4_soa_stages_inplace(T* re, T* im, usize n, const Cplx* w_fwd) noexcept {
    if (n <= 1) return;
    if (!is_pow4(n)) {
        portable_radix2_soa_stages_inplace(re, im, n, w_fwd);
        return;
    }

    const usize digits = ilog2_pow2(n) / 2;
    for (usize i = 0; i < n; ++i) {
        usize j = 0;
        usize x = i;
        for (usize d = 0; d < digits; ++d) {
            j = (j << 2) | (x & 3u);
            x >>= 2;
        }
        if (i < j) {
            keel::swap(re[i], re[j]);
            keel::swap(im[i], im[j]);
        }
    }

    const usize n_half = n >> 1;
    for (usize len = 4; len <= n; len <<= 2) {
        const usize m = len >> 2;
        const usize step = n / len;
        for (usize i = 0; i < n; i += len) {
            for (usize j = 0; j < m; ++j) {
                const Cplx w1 = w_fwd[j * step];
                const Cplx w2 = w_fwd[2 * j * step];
                const usize i3 = 3 * j * step;
                const Cplx w3 = (i3 < n_half) ? w_fwd[i3] : -w_fwd[i3 - n_half];

                const T u_re = re[i + j];
                const T u_im = im[i + j];
                const T v_re = re[i + j + m];
                const T v_im = im[i + j + m];
                const T w_re = re[i + j + 2 * m];
                const T w_im = im[i + j + 2 * m];
                const T z_re = re[i + j + 3 * m];
                const T z_im = im[i + j + 3 * m];

                const T x1re = v_re * w1.re - v_im * w1.im;
                const T x1im = v_re * w1.im + v_im * w1.re;
                const T x2re = w_re * w2.re - w_im * w2.im;
                const T x2im = w_re * w2.im + w_im * w2.re;
                const T x3re = z_re * w3.re - z_im * w3.im;
                const T x3im = z_re * w3.im + z_im * w3.re;

                const T are = u_re + x2re;
                const T aim = u_im + x2im;
                const T bre = u_re - x2re;
                const T bim = u_im - x2im;
                const T cre = x1re + x3re;
                const T cim = x1im + x3im;
                const T dre = x1re - x3re;
                const T dim = x1im - x3im;

                re[i + j] = are + cre;
                im[i + j] = aim + cim;
                re[i + j + m] = bre + dim;
                im[i + j + m] = bim - dre;
                re[i + j + 2 * m] = are - cre;
                im[i + j + 2 * m] = aim - cim;
                re[i + j + 3 * m] = bre - dim;
                im[i + j + 3 * m] = bim + dre;
            }
        }
    }
}

template <typename T, typename Cplx>
inline void portable_rfft_twist_full(const T* z_re, const T* z_im, usize n, const Cplx* w_full, T scale,
                                     Cplx* dst, usize dst_stride) noexcept {
    const usize half = n >> 1;
    const T z0_re = z_re[0];
    const T z0_im = z_im[0];
    dst[0] = Cplx((z0_re + z0_im) * scale, T(0));
    dst[half * dst_stride] = Cplx((z0_re - z0_im) * scale, T(0));

    const T half_scale = T(0.5) * scale;
    for (usize k = 1; k < half; ++k) {
        const T zk_re = z_re[k];
        const T zk_im = z_im[k];
        const T zj_re = z_re[half - k];
        const T zj_im = -z_im[half - k];

        const T e_re = half_scale * (zk_re + zj_re);
        const T e_im = half_scale * (zk_im + zj_im);
        const T d_re = half_scale * (zk_re - zj_re);
        const T d_im = half_scale * (zk_im - zj_im);

        const T o_re = d_im;
        const T o_im = -d_re;

        const T wk_re = w_full[k].re;
        const T wk_im = w_full[k].im;
        const T wo_re = wk_re * o_re - wk_im * o_im;
        const T wo_im = wk_re * o_im + wk_im * o_re;

        const Cplx xk(e_re + wo_re, e_im + wo_im);
        dst[k * dst_stride] = xk;
        dst[(n - k) * dst_stride] = conj(xk);
    }
}

template <typename T, typename Cplx>
inline void portable_rfft_twist_half(const T* z_re, const T* z_im, usize n, const Cplx* w_full, T scale,
                                     Cplx* half_dst, usize dst_stride) noexcept {
    const usize half = n >> 1;
    const T z0_re = z_re[0];
    const T z0_im = z_im[0];
    half_dst[0] = Cplx((z0_re + z0_im) * scale, T(0));
    half_dst[half * dst_stride] = Cplx((z0_re - z0_im) * scale, T(0));

    const T half_scale = T(0.5) * scale;
    for (usize k = 1; k < half; ++k) {
        const T zk_re = z_re[k];
        const T zk_im = z_im[k];
        const T zj_re = z_re[half - k];
        const T zj_im = -z_im[half - k];

        const T e_re = half_scale * (zk_re + zj_re);
        const T e_im = half_scale * (zk_im + zj_im);
        const T d_re = half_scale * (zk_re - zj_re);
        const T d_im = half_scale * (zk_im - zj_im);

        const T o_re = d_im;
        const T o_im = -d_re;

        const T wk_re = w_full[k].re;
        const T wk_im = w_full[k].im;
        const T wo_re = wk_re * o_re - wk_im * o_im;
        const T wo_im = wk_re * o_im + wk_im * o_re;

        half_dst[k * dst_stride] = Cplx(e_re + wo_re, e_im + wo_im);
    }
}

template <typename T, typename Cplx>
inline void portable_irfft_untwist_conj(const Cplx* half_src, usize src_stride, usize n, const Cplx* w_full,
                                        T* z_re, T* z_im) noexcept {
    const usize half = n >> 1;
    const T x0 = half_src[0].re;
    const T xh = half_src[half * src_stride].re;
    z_re[0] = T(0.5) * (x0 + xh);
    z_im[0] = -T(0.5) * (x0 - xh); // conj(Z[0])

    for (usize k = 1; k < half; ++k) {
        const Cplx xk = half_src[k * src_stride];
        const Cplx xj = half_src[(half - k) * src_stride];
        const T xj_re = xj.re;
        const T xj_im_conj = -xj.im;

        const T e_re = T(0.5) * (xk.re + xj_re);
        const T e_im = T(0.5) * (xk.im + xj_im_conj);
        const T d_re = T(0.5) * (xk.re - xj_re);
        const T d_im = T(0.5) * (xk.im - xj_im_conj);

        const T w_re = w_full[k].re;
        const T w_im = w_full[k].im;

        // i * conj(W[k]) * D = (w_im * d_re - w_re * d_im) + i * (w_re * d_re + w_im * d_im)
        const T io_re = w_im * d_re - w_re * d_im;
        const T io_im = w_re * d_re + w_im * d_im;

        // Store conj(Z[k]) = (e_re + io_re, -(e_im + io_im))
        z_re[k] = e_re + io_re;
        z_im[k] = -(e_im + io_im);
    }
}

template <typename T>
inline void portable_irfft_store_interleaved(const T* z_re, const T* z_im, usize half, T post_scale,
                                             T* real_dst, usize dst_stride) noexcept {
    const T neg_scale = -post_scale;
    if (dst_stride == 1) {
        for (usize m = 0; m < half; ++m) {
            real_dst[2 * m] = z_re[m] * post_scale;
            real_dst[2 * m + 1] = z_im[m] * neg_scale;
        }
    } else {
        for (usize m = 0; m < half; ++m) {
            real_dst[(2 * m) * dst_stride] = z_re[m] * post_scale;
            real_dst[(2 * m + 1) * dst_stride] = z_im[m] * neg_scale;
        }
    }
}

} // namespace keel::detail
