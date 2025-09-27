#ifndef __FFT_H
#define __FFT_H

#include <stdint.h>
#include <cstddef>

struct poly_t;

/// supported groups:
/// 24bit - deg 7^12 - FFT size 2^30 * 3 * 5
/// 28bit - deg 7^13 - FFT size 2^31 * 3 * 17
/// 32bit - deg 7^15 - FFT size 2^32 * 5 * 257
/// (p - 1) / 2^23   - FFT size 2^31 * 5 * 257
/// (p - 1) / 2^24   - FFT size 2^32 * 257
/// (p - 1) / 2^25   - FFT size 2^31 * 257
/// (p - 1) / 2^26   - FFT size 2^32 * 5 * 17
/// (p - 1) / 2^27   - FFT size 2^31 * 5 * 17
/// (p - 1) / 2^28   - FFT size 2^32 * 17
/// (p - 1) / 2^29   - FFT size 2^31 * 17
/// (p - 1) / 2^30   - FFT size 2^32 * 5
/// (p - 1) / 2^31   - FFT size 2^31 * 5
/// (p - 1) / 2^32   - FFT size 2^32
/// (p - 1) / 2^32 * 3 - FFT size 2^32 * 3
/// (p - 1) / 2^32 / 3 * 5 - FFT size 2^31 * 5
/// (p - 1) / 2^32 / 3 / 5 * 17 - FFT size 2^30 * 5
/// (p - 1) / 2^32 / 3 / 5 / 17 * 257 -  FFT size 2^30 * 5
/// FFT size 2^e for 1 <= e <= 32


struct fft64 {
    static constexpr uint64_t modulus = 0xffffffff00000001ULL;
    static constexpr inline uint64_t add(uint64_t x, uint64_t y) {
        if (x >= modulus - y) {
            return x + y - modulus;
        } else {
            return x + y;
        }
    }
    static constexpr inline uint64_t sub(uint64_t x, uint64_t y) {
        if (x >= y) {
            return x - y;
        } else {
            return x + modulus - y;
        }
    }
    static constexpr inline uint64_t mul(uint64_t x, uint64_t y) {
        __uint128_t tmp = (__uint128_t)x * y;
        return (uint64_t)(tmp % modulus);
    }
    static inline uint64_t inv(uint64_t x) {
        if (x == 0) return 0;
        if (x == 1) return 1;

        __int128_t u = 1, v = 0;
        uint64_t a = x, b = modulus, ret = 0;

        for (;;) {
            uint64_t t = b / a;
            b -= t * a;
            v -= t * u;

            if (b == 1) {
                v %= modulus;
                if (v < 0) v += modulus;
                ret = v;
                break;
            }

            t = a / b;
            a -= t * b;
            u -= t * v;

            if (a == 1) {
                u %= modulus;
                if (u < 0) u += modulus;
                ret = u;
                break;
            }
        }

        return ret;
    }
    static constexpr inline uint64_t pow(uint64_t x, uint64_t e) {
        uint64_t ret = 1;
        while (e) {
            if (e & 1) ret = mul(ret, x);
            x = mul(x, x);
            e >>= 1;
        }
        return ret;
    }
    static uint64_t root(uint64_t x, long e);
    static inline uint64_t rou(long e) {
        if (e == 2) return rou2;
        if (e == 3) return rou3;
        if (e == 5) return rou5;
        if (e == 17) return rou17;
        if (e == 257) return rou257;
        if (e == 65537) return rou65537;
        return -1ULL;
    }

    static constexpr long roots_limit = 1023;
    static constexpr uint64_t rou2   = 0xffffffff00000000ULL;
    static constexpr uint64_t rou3   = 0xffffffffULL;
    static constexpr uint64_t rou5   = 0x2efb5c2a6f35241ULL;
    static constexpr uint64_t rou17  = 0x2eac242c9d116ddULL;
    static constexpr uint64_t rou257 = 0x14b76c62b329dd9ULL;
    static constexpr uint64_t rou65537 = 0xffffb50ee138261fULL;
    static constexpr uint64_t g = 0x7ULL;
    
    static long fft_group(long &p0, long &p1, long &e2, long size);
    static long fft_pages(long size);
    static long next_smaller_pages(long size);
    static long fft_impl(poly_t *p, long p0, long p1, long e2, long post = 0, poly_t *dst = NULL);
    static long invfft_impl(poly_t *p, long p0, long p1, long e2);
    template <class Oracle>
    static void gpu_eval(poly_t *p, long total_tasks, uint64_t *task_bias, uint64_t *page_bias);

    static void gpu_invfft_odd(poly_t *p, long p0, long p1, long e2);
    static void cpu_invfft_even(chunk_t *ck, long T, long p0, long p1, long e2, uint64_t *gpinv_rev);
    static void gpu_invfft_even(chunk_t *ck, long T, long p0, long p1, long e2, uint64_t *gpinv_rev);
    static void gpu_fft_odd(poly_t *p, long p0, long p1, long e2);
    static void cpu_fft_even(chunk_t *ck, long T, long p0, long p1, long e2, uint64_t *gp_rev);
    static void gpu_fft_even(chunk_t *ck, long T, long p0, long p1, long e2, uint64_t *gp_rev);
    static void gpu_tg2(poly_t *p, long p0, long p1, long e2);
};



#endif