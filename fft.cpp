#include "poly.h"
#include "fft.h"

template <class FFT_traits>
void invfft(poly_t *p, long target_size) {
    long p0, p1, e2;
    FFT_traits::fft_group(p0, p1, e2, target_size);
    FFT_traits::invfft_impl(p, p0, p1, e2);
}

template <class FFT_traits>
void fft(poly_t *p, long target_size, long post, poly_t *dst) {
    long p0, p1, e2;
    FFT_traits::fft_group(p0, p1, e2, target_size);
    FFT_traits::fft_impl(p, p0, p1, e2, post, dst);
}

template <class FFT_traits>
void tg2_gpu(poly_t *p, long target_size) {
    long p0, p1, e2;
    FFT_traits::fft_group(p0, p1, e2, target_size);
    if (e2 <= 26) {
        lg_err("fft64::tg2_gpu: e2 <= 26 not implemented");
        abort();
    }
    FFT_traits::gpu_tg2(p, p0, p1, e2);
}

template void invfft<fft64>(poly_t *p, long target_size);
template void fft<fft64>(poly_t *p, long target_size, long post, poly_t *dst);
template void tg2_gpu<fft64>(poly_t *p, long target_size);

uint64_t fft64::root(uint64_t x, long e) {
    if (x == 0ULL) return 0ULL;

    if (e == 65537) {
        return pow(x, 281470681743360ULL) == 1 ? pow(x, 140737488289793ULL) : -1ULL;
    }
    if (e == 257) {
        return pow(x, 71777214277877760ULL) == 1 ? pow(x, 53902732901285633ULL) : -1ULL;
    }
    if (e == 17) {
        return pow(x, 1085102592318504960ULL) == 1 ? pow(x, 957443463810445553ULL) : -1ULL;
    }
    if (e == 5) {
        return pow(x, 3689348813882916864ULL) == 1 ? pow(x, 737869762776583373ULL) : -1ULL;
    }
    if (e == 3) {
        return pow(x, 6148914689804861440ULL) == 1 ? pow(x, 4099276459869907627ULL) : -1ULL;
    }
    if (e == 2) {
        if (pow(x, 9223372034707292160ULL) != 1) return -1ULL;
        uint64_t g = 1753635133440165772ULL;
        uint64_t p = g;
        uint64_t r = pow(x, 4294967295ULL);
        uint64_t acc = 0ULL;
        for (long l = 30; l >= 0; l--) {
            p = mul(p, p);
            if (pow(r, 1ULL << l) != 1) {
                r = mul(r, p);
                acc += (1 << (30 - l));
            }
        }
        return mul(pow(x, 2147483648ULL), pow(g, acc));
    }
    lg_err("fft64::root: not implemented for %ld", e);
    return -1ULL;
}

uint64_t *gen_gp_rev(uint64_t gp, long nbits) {
    uint64_t *ret;
    long ret_len = 1UL << nbits;
    if (ret_len > poly_t::page_size / sizeof(uint64_t)) {
        ret_len = poly_t::page_size / sizeof(uint64_t);
    }
    int alloc_ret = posix_memalign((void **)&ret, 256, ret_len * sizeof(uint64_t));
    if (alloc_ret != 0) {
        lg_err("fft64::gen_gp_rev: posix_memalign failed");
        return NULL;
    }

    for (long i = 0; i < ret_len; i++) {
        ret[i] = fft64::pow(gp, bit_rev(i, nbits));
    }
    return ret;
}

long fft64::fft_group(long &p0, long &p1, long &e2, long size) {
    if (size <= (1LL << 32)) {
        p0 = 1; p1 = 1; e2 = 64 - __builtin_clzll(size - 1);
    } else if (size <= (5LL << 30)) {
        p0 = 5; p1 = 1; e2 = 30;
    } else if (size <= (5LL << 31)) {
        p0 = 5; p1 = 1; e2 = 31;
    } else if (size <= (3LL << 32)) {
        p0 = 3; p1 = 1; e2 = 32;
    } else if (size <= (15LL << 30)) {
        p0 = 5; p1 = 3; e2 = 30;
    } else if (size <= (5LL << 32)) {
        p0 = 5; p1 = 1; e2 = 32;
    } else if (size <= (17LL << 31)) {
        p0 = 17; p1 = 1; e2 = 31;
    } else if (size <= (17LL << 32)) {
        p0 = 17; p1 = 1; e2 = 32;
    } else if (size <= (51LL << 31)) {
        p0 = 17; p1 = 3; e2 = 31;
    } else if (size <= (85LL << 31)) {
        p0 = 17; p1 = 5; e2 = 31;
    } else if (size <= (85LL << 32)) {
        p0 = 17; p1 = 5; e2 = 32;
    } else if (size <= (257LL << 31)) {
        p0 = 257; p1 = 1; e2 = 31;
    } else if (size <= (257LL << 32)) {
        p0 = 257; p1 = 1; e2 = 32;
    } else if (size <= (1285LL << 31)) {
        p0 = 257; p1 = 5; e2 = 31;
    } else if (size <= (1285LL << 32)) {
        p0 = 257; p1 = 5; e2 = 32;
    } else {
        lg_err("fft64::fft_group: not implemented for %ld", size);
        return -1;
    }
    return 0;
}

long fft64::fft_pages(long size) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);

    long p0, p1, e2;
    fft_group(p0, p1, e2, size);
    long fft_size = (p0 * p1) << e2;
    
    return (fft_size - 1) / page_u64 + 1L;
}

long fft64::next_smaller_pages(long size) {
    if (size < 2) return 1;
    if (size < (1LL << 32)) {
        return fft_pages(1LL << (63 - __builtin_clzll(size - 1)));
    } else if (size < (5LL << 30)) {
        return fft_pages(1LL << 32);
    } else if (size < (5LL << 31)) {
        return fft_pages(5LL << 30);
    } else if (size < (3LL << 32)) {
        return fft_pages(5LL << 31);
    } else if (size < (15LL << 30)) {
        return fft_pages(3LL << 32);
    } else if (size < (5LL << 32)) {
        return fft_pages(15LL << 30);
    } else if (size < (17LL << 31)) {
        return fft_pages(5LL << 32);
    } else if (size < (17LL << 32)) {
        return fft_pages(17LL << 31);
    } else if (size < (51LL << 31)) {
        return fft_pages(17LL << 32);
    } else if (size < (85LL << 31)) {
        return fft_pages(51LL << 31);
    } else if (size < (85LL << 32)) {
        return fft_pages(85LL << 31);
    } else if (size < (257LL << 31)) {
        return fft_pages(85LL << 32);
    } else if (size < (257LL << 32)) {
        return fft_pages(257LL << 31);
    } else if (size < (1285LL << 31)) {
        return fft_pages(257LL << 32);
    } else if (size < (1285LL << 32)) {
        return fft_pages(1285LL << 31);
    } else {
        return fft_pages(1285LL << 32);
    }
}

long fft64::fft_impl(poly_t *p, long p0, long p1, long e2, long post, poly_t *dst) {
    constexpr uint64_t page_u64 = poly_t::page_size / sizeof(uint64_t);
    
    const long fft_pages = fft64::fft_pages((p0 * p1) << e2);

    if (p0 * p1 != 1) {
        gpu_fft_odd(p, p0, p1, e2);

        const long total_tasks = fft_pages / p1 / p0;

        if (p1 != 1) {
            int32_t *pages = (int32_t *) malloc(sizeof(int32_t) * fft_pages);
            for (long i = 0; i < fft_pages; i++) {
                pages[i] = p->page[i];
            }
            
            for (long i = 0; i < p1 * p0; i++) {
                for (long k = 0; k < total_tasks; k++) {
                    p->page[k + total_tasks * ((i % p1) * p0 + (i / p1))] = pages[k + total_tasks * i];
                }
            }

            free(pages);
        }
    }

    const uint64_t gp = pow(g, (modulus - 1) >> e2);
    const uint64_t gg = pow(g, (modulus - 1) / (p0 * p1 * (1UL << e2)));
    uint64_t *gp_rev = gen_gp_rev(gp, e2 - 1);
    uint64_t *gp_revf = post == 2 ? gen_gp_rev(gp, e2) : NULL;

    const long Tpages = fft_pages / (p0 * p1);
    chunk_t *cks = (chunk_t *) malloc(sizeof(chunk_t) * Tpages);
    for (long T = 0; T < p0 * p1; T++) {
        for (long i = 0; i < Tpages; i++) {
            cks[i] = *cm.fetch(p->page[i + T * Tpages]);
        }
        for (long i = 0; i < Tpages && T != p0 * p1 - 1; i++) {
            cm.prefetch(p->page[i + (T + 1) * Tpages]);
        }

        if (e2 >= 25) {
            gpu_fft_even(cks, T, p0, p1, e2, gp_rev);
        } else {
            cpu_fft_even(cks, T, p0, p1, e2, gp_rev);
        }
        

        if (post == 1) {
            #pragma omp parallel for schedule(static)
            for (long task = 0; task < Tpages; task++) {
                chunk_t *ck_dst = cm.fetch(dst->page[task + T * Tpages]);
                for (long l = 0; l < page_u64 && l < (1L << e2); l++) {
                    ck_dst->data[l] = mul(ck_dst->data[l], cks[task].data[l]);
                }
                cm.release_sync(ck_dst->id);
                cm.release_del(cks[task].id);
            }
        }

        if (post == 2) {
            #pragma omp parallel for schedule(static)
            for (long task = 0; task < Tpages; task++) {
                chunk_t *ck_dst = cm.fetch(dst->page[task + T * Tpages]);
                const uint64_t gt = pow(gg, T + p0 * p1 * bit_rev(task * page_u64, e2));
                for (long l = 0; l < page_u64 && l < (1L << e2); l++) {
                    uint64_t tmp0 = mul(ck_dst->data[l], ck_dst->data[l]);
                    uint64_t tmp1 = mul(cks[task].data[l], cks[task].data[l]);
                    ck_dst->data[l] = sub(tmp0, mul(tmp1, mul(gp_revf[l], gt)));
                }
                cm.release_sync(ck_dst->id);
                cm.release_del(cks[task].id);
            }
        }

        if (!post) {
            for (long i = 0; i < Tpages; i++) {
                cm.release_sync(cks[i].id);
            }
        }
    }
    free(cks);
    free(gp_rev);
    if (post == 2) free(gp_revf);

    if (post) p->pages = 0;

    return 0;
}

long fft64::invfft_impl(poly_t *p, long p0, long p1, long e2) {
    constexpr uint64_t page_u64 = poly_t::page_size / sizeof(uint64_t);

    const long fft_pages = fft64::fft_pages((p0 * p1) << e2);
    uint64_t *gpinv_rev = gen_gp_rev(pow(inv(g), (modulus - 1) >> e2), e2 - 1);

    const long Tpages = fft_pages / (p0 * p1);
    chunk_t *cks = (chunk_t *) malloc(sizeof(chunk_t) * Tpages);
    for (long T = 0; T < p0 * p1; T++) {
        for (long i = 0; i < Tpages; i++) {
            cks[i] = *cm.fetch(p->page[i + T * Tpages]);
        }
        for (long i = 0; i < Tpages && T != p0 * p1 - 1; i++) {
            cm.prefetch(p->page[i + (T + 1) * Tpages]);
        }
        
        if (e2 >= 25) {
            gpu_invfft_even(cks, T, p0, p1, e2, gpinv_rev);
        } else {
            cpu_invfft_even(cks, T, p0, p1, e2, gpinv_rev);
        }

        for (long i = 0; i < Tpages; i++) {
            cm.release_sync(cks[i].id);
        }
    }
    free(cks);
    free(gpinv_rev);

    if (p0 * p1 != 1) {
        const long total_tasks = fft_pages / p1 / p0;

        if (p1 != 1) {
            int32_t *pages = (int32_t *) malloc(sizeof(int32_t) * fft_pages);
            for (long i = 0; i < fft_pages; i++) {
                pages[i] = p->page[i];
            }
            
            for (long i = 0; i < p1 * p0; i++) {
                for (long k = 0; k < total_tasks; k++) {
                    p->page[k + total_tasks * i] = pages[k + total_tasks * ((i % p1) * p0 + (i / p1))];
                }
            }

            free(pages);
        }

        gpu_invfft_odd(p, p0, p1, e2);
    }

    return 0;
}
