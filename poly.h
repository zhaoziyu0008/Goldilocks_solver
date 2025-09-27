#ifndef __POLY_H
#define __POLY_H

#include "manager.hpp"

struct poly_t {
    static constexpr long page_size = resource.page_size;
    
    poly_t(long max_pages = -1);
    ~poly_t();

    int copy_from(const poly_t& other);

    int alloc_page(long target_pages = -1);
    int free_page(long target_pages = -1);

    int load(const char *dir, int rm_src = 0);
    int sync(const char *dir, int rm_src = 0);

    long pages = 0;
    int *page = NULL;
    private:
    long _max_pages;
};

/// will destroy p
template <class FFT_traits>
void enum_roots(long &num_ret, uint64_t *ret, poly_t *p, long period);

/// @brief find all common roots of p and x^period - ratio, destroy p
template<class FFT_traits>
uint64_t *roots(poly_t *p, long target_size, long period, uint64_t ratio = 1);

/// @brief mod p by x^period - ratio, destroy unnecessary pages of p
template <class FFT_traits>
void period_red(poly_t *p, long target_size, long period, uint64_t ratio = 1);

template <class FFT_traits>
void invfft(poly_t *p, long target_size);

/// post 1 for mul, 2 for tg2，destory p if post
template <class FFT_traits>
void fft(poly_t *p, long target_size, long post = 0, poly_t *dst = NULL);

template <class FFT_traits>
void tg2(poly_t *p, long target_size);

template <class FFT_traits>
void tg2_gpu(poly_t *p, long target_size);

/// do not destroy p, extra allocated chunks in dst remains untouched
template <class FFT_traits>
void compose(poly_t *dst, poly_t *p, long target_size, uint64_t ratio);

/// will allocate pages for dst, destroy p
template <class FFT_traits>
void collect(poly_t *dst, poly_t *p, long target_size, long tg);

template <class FFT_traits>
void tg3(poly_t *p, long target_size);

template <class FFT_traits>
void tg5(poly_t *p, long target_size);

template <class FFT_traits>
void tg17(poly_t *p, long target_size);

template <class FFT_traits>
void tg257(poly_t *p, long target_size);

inline uint32_t bit_rev(uint32_t x, int nbits) {
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16);
    return x >> (32 - nbits);
}

/// will allocate pages for p
template<class Oracle, class FFT_traits>
void eval(poly_t *p, long target_size) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);
    const long fft_pages    = FFT_traits::fft_pages(target_size);

    p->alloc_page(fft_pages);

    long pp, p0, p1, e2;
    FFT_traits::fft_group(p0, p1, e2, target_size);
    pp = p0 * p1;
    const uint64_t g = FFT_traits::pow(FFT_traits::g, (FFT_traits::modulus - 1) / pp / (1UL << e2));

    uint64_t *page_bias, *task_bias;
    assert(posix_memalign((void **)&page_bias, poly_t::page_size, poly_t::page_size) == 0);
    assert(posix_memalign((void **)&task_bias, 256, fft_pages * sizeof(uint64_t)) == 0);
    
    #pragma omp parallel for
    for (uint32_t i = 0; i < page_u64; i++) {
        page_bias[i] = FFT_traits::pow(g, pp * bit_rev(i, e2));
    }

    #pragma omp parallel for
    for (long task = 0; task < fft_pages; task++) {
        long page_head = task * page_u64;
        uint64_t bias = FFT_traits::pow(g, (page_head / (1UL << e2)) + pp * bit_rev(page_head % (1UL << e2), e2));
        task_bias[task] = bias;
    }

    FFT_traits::template gpu_eval<Oracle>(p, fft_pages, task_bias, page_bias);

    free(task_bias);
    free(page_bias);

    /// in poseidon, we never need it
    if (e2 < __builtin_ctzll(page_u64)) {
        chunk_t *ck_p0 = cm.fetch(p->page[0]);
        memset(ck_p0->data + (1UL << e2), 0, (page_u64 - (1UL << e2)) * sizeof(uint64_t));
        cm.release_sync(p->page[0]);
    }
}

template<class FFT_traits>
void check_sol(poly_t *p, long target_size, uint64_t ratio) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);
    const long total_tasks = p->pages;
    uint64_t res = 0;
    pthread_spinlock_t lock;
    pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED);

    #pragma omp parallel for
    for (long task = 0; task < total_tasks; task++) {
        uint64_t tmp = 0;
        chunk_t *ck = cm.fetch(p->page[task]);
        for (long j = page_u64 - 1; j >= 0; j--) {
            uint64_t val = ck->data[j];
            tmp = FFT_traits::add(val, FFT_traits::mul(ratio, tmp));
        }
        tmp = FFT_traits::mul(tmp, FFT_traits::pow(ratio, task * page_u64));
        cm.release(p->page[task]);

        pthread_spin_lock(&lock);
        res = FFT_traits::add(res, tmp);
        pthread_spin_unlock(&lock);
    }

    pthread_spin_destroy(&lock);
    if (res == 0) {
        lg_dbg("current solution: %016lx", ratio);
    } else {
        lg_dbg("solution %016lx not found, res = %016lx", ratio, res);
        abort();
    }
}

template<class Oracle, class FFT_traits>
uint64_t root() {
    #define G(x) (FFT_traits::fft_pages(x) * poly_t::page_size / 8L * 1e-9)
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);
    long size = Oracle::degree + 1;
    uint64_t order = 0xffffffff00000000ULL;

    poly_t f(FFT_traits::fft_pages(size));
    timer.record();
    eval<Oracle, FFT_traits>(&f, size);
    lg_info("eval(%.3fG) done, %.3fs elapsed, %.3f GHash/s", G(size), timer.elapsed(), G(size) / timer.elapsed());
    timer.record();
    invfft<FFT_traits>(&f, size);
    lg_info("invfft(%.3fG) done, %.3fs elapsed", G(size), timer.elapsed());

    for (; order % 2 == 0; order /= 2) {
        timer.record();
        if (order <= FFT_traits::next_smaller_pages(size) * page_u64) {
            period_red<FFT_traits>(&f, size, order);
            lg_info("period_red done, %.3fs elapsed, size %.3f G --> %.3f G", timer.elapsed(), G(size), G(order));
            timer.record();
            size = order;
        }

        tg2_gpu<FFT_traits>(&f, size);
        lg_info("tg2(%.3fG) done, %.3fs elapsed, curr order %ld", G(size), timer.elapsed(), order / 2);
    }
	f.sync(".tmp");

    timer.record();
    uint64_t *odd_parts = roots<FFT_traits>(&f, size, order, 1);
    lg_info("odd_parts done, %.3fs elapsed", timer.elapsed());
    
    uint64_t ret = -1ULL;
    for (long i = 0; odd_parts[i] != -1ULL; i++) {
        lg_info("try odd_parts[%ld] = %016lx", i, odd_parts[i]);
        timer.record();
        eval<Oracle, FFT_traits>(&f, Oracle::degree + 1);
        lg_info("eval(%.3fG) done, %.3fs elapsed, %.3f GHash/s", G(Oracle::degree + 1), 
                timer.elapsed(), G(Oracle::degree + 1) / timer.elapsed());
        timer.record();
        invfft<FFT_traits>(&f, Oracle::degree + 1);
        lg_info("invfft(%.3fG) done, %.3fs elapsed", G(Oracle::degree + 1), timer.elapsed());
        timer.record();
        
        uint64_t *rlist = roots<FFT_traits>(&f, Oracle::degree + 1, 1ULL << 32, odd_parts[i]);
        lg_info("roots done, %.3fs elapsed", timer.elapsed());
        
        ret = rlist[0];
        free(rlist);

        if (ret != -1ULL) break;
    }
    if (ret != -1ULL) lg_info("found root %016lx", ret);
    else lg_info("no root found");

    free(odd_parts);

    return ret;
}

#endif