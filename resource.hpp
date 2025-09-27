#ifndef __RESOURCE_HPP
#define __RESOURCE_HPP

#include <pthread.h>
#include <stdlib.h>
#include <assert.h>

struct resource_holder_t {
private:
    resource_holder_t();
    ~resource_holder_t();

public:
    static constexpr long page_size = 2097152; // 2M
    static constexpr long ram_pages = 225000;  // 0.45T
    static constexpr long ssd_pages = 0; // 96T
    static constexpr long ram_only  = 1;
    
    static resource_holder_t& instance() {
        static resource_holder_t instance;
        return instance;
    }

    long alloc_page(void **ptr) {
        pthread_spin_lock(&_pages_lock);
        assert(_num_pages > 0);
        *ptr = _pages[--_num_pages];
        pthread_spin_unlock(&_pages_lock);
        return 0;
    }

    long free_page(void *ptr) {
        long bias = (long)ptr - (long)_memory_pool;
        assert(bias >= 0 && bias < page_size * ram_pages && bias % page_size == 0);
        pthread_spin_lock(&_pages_lock);
        assert(_num_pages < ram_pages);
        _pages[_num_pages++] = ptr;
        pthread_spin_unlock(&_pages_lock);
        return 0;
    }

    resource_holder_t(const resource_holder_t&) = delete;
    resource_holder_t& operator=(const resource_holder_t&) = delete;

private:
    void *_memory_pool = NULL;
    void      **_pages = NULL;
    long    _num_pages = 0;
    pthread_spinlock_t _pages_lock;
};

inline resource_holder_t& resource = resource_holder_t::instance();


/// use Marc Steven's G6K thread pool now
#include "thread_pool.hpp"

#include <sys/time.h>

struct my_timer_t {
private:
    my_timer_t() {
        gettimeofday(&_last_record, NULL);
    }

    ~my_timer_t() {}
public:
    static my_timer_t& instance() {
        static my_timer_t instance;
        return instance;
    }

    void record() {
        gettimeofday(&_last_record, NULL);
    }

    double elapsed() {
        struct timeval now;
        gettimeofday(&now, NULL);
        return (now.tv_sec - _last_record.tv_sec) + (now.tv_usec - _last_record.tv_usec) / 1e6;
    }

    my_timer_t (const my_timer_t&) = delete;
    my_timer_t& operator=(const my_timer_t&) = delete;

private:
    struct timeval _last_record;
};

inline my_timer_t& timer = my_timer_t::instance();

#endif