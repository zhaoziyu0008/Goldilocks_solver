#ifndef __GPU_IMPL_CUH
#define __GPU_IMPL_CUH

#include "resource.hpp"

#include <cuda_runtime.h>

#define CHECK_CUDA_ERR(val) do {                                        \
    cudaError_t err = val;                                              \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d \"%s\", code = %d(%s)\n",  \
        __FILE__, __LINE__, #val, err, cudaGetErrorString(err));        \
        cudaGetLastError();                                             \
        abort();                                                        \
    }                                                                   \
} while (0)

__device__ __forceinline__ uint64_t cuadd(uint64_t x, uint64_t y) {
    if (x >= 0xffffffff00000001ULL - y) {
        return x + y - 0xffffffff00000001ULL;
    } else {
        return x + y;
    }
}

__device__ __forceinline__ uint64_t cusub(uint64_t x, uint64_t y) {
    if (x >= y) {
        return x - y;
    } else {
        return x + 0xffffffff00000001ULL - y;
    }
}

__device__ __forceinline__ uint64_t cumul(uint64_t x, uint64_t y) {
    uint32_t x_lo = x & 0xffffffff;
    uint32_t x_hi = x >> 32;
    uint32_t y_lo = y & 0xffffffff;
    uint32_t y_hi = y >> 32;
    uint64_t xhyh = (uint64_t)x_hi * y_hi;
    uint64_t xlyh = (uint64_t)x_lo * y_hi;
    uint64_t xhyl = (uint64_t)x_hi * y_lo;
    uint64_t xlyl = (uint64_t)x_lo * y_lo;
    xhyh += xhyl >> 32;
    xhyh += xlyh >> 32;
    uint32_t nh96 = xhyh >> 32;
    uint32_t ph64 = xhyh & 0xffffffff;
    uint64_t tt32 = (xlyh & 0xffffffff) + (xlyl >> 32) + (uint64_t) ph64 + (xhyl & 0xffffffff);
    uint64_t acc = (uint64_t) 0xffffffff * (uint32_t)(tt32 >> 32);
    acc += (uint32_t) (xlyl & 0xffffffff);
    acc -= (uint64_t) nh96 + ph64;
    if (acc & 0x8000000000000000ULL) acc += 0xffffffff00000001ULL;
    uint64_t th32 = tt32 << 32;
    if (acc >= 0xffffffff00000001ULL - th32) {
        acc += th32 - 0xffffffff00000001ULL;
    } else {
        acc += th32;
    }
    return acc;
}

__device__ __forceinline__ unsigned int _lane_id() {
    unsigned int ret; 
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ unsigned int _thread_id() {
    unsigned int ret; 
    asm volatile ("mov.u32 %0, %tid.x;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ void _ldgsts_128b_async(void *dst, void *src) {
    int dst_sh_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, 16;\n"
        :
        : "r"(dst_sh_addr), "l"(src)
    );
}

__device__ __forceinline__ void _ldgsts_64b_async(void *dst, void *src) {
    int dst_sh_addr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8, 8;\n"
        :
        : "r"(dst_sh_addr), "l"(src)
    );
}

__device__ __forceinline__ void _commit_async_group() {
    asm volatile(
        "cp.async.commit_group;\n"
        :
        :
    );
}

__device__ __forceinline__ void _wait_async_group() {
    asm volatile(
        "cp.async.wait_group 0;\n"
        :
        :
    );
}

template <uint32_t p>
__global__ void kernel_odd(uint64_t *dst, uint32_t size, uint64_t *col0, uint64_t *multipliers);


struct cuda_thread_t {
    cuda_thread_t() : thread(1) {}
    ~cuda_thread_t() {}

    int device_id;
    cudaStream_t stream;

    uint64_t *d_buf;
    uint64_t  d_buf_size;
    thread_pool::thread_pool thread;
};

struct cuda_thread_pool_t {
    static constexpr long ctp_devices = 8;
    static constexpr long ctp_threads = 32;

    cuda_thread_t worker[ctp_threads];

    cuda_thread_pool_t(const cuda_thread_pool_t&) = delete;
    cuda_thread_pool_t& operator=(const cuda_thread_pool_t&) = delete;

    static cuda_thread_pool_t& instance() {
        static cuda_thread_pool_t instance;
        return instance;
    }

    void wait() {
        for (long i = 0; i < ctp_threads; i++) {
            worker[i].thread.wait_sleep();
        }
    }

private:
    cuda_thread_pool_t() {
        for (long i = 0; i < ctp_threads; i++) {
            worker[i].device_id = i % ctp_devices;
            worker[i].d_buf_size = resource.page_size * 1296L;
            worker[i].thread.push([this, i]() {
                CHECK_CUDA_ERR(cudaSetDevice(worker[i].device_id));
                CHECK_CUDA_ERR(cudaStreamCreate(&worker[i].stream));
                CHECK_CUDA_ERR(cudaMalloc((void**)&worker[i].d_buf, worker[i].d_buf_size));
                CHECK_CUDA_ERR(cudaFuncSetAttribute(kernel_odd<257>, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * 257));
            });
        }
        for (long i = 0; i < ctp_threads; i++) {
            worker[i].thread.wait_sleep();
        }
    }

    ~cuda_thread_pool_t() {
        for (long i = 0; i < ctp_threads; i++) {
            worker[i].thread.push([&, i]() {
                /// CHECK_CUDA_ERR(cudaStreamSynchronize(worker[i].stream));
                /// CHECK_CUDA_ERR(cudaStreamDestroy(worker[i].stream));
                /// CHECK_CUDA_ERR(cudaFree(worker[i].d_buf));
            });
        }
    }
            
};

static inline cuda_thread_pool_t& ctp = cuda_thread_pool_t::instance();



#endif