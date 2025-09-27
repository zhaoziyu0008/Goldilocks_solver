#include "gpu_impl.cuh"

#include "poly.h"
#include "oracles.hpp"

resource_holder_t::resource_holder_t() {
    assert(posix_memalign(&_memory_pool, page_size, page_size * ram_pages) == 0);
    assert(posix_memalign((void **)&_pages, 4096, sizeof(void *) * ram_pages) == 0);

    CHECK_CUDA_ERR(cudaHostRegister(_memory_pool, page_size * ram_pages, cudaHostRegisterPortable));
    
    for (long i = 0; i < ram_pages; i++) {
        _pages[i] = (void *)((char *)_memory_pool + page_size * i);
    }

    _num_pages = ram_pages;
    pthread_spin_init(&_pages_lock, PTHREAD_PROCESS_SHARED);
}

resource_holder_t::~resource_holder_t() {
    assert(_num_pages == ram_pages);
    pthread_spin_destroy(&_pages_lock);
    free(_pages);

    /// CHECK_CUDA_ERR(cudaHostUnregister(_memory_pool));
    /// free(_memory_pool);
}


constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);
constexpr long eval_blocks = 64;
constexpr long eval_threads = 256;

__device__ __forceinline__ void culinearE(uint64_t state[8]) {
    uint64_t tmp[8] = {};
    for (long i = 0; i < 8; i++) {
        for (long j = 0; j < 8; j++) {
            tmp[i] = cuadd(tmp[i], cumul(MDSE[i][j], state[j]));
        }
    }
    for (long i = 0; i < 8; i++) state[i] = tmp[i];
}

template <class Oracle>
__device__ __forceinline__ void culinearI(uint64_t state[8]) {
    uint64_t sum = 0;
    for (long i = 0; i < 8; i++) {
        sum = cuadd(sum, state[i]);
    }
    for (long i = 0; i < 8; i++) {
        state[i] = cuadd(sum, cumul(Oracle::get_MDSI(i), state[i]));
    }
}

__device__ __forceinline__ void cusbox(uint64_t &x) {
    uint64_t y = cumul(x, x);
    uint64_t z = cumul(y, y);
    x = cumul(x, cumul(y, z));
}

template <class Oracle>
__global__ void kernel_eval(uint64_t *dst, uint64_t *page_bias, uint64_t task_bias) {
    constexpr uint32_t R_F = Oracle::R_F;
    constexpr uint32_t R_P = Oracle::R_P;
    constexpr uint32_t stride = eval_blocks * eval_threads;

    __shared__ uint64_t sh_buf[eval_threads * 2];

    uint32_t ind = threadIdx.x + blockIdx.x * eval_threads;
    uint32_t sh_bias = 0;

    _ldgsts_64b_async(sh_buf + threadIdx.x, page_bias + ind);
    _commit_async_group();
    _wait_async_group();

    while (ind < (uint32_t)page_u64) {
        uint64_t x = sh_buf[sh_bias + threadIdx.x];
        sh_bias ^= eval_threads;
        if (ind + stride < page_u64) {
            _ldgsts_64b_async(sh_buf + sh_bias + threadIdx.x, page_bias + ind + stride);
            _commit_async_group();
        }

        x = cumul(x, task_bias);

        uint64_t state[8];
        state[0] = x;
        state[1] = cumul(Oracle::LX1, x);
        state[2] = cumul(Oracle::LX2, x);
        state[3] = state[4] = state[5] = state[6] = 0;
        state[7] = Oracle::CX7;

        int RC_counter = 8;

        culinearE(state);

        #pragma unroll
        for (uint64_t r = 0; r < R_F / 2 - 1; r++) {
            for (uint64_t i = 0; i < 8; i++) {
                state[i] = cuadd(state[i], Oracle::get_RC(RC_counter++));
            }
            for (uint64_t i = 0; i < 8; i++) {
                cusbox(state[i]);
            }
            culinearE(state);
        }

        #pragma unroll
        for (uint64_t r = 0; r < R_P; r++) {
            state[0] = cuadd(state[0], Oracle::get_RC(RC_counter++));
            cusbox(state[0]);
            culinearI<Oracle>(state);
        }

        #pragma unroll
        for (uint64_t r = 0; r < R_F / 2; r++) {
            for (uint64_t i = 0; i < 8; i++) {
                state[i] = cuadd(state[i], Oracle::get_RC(RC_counter++));
            }
            for (uint64_t i = 0; i < 8; i++) {
                cusbox(state[i]);
            }
            culinearE(state);
        }

        _wait_async_group();
        dst[ind] = state[7];

        ind += stride;
    }
}

template <class Oracle>
void fft64::gpu_eval(poly_t *p, long total_tasks, uint64_t *task_bias, uint64_t *page_bias) {
    for (long ct = 0; ct < ctp.ctp_threads; ct++) {
        ctp.worker[ct].thread.push([=]() {
            uint64_t *d_dst = ctp.worker[ct].d_buf;
            uint64_t *d_page_bias = ctp.worker[ct].d_buf + page_u64;
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_page_bias, page_bias, page_u64 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
            for (long task = ct; task < total_tasks; task += ctp.ctp_threads) {
                const uint64_t bias = task_bias[task];
                chunk_t *ck = cm.fetch(p->page[task]);
                if (task + 5 * ctp.ctp_threads < total_tasks) {
                    cm.prefetch(p->page[task + 5 * ctp.ctp_threads]);
                }
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_dst, ck->data, page_u64 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                kernel_eval<Oracle><<<eval_blocks, eval_threads, 0, ctp.worker[ct].stream>>>(
                    d_dst, d_page_bias, bias
                );
                CHECK_CUDA_ERR(cudaMemcpyAsync(ck->data, d_dst, page_u64 * 8, 
                            cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
                cm.release_sync(p->page[task]);
            }
        });
    }
    ctp.wait();
}

constexpr long odd_blocks = 64;
constexpr long odd_threads = 256;

template <uint32_t p>
__global__ void kernel_odd(uint64_t *dst, uint32_t size, uint64_t *col0, uint64_t *multipliers) {
    extern __shared__ uint64_t mat32[]; /// of size p * 32 * sizeof(uint64_t)

    __shared__ uint64_t sh_col0[p];
    __shared__ uint64_t sh_multipliers[p];

    int ind = blockIdx.x * 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    if (threadIdx.x < p) {
        sh_col0[threadIdx.x] = col0[threadIdx.x];
        sh_multipliers[threadIdx.x] = multipliers[threadIdx.x];
    }
    if (threadIdx.x + odd_threads < p) {
        sh_col0[threadIdx.x + odd_threads] = col0[threadIdx.x + odd_threads];
        sh_multipliers[threadIdx.x + odd_threads] = multipliers[threadIdx.x + odd_threads];
    }
    __syncthreads();

    while (ind < size) {
        for (int i = warp_id; i < p; i += odd_threads / 32) {
            uint64_t x = dst[ind + lane_id + i * size];
            mat32[i + lane_id * p] = x;
        }
        __syncthreads();
        for (int i = warp_id; i < p; i += odd_threads / 32) {
            uint64_t res = 0;
            uint64_t m = sh_multipliers[i];
            uint64_t w = sh_col0[i];
            for (int j = 0; j < p; j++) {
                res = cuadd(res, cumul(w, mat32[j + lane_id * p]));
                w = cumul(m, w);
            }
            
            dst[ind + lane_id + i * size] = res;
        }
        __syncthreads();
        
        ind += gridDim.x * 32;
    }
}

constexpr long e18_threads = 256;

__global__ void kernel_invfft_e18(uint64_t *dst, uint64_t *twist, uint64_t *twist_bias, uint64_t GSizeinv) {
    __shared__ uint64_t mat32[4096];
    __shared__ uint64_t sh_twist_bias[18];
    __shared__ uint64_t twist11[128];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (threadIdx.x < 128) {
        twist11[threadIdx.x] = twist[threadIdx.x];
    }
    if (threadIdx.x < 18) {
        sh_twist_bias[threadIdx.x] = twist_bias[blockIdx.x * 18 + threadIdx.x];
    }

    uint64_t *block_dst = dst + blockIdx.x * 262144;

    for (int l = 0; l < 262144; l += 2048) {
        for (int i = 0; i < 2048; i += e18_threads) {
            mat32[i + threadIdx.x] = cumul(GSizeinv, block_dst[l + i + threadIdx.x]);
            mat32[i + threadIdx.x + 2048] = twist[l + i + threadIdx.x];
        }
        __syncthreads();

        uint64_t *twist_ptr = mat32 + 2048;
        uint32_t lbs = 0;
        for (int blocks = 1024; blocks >= 1; blocks >>= 1) {
            uint64_t twist_bias_val = sh_twist_bias[lbs];
            for (int i = threadIdx.x; i < 1024; i += e18_threads) {
                int bias = ((i >> lbs) << (lbs + 1)) + (i % (1 << lbs));
                uint64_t tmp0 = mat32[bias];
                uint64_t tmp1 = mat32[bias + (1 << lbs)];
                uint64_t twist_val = cumul(twist_bias_val, twist_ptr[i >> lbs]);
                mat32[bias] = cuadd(tmp0, tmp1);
                mat32[bias + (1 << lbs)] = cumul(cusub(tmp0, tmp1), twist_val);
            }
            twist_ptr += blocks;
            lbs++;
            __syncthreads();
        }
        
        for (int i = 0; i < 2048; i += e18_threads) {
            block_dst[l + i + threadIdx.x] = mat32[i + threadIdx.x];
        }
        __syncthreads();
    }

    for (int ind = 0; ind < 2048; ind += 32) {
        for (int i = warp_id; i < 128; i += e18_threads / 32) {
            mat32[lane_id + i * 32] = block_dst[lane_id + ind + i * 2048];
        }
        __syncthreads();

        uint32_t lbs = 0;
        for (int blocks = 64; blocks >= 1; blocks >>= 1) {
            for (int i = warp_id; i < 64; i += e18_threads / 32) {
                uint64_t twist_bias_val = sh_twist_bias[lbs + 11];
                int bias = ((i >> lbs) << (lbs + 1)) + (i % (1 << lbs));
                uint64_t tmp0 = mat32[lane_id + bias * 32];
                uint64_t tmp1 = mat32[lane_id + bias * 32 + (32 << lbs)];
                uint64_t twist_val = cumul(twist_bias_val, twist11[i >> lbs]);
                mat32[lane_id + bias * 32] = cuadd(tmp0, tmp1);
                mat32[lane_id + bias * 32 + (32 << lbs)] = cumul(cusub(tmp0, tmp1), twist_val);
            }
            lbs++;
            __syncthreads();
        }

        for (int i = warp_id; i < 128; i += e18_threads / 32) {
            block_dst[lane_id + ind + i * 2048] = mat32[lane_id + i * 32];
        }
        __syncthreads();
    }
}

__global__ void kernel_fft_e18(uint64_t *dst, uint64_t *twist, uint64_t *twist_bias) {
    __shared__ uint64_t mat32[4096];
    __shared__ uint64_t sh_twist_bias[18];
    __shared__ uint64_t twist11[128];

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (threadIdx.x < 128) {
        twist11[threadIdx.x] = twist[threadIdx.x];
    }
    if (threadIdx.x < 18) {
        sh_twist_bias[threadIdx.x] = twist_bias[blockIdx.x * 18 + threadIdx.x];
    }

    uint64_t *block_dst = dst + blockIdx.x * 262144;

    for (int ind = 0; ind < 2048; ind += 32) {
        for (int i = warp_id; i < 128; i += e18_threads / 32) {
            mat32[lane_id + i * 32] = block_dst[lane_id + ind + i * 2048];
        }
        __syncthreads();

        uint32_t lbs = 6;
        for (int blocks = 1; blocks <= 64; blocks <<= 1) {
            for (int i = warp_id; i < 64; i += e18_threads / 32) {
                uint64_t twist_bias_val = sh_twist_bias[lbs + 11];
                int bias = ((i >> lbs) << (lbs + 1)) + (i % (1 << lbs));
                uint64_t tmp0 = mat32[lane_id + bias * 32];
                uint64_t tmp1 = mat32[lane_id + bias * 32 + (32 << lbs)];
                uint64_t twist_val = cumul(twist_bias_val, twist11[i >> lbs]);
                uint64_t tmp2 = cumul(tmp1, twist_val);
                mat32[lane_id + bias * 32] = cuadd(tmp0, tmp2);
                mat32[lane_id + bias * 32 + (32 << lbs)] = cusub(tmp0, tmp2);
            }
            lbs--;
            __syncthreads();
        }

        for (int i = warp_id; i < 128; i += e18_threads / 32) {
            block_dst[lane_id + ind + i * 2048] = mat32[lane_id + i * 32];
        }
        __syncthreads();
    }

    for (int l = 0; l < 262144; l += 2048) {
        for (int i = 0; i < 2048; i += e18_threads) {
            mat32[i + threadIdx.x] = block_dst[l + i + threadIdx.x];
            mat32[i + threadIdx.x + 2048] = twist[l + i + threadIdx.x];
        }
        __syncthreads();

        uint64_t *twist_ptr = mat32 + 2048 + 2048 - 2;
        uint32_t lbs = 10;
        for (int blocks = 1; blocks <= 1024; blocks <<= 1) {
            uint64_t twist_bias_val = sh_twist_bias[lbs];
            for (int i = threadIdx.x; i < 1024; i += e18_threads) {
                int bias = ((i >> lbs) << (lbs + 1)) + (i % (1 << lbs));
                uint64_t tmp0 = mat32[bias];
                uint64_t tmp1 = mat32[bias + (1 << lbs)];
                uint64_t twist_val = cumul(twist_bias_val, twist_ptr[i >> lbs]);
                uint64_t tmp2 = cumul(tmp1, twist_val);
                mat32[bias] = cuadd(tmp0, tmp2);
                mat32[bias + (1 << lbs)] = cusub(tmp0, tmp2);
            }
            twist_ptr -= 2 * blocks;
            lbs--;
            __syncthreads();
        }
        
        for (int i = 0; i < 2048; i += e18_threads) {
            block_dst[l + i + threadIdx.x] = mat32[i + threadIdx.x];
        }
        __syncthreads();
    }
}

constexpr long e32_blocks = 64;
constexpr long e32_threads = 256;

__global__ void kernel_invfft_e32(uint64_t *dst, int pages, uint64_t *twist) {
    extern __shared__ uint64_t mat32[];
    __shared__ uint64_t sh_twist[128];

    int ind = blockIdx.x * 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (threadIdx.x < pages) {
        sh_twist[threadIdx.x] = twist[threadIdx.x];
    }
    __syncthreads();

    while (ind < 262144) {
        for (int i = warp_id; i < pages; i += e32_threads / 32) {
            mat32[lane_id + i * 32] = dst[ind + lane_id + i * 262144];
        }
        __syncthreads();

        uint64_t *twist_ptr = sh_twist;
        uint32_t lbs = 0;
        for (int blocks = pages / 2; blocks >= 1; blocks >>= 1) {
            for (int i = warp_id; i < pages / 2; i += e32_threads / 32) {
                int bias = ((i >> lbs) << (lbs + 1)) + (i % (1 << lbs));
                uint64_t tmp0 = mat32[lane_id + bias * 32];
                uint64_t tmp1 = mat32[lane_id + bias * 32 + (32 << lbs)];
                uint64_t twist_val = twist_ptr[i >> lbs];
                mat32[lane_id + bias * 32] = cuadd(tmp0, tmp1);
                mat32[lane_id + bias * 32 + (32 << lbs)] = cumul(cusub(tmp0, tmp1), twist_val);
            }
            twist_ptr += blocks;
            lbs++;
            __syncthreads();
        }

        for (int i = warp_id; i < pages; i += e32_threads / 32) {
            dst[ind + lane_id + i * 262144] = mat32[lane_id + i * 32];
        }
        __syncthreads();
        ind += e32_blocks * 32;
    }
}

__global__ void kernel_fft_e32(uint64_t *dst, int pages, uint64_t *twist) {
    extern __shared__ uint64_t mat32[];
    __shared__ uint64_t sh_twist[128];

    int ind = blockIdx.x * 32;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    if (threadIdx.x < pages) {
        sh_twist[threadIdx.x] = twist[threadIdx.x];
    }
    __syncthreads();

    while (ind < 262144) {
        for (int i = warp_id; i < pages; i += e32_threads / 32) {
            mat32[lane_id + i * 32] = dst[ind + lane_id + i * 262144];
        }
        __syncthreads();

        uint64_t *twist_ptr = sh_twist + pages - 2;
        uint32_t lbs = 31 - __clz(pages / 2);
        for (int blocks = 1; blocks <= pages / 2; blocks <<= 1) {
            for (int i = warp_id; i < pages / 2; i += e32_threads / 32) {
                int bias = ((i >> lbs) << (lbs + 1)) + (i % (1 << lbs));
                uint64_t tmp0 = mat32[lane_id + bias * 32];
                uint64_t tmp1 = mat32[lane_id + bias * 32 + (32 << lbs)];
                uint64_t twist_val = twist_ptr[i >> lbs];
                uint64_t tmp2 = cumul(tmp1, twist_val);
                mat32[lane_id + bias * 32] = cuadd(tmp0, tmp2);
                mat32[lane_id + bias * 32 + (32 << lbs)] = cusub(tmp0, tmp2);
            }
            twist_ptr -= 2 * blocks;
            lbs--;
            __syncthreads();
        }

        for (int i = warp_id; i < pages; i += e32_threads / 32) {
            dst[ind + lane_id + i * 262144] = mat32[lane_id + i * 32];
        }
        __syncthreads();
        ind += e32_blocks * 32;
    }
}

void launch_kernel_odd(uint64_t *dst, uint64_t size, uint64_t *col0, uint64_t *multipliers, long p, cudaStream_t &stream) {
    if (p == 3) {
        kernel_odd<3><<<odd_blocks, odd_threads, 256 * p, stream>>>(
            dst, size, col0, multipliers
        );
    } else if (p == 5) {
        kernel_odd<5><<<odd_blocks, odd_threads, 256 * p, stream>>>(
            dst, size, col0, multipliers
        );
    } else if (p == 17) {
        kernel_odd<17><<<odd_blocks, odd_threads, 256 * p, stream>>>(
            dst, size, col0, multipliers
        );
    } else if (p == 257) {
        kernel_odd<257><<<odd_blocks, odd_threads, 256 * p, stream>>>(
            dst, size, col0, multipliers
        );
    } else {
        lg_err("launch_kernel_odd: p = %ld is not supported", p);
        abort();
    }
}

void fft64::gpu_invfft_odd(poly_t *p, long p0, long p1, long e2) {
    const long total_tasks = (1L << (e2 - 18));
    const uint64_t wp1 = pow(inv(g), (modulus - 1) / p1);
    const uint64_t wp0 = pow(inv(g), (modulus - 1) / p0);
    const uint64_t wpp = pow(inv(g), (modulus - 1) / (p0 * p1));

    for (long ct = 0; ct < ctp.ctp_threads; ct++) {
        ctp.worker[ct].thread.push([=]() {
            chunk_t cks[1285];
            uint64_t *h_col0 = (uint64_t *)malloc(2048 * 8);
            uint64_t *h_multipliers = (uint64_t *)malloc(2048 * 8);
            uint64_t *d_col0 = ctp.worker[ct].d_buf + page_u64 * 1285;
            uint64_t *d_multipliers = d_col0 + 512;
            for (long task = ct; task < total_tasks; task += ctp.ctp_threads) {
                for (long i = 0; i < p0 * p1; i++) {
                    cks[i] = *cm.fetch(p->page[task + i * total_tasks]);
                }
                for (long i = 0; i < p0 * p1 && task + ctp.ctp_threads < total_tasks; i++) {
                    cm.prefetch(p->page[task + i * total_tasks + ctp.ctp_threads]);
                }

                for (long i = 0; i < p0 * p1; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                        cks[i].data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream)); 
                }
                
                if (p1 > 1) {
                    for (long i = 0; i < p1; i++) h_multipliers[i] = fft64::pow(wp1, i);
                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers, p1 * 8, 
                        cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    for (long j = 0; j < p0; j++) {
                        for (long i = 0; i < p1; i++) h_col0[i + j * p1] = fft64::pow(wpp, i * j);
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_col0, h_col0 + j * p1, p1 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                        launch_kernel_odd(ctp.worker[ct].d_buf + j * p1 * page_u64, page_u64, 
                            d_col0, d_multipliers, p1, ctp.worker[ct].stream);
                    }
                }
                
                for (long i = 0; i < p0; i++) h_col0[p0 * p1 + i] = 1;
                for (long i = 0; i < p0; i++) h_multipliers[p1 + i] = fft64::pow(wp0, i);
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_col0, h_col0 + p0 * p1, p0 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers + p1, p0 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                launch_kernel_odd(ctp.worker[ct].d_buf, page_u64 * p1, 
                    d_col0, d_multipliers, p0, ctp.worker[ct].stream);
                for (long i = 0; i < p0 * p1; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(cks[i].data, ctp.worker[ct].d_buf + i * page_u64, 
                        page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                }

                CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));

                for (long i = 0; i < p0 * p1; i++) {
                    cm.release_sync(cks[i].id);
                }
            }
            free(h_col0);
            free(h_multipliers);
        });
    }
    ctp.wait();
}

void fft64::gpu_fft_odd(poly_t *p, long p0, long p1, long e2) {
    const long total_tasks = (1L << (e2 - 18));
    const uint64_t wp1 = pow(g, (modulus - 1) / p1);
    const uint64_t wp0 = pow(g, (modulus - 1) / p0);
    const uint64_t wpp = pow(g, (modulus - 1) / (p0 * p1));

    for (long ct = 0; ct < ctp.ctp_threads; ct++) {
        ctp.worker[ct].thread.push([=]() {
            chunk_t cks[1285];
            uint64_t *h_col0 = (uint64_t *)malloc(2048 * 8);
            uint64_t *h_multipliers = (uint64_t *)malloc(1048576 * 8);
            uint64_t *d_col0 = ctp.worker[ct].d_buf + page_u64 * 1285;
            uint64_t *d_multipliers = d_col0 + 512;
            for (long task = ct; task < total_tasks; task += ctp.ctp_threads) {
                for (long i = 0; i < p0 * p1; i++) {
                    cks[i] = *cm.fetch(p->page[task + i * total_tasks]);
                }
                for (long i = 0; i < p0 * p1 && task + ctp.ctp_threads < total_tasks; i++) {
                    cm.prefetch(p->page[task + i * total_tasks + ctp.ctp_threads]);
                }

                for (long i = 0; i < p0 * p1; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                        cks[i].data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream)); 
                }
                
                for (long i = 0; i < p0 + p1; i++) h_col0[i] = 1;
                for (long i = 0; i < p0; i++) h_multipliers[i] = fft64::pow(wp0, i);
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_col0, h_col0, (p0 + p1) * 8, 
                        cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers, p0 * 8, 
                        cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                launch_kernel_odd(ctp.worker[ct].d_buf, page_u64 * p1, d_col0, d_multipliers, p0, ctp.worker[ct].stream);

                if (p1 > 1) {
                    for (long j = 0; j < p0; j++) {
                        for (long i = 0; i < p1; i++) h_multipliers[(j + 1) * p0 + i] = mul(fft64::pow(wpp, j), fft64::pow(wp1, i));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers + (j + 1) * p0, p1 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                        launch_kernel_odd(ctp.worker[ct].d_buf + j * p1 * page_u64, page_u64, 
                            d_col0, d_multipliers, p1, ctp.worker[ct].stream);
                    }
                }

                for (long i = 0; i < p0 * p1; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(cks[i].data, ctp.worker[ct].d_buf + i * page_u64, 
                        page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                }

                CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));

                for (long i = 0; i < p0 * p1; i++) {
                    cm.release_sync(cks[i].id);
                }
            }
            free(h_col0);
            free(h_multipliers);
        });
    }
    ctp.wait();
}

void fft64::cpu_invfft_even(chunk_t *cks, long T, long p0, long p1, long e2, uint64_t *gpinv_rev) {
    const long Tpages = fft_pages((p0 * p1) << e2) / p0 / p1;
    const uint64_t gpinv = pow(inv(g), (modulus - 1) >> e2);
    const uint64_t gginv = pow(inv(g), (modulus - 1) / (p0 * p1 * (1UL << e2)));
    const uint64_t GSizeinv = inv((p0 * p1) << e2);

    #pragma omp parallel for schedule(static)
    for (long task = 0; task < Tpages; task++) {
        const long E0 = e2 < __builtin_ctzll(page_u64) ? e2 : __builtin_ctzll(page_u64);
        uint64_t *c0 = cks[task].data;
        for (long l = 0; l < page_u64 && l < (1L << e2); l++) c0[l] = mul(c0[l], GSizeinv);
        for (long E = 0; E < E0; E++) {
            const uint64_t gbias = mul(pow(gpinv, bit_rev(task * (page_u64 >> (E + 1)), e2 - 1)), pow(gginv, T << E));
            const long stride_u64 = 1UL << E;
            const long task_block = (page_u64 < (1L << e2) ? page_u64 : (1 << e2)) / stride_u64 / 2;
            for (long block_id = 0; block_id < task_block; block_id++) {
                uint64_t twist = mul(gbias, gpinv_rev[block_id]);
                for (long l = 0; l < stride_u64; l++) {
                    uint64_t tmp0 = c0[(block_id * 2 + 0) * stride_u64 + l];
                    uint64_t tmp1 = c0[(block_id * 2 + 1) * stride_u64 + l];
                    c0[(block_id * 2 + 0) * stride_u64 + l] = add(tmp0, tmp1);
                    c0[(block_id * 2 + 1) * stride_u64 + l] = mul(sub(tmp0, tmp1), twist);
                }
            }
        }
    }

    for (long E = __builtin_ctzll(page_u64); E < e2; E++) {
        const uint64_t gE = pow(gginv, T << E);
        const long stride_pages = 1UL << (E - __builtin_ctzll(page_u64));
        #pragma omp parallel for schedule(static)
        for (long task = 0; task < Tpages / 2; task++) {
            const long block_id = task / stride_pages;
            const long page_id = task % stride_pages;
            uint64_t *c0 = cks[(block_id * 2 + 0) * stride_pages + page_id].data;
            uint64_t *c1 = cks[(block_id * 2 + 1) * stride_pages + page_id].data;
            uint64_t twist = mul(gpinv_rev[block_id], gE);
            for (long l = 0; l < page_u64; l++) {
                uint64_t tmp0 = add(c0[l], c1[l]);
                uint64_t tmp1 = mul(sub(c0[l], c1[l]), twist);
                c0[l] = tmp0;
                c1[l] = tmp1;
            }
        }
    }
}

void fft64::cpu_fft_even(chunk_t *cks, long T, long p0, long p1, long e2, uint64_t *gp_rev) {
    const long Tpages = fft_pages((p0 * p1) << e2) / p0 / p1;
    const uint64_t gp = pow(g, (modulus - 1) >> e2);
    const uint64_t gg = pow(g, (modulus - 1) / (p0 * p1 * (1UL << e2)));

    for (long E = e2 - 1; E >= __builtin_ctzll(page_u64); E--) {
        const uint64_t gE = pow(gg, T << E);
        const long stride_pages = 1UL << (E - __builtin_ctzll(page_u64));
        #pragma omp parallel for schedule(static)
        for (long task = 0; task < Tpages / 2; task++) {
            const long block_id = task / stride_pages;
            const long page_id = task % stride_pages;
            uint64_t *c0 = cks[(block_id * 2 + 0) * stride_pages + page_id].data;
            uint64_t *c1 = cks[(block_id * 2 + 1) * stride_pages + page_id].data;
            uint64_t twist = mul(gp_rev[block_id], gE);
            for (long l = 0; l < page_u64; l++) {
                uint64_t tmp0 = add(c0[l], mul(c1[l], twist));
                uint64_t tmp1 = sub(c0[l], mul(c1[l], twist));
                c0[l] = tmp0;
                c1[l] = tmp1;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (long task = 0; task < Tpages; task++) {
        const long E0 = e2 < __builtin_ctzll(page_u64) ? e2 : __builtin_ctzll(page_u64);
        uint64_t *c0 = cks[task].data;
        for (long E = E0 - 1; E >= 0; E--) {
            const uint64_t gbias = mul(pow(gp, bit_rev(task * (page_u64 >> (E + 1)), e2 - 1)), pow(gg, T << E));
            const long stride_u64 = 1UL << E;
            const long task_block = (page_u64 < (1L << e2) ? page_u64 : (1 << e2)) / stride_u64 / 2;
            for (long block_id = 0; block_id < task_block; block_id++) {
                uint64_t twist = mul(gbias, gp_rev[block_id]);
                for (long l = 0; l < stride_u64; l++) {
                    uint64_t tmp0 = c0[(block_id * 2 + 0) * stride_u64 + l];
                    uint64_t tmp1 = c0[(block_id * 2 + 1) * stride_u64 + l];
                    c0[(block_id * 2 + 0) * stride_u64 + l] = add(tmp0, mul(tmp1, twist));
                    c0[(block_id * 2 + 1) * stride_u64 + l] = sub(tmp0, mul(tmp1, twist));
                }
            }
        }
    }
}

uint64_t *compute_chunk_twist(uint64_t *gpinv_rev) {
    uint64_t *ret = (uint64_t *)malloc(2 << 20);
    for (long l = 0; l < 262144; l += 2048) {
        uint64_t *dst = ret + l;
        for (long E = 1; E < 12; E++) {
            for (long i = 0; i < (2048 >> E); i++) {
                dst[i] = gpinv_rev[i + (l >> E)];
            }
            dst += 2048 >> E;
        }
    }
    return ret;
}

void fft64::gpu_invfft_even(chunk_t *cks, long T, long p0, long p1, long e2, uint64_t *gpinv_rev) {
    const long Tpages = 1 << (e2 - 18);
    const uint64_t gpinv = pow(inv(g), (modulus - 1) >> e2);
    const uint64_t gginv = pow(inv(g), (modulus - 1) / (p0 * p1 * (1UL << e2)));
    const uint64_t GSizeinv = inv((p0 * p1) << e2);
    uint64_t *h_twist = compute_chunk_twist(gpinv_rev);
    
    for (long ct = 0; ct < ctp.ctp_threads; ct++) {
        ctp.worker[ct].thread.push([=]() {
            uint64_t *d_twist = ctp.worker[ct].d_buf + page_u64 * 128;
            uint64_t *d_ck_twist = d_twist + page_u64 * 129;
            uint64_t twist_bias[2304];
            for (long task = ct; task < Tpages / 128; task += ctp.ctp_threads) {
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_ck_twist, h_twist, page_u64 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                for (long bias = task * 128; bias < task * 128 + 128; bias++) {
                    for (long E = 0; E < 18; E++) {
                        twist_bias[(bias - task * 128) * 18 + E] = 
                            mul(pow(gpinv, bit_rev(bias * (page_u64 >> (E + 1)), e2 - 1)), pow(gginv, T << E));
                    }
                }
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist_bias, 128 * 18 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                for (long i = 0; i < 128; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                        cks[task * 128 + i].data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                }
                kernel_invfft_e18<<<128, e18_threads, 0, ctp.worker[ct].stream>>>(
                    ctp.worker[ct].d_buf, d_ck_twist, d_twist, GSizeinv
                );
                
                uint64_t twist[128];
                int pos = 0;
                for (long E = 18; E < 25; E++) {
                    const uint64_t gE = pow(gginv, T << E);
                    for (int l = 0; l < (1 << (25 - E - 1)); l++) twist[pos++] = mul(gpinv_rev[(task << (25 - E - 1)) + l], gE);
                }
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                kernel_invfft_e32<<<e32_blocks, e32_threads, 256 * 128, ctp.worker[ct].stream>>>(
                    ctp.worker[ct].d_buf, 128, d_twist
                );
                for (long i = 0; i < 128; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(cks[task * 128 + i].data, ctp.worker[ct].d_buf + i * page_u64, 
                        page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                }
                CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
            }
        });
    }
    ctp.wait();

    free(h_twist);
    
    if (e2 == 25) return;
    for (long ct = 0; ct < ctp.ctp_threads; ct++) {
        const long pages = 1 << (e2 - 25);
        ctp.worker[ct].thread.push([=]() {
            uint64_t twist[128];
            int pos = 0;
            for (long E = 25; E < e2; E++) {
                const uint64_t gE = pow(gginv, T << E);
                for (int l = 0; l < (1 << (e2 - E - 1)); l++) twist[pos++] = mul(gpinv_rev[l], gE);
            }
            uint64_t *d_twist = ctp.worker[ct].d_buf + page_u64 * 128;
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
            for (long task = ct; task < 128; task += ctp.ctp_threads) {
                for (long i = 0; i < pages; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                        cks[task + i * 128].data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                }
                kernel_invfft_e32<<<e32_blocks, e32_threads, 256 * pages, ctp.worker[ct].stream>>>(
                    ctp.worker[ct].d_buf, pages, d_twist
                );
                for (long i = 0; i < pages; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(cks[task + i * 128].data, ctp.worker[ct].d_buf + i * page_u64, 
                        page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                }
                CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
            }
        });
    }
    ctp.wait();
}

void fft64::gpu_fft_even(chunk_t *cks, long T, long p0, long p1, long e2, uint64_t *gp_rev) {
    const long Tpages = 1 << (e2 - 18);
    const uint64_t gp = pow(g, (modulus - 1) >> e2);
    const uint64_t gg = pow(g, (modulus - 1) / (p0 * p1 * (1UL << e2)));
    uint64_t *h_twist = compute_chunk_twist(gp_rev);

    for (long ct = 0; ct < ctp.ctp_threads && e2 > 25; ct++) {
        const long pages = 1 << (e2 - 25);
        ctp.worker[ct].thread.push([=]() {
            uint64_t twist[128];
            int pos = 0;
            for (long E = 25; E < e2; E++) {
                const uint64_t gE = pow(gg, T << E);
                for (int l = 0; l < (1 << (e2 - E - 1)); l++) twist[pos++] = mul(gp_rev[l], gE);
            }
            uint64_t *d_twist = ctp.worker[ct].d_buf + page_u64 * 128;
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
            for (long task = ct; task < 128; task += ctp.ctp_threads) {
                for (long i = 0; i < pages; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                        cks[task + i * 128].data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                }
                kernel_fft_e32<<<e32_blocks, e32_threads, 256 * pages, ctp.worker[ct].stream>>>(
                    ctp.worker[ct].d_buf, pages, d_twist
                );
                for (long i = 0; i < pages; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(cks[task + i * 128].data, ctp.worker[ct].d_buf + i * page_u64, 
                        page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                }
                CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
            }
        });
    }
    ctp.wait();
    
    for (long ct = 0; ct < ctp.ctp_threads; ct++) {
        ctp.worker[ct].thread.push([=]() {
            uint64_t *d_twist = ctp.worker[ct].d_buf + page_u64 * 128;
            uint64_t *d_ck_twist = d_twist + page_u64 * 129;
            uint64_t twist_bias[2304];
            for (long task = ct; task < Tpages / 128; task += ctp.ctp_threads) {
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_ck_twist, h_twist, page_u64 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                for (long i = 0; i < 128; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                        cks[task * 128 + i].data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                }

                uint64_t twist[128];
                int pos = 0;
                for (long E = 18; E < 25; E++) {
                    const uint64_t gE = pow(gg, T << E);
                    for (int l = 0; l < (1 << (25 - E - 1)); l++) twist[pos++] = mul(gp_rev[(task << (25 - E - 1)) + l], gE);
                }
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                kernel_fft_e32<<<e32_blocks, e32_threads, 256 * 128, ctp.worker[ct].stream>>>(
                    ctp.worker[ct].d_buf, 128, d_twist
                );
                
                for (long bias = task * 128; bias < task * 128 + 128; bias++) {
                    for (long E = 0; E < 18; E++) {
                        twist_bias[(bias - task * 128) * 18 + E] = 
                            mul(pow(gp, bit_rev(bias * (page_u64 >> (E + 1)), e2 - 1)), pow(gg, T << E));
                    }
                }
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist_bias, 128 * 18 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                kernel_fft_e18<<<128, e18_threads, 0, ctp.worker[ct].stream>>>(
                    ctp.worker[ct].d_buf, d_ck_twist, d_twist
                );
                
                for (long i = 0; i < 128; i++) {
                    CHECK_CUDA_ERR(cudaMemcpyAsync(cks[task * 128 + i].data, ctp.worker[ct].d_buf + i * page_u64, 
                        page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                }
                CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
            }
        });
    }
    ctp.wait();

    free(h_twist);
}

uint64_t *gen_gp_rev(uint64_t gp, long nbits);

__global__ void kernel_tg2(uint64_t *dste, uint64_t *dsto, uint64_t *gp_revf, uint64_t *gt) {
    uint64_t block_gt = gt[blockIdx.x];
    uint64_t *block_dste = dste + blockIdx.x * 262144;
    uint64_t *block_dsto = dsto + blockIdx.x * 262144;

    for (int i = threadIdx.x; i < 262144; i += blockDim.x) {
        uint64_t tmp0 = block_dste[i];
        uint64_t tmp1 = block_dsto[i];
        uint64_t tmp2 = gp_revf[i];
        uint64_t tmp3 = cumul(tmp0, tmp0);
        uint64_t tmp4 = cumul(tmp1, tmp1);
        uint64_t tmp5 = cumul(block_gt, tmp2);
        block_dste[i] = cusub(tmp3, cumul(tmp4, tmp5));
    }
}

__global__ void kernel_sort2(uint64_t *dste, uint64_t *dsto, uint64_t *src) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < 262144; i += stride) {
        uint64_t tmp0 = src[2 * i];
        uint64_t tmp1 = src[2 * i + 1];
        dste[i] = tmp0;
        dsto[i] = tmp1;
    }
}

void fft64::gpu_tg2(poly_t *p, long p0, long p1, long e2) {
    const long fft_pages = fft64::fft_pages((p0 * p1) << e2);

    poly_t fo(fft_pages), fe(fft_pages);
    fo.alloc_page(fft_pages);
    fo.pages = 0;

    /// fft odd
    if (1) {
        const long total_tasks = (1L << (e2 - 18));
        const uint64_t wp1 = pow(g, (modulus - 1) / p1);
        const uint64_t wp0 = pow(g, (modulus - 1) / p0);
        const uint64_t wpp = pow(g, (modulus - 1) / (p0 * p1));

        for (long ct = 0; ct < ctp.ctp_threads; ct++) {
            ctp.worker[ct].thread.push([=, &fo, &fe]() {
                chunk_t *cko[1285], *cke[1285], *ckp[1285 * 2];
                uint64_t *h_col0 = (uint64_t *)malloc(2048 * 8);
                uint64_t *h_multipliers = (uint64_t *)malloc(1048576 * 8);
                uint64_t *d_col0 = ctp.worker[ct].d_buf + page_u64 * 1285;
                uint64_t *d_multipliers = d_col0 + 512;
                uint64_t *d_tmp = ctp.worker[ct].d_buf + page_u64 * 1286;

                int32_t num_holding_chunks = 0;
                int32_t *holding_chunks = (int32_t *)malloc(65536);
                for (long task = ct; task < total_tasks; task += ctp.ctp_threads) {
                    for (long i = 0; i < p0 * p1; i++) {
                        long pos = task + i * total_tasks;
                        ckp[2 * i + 0] = pos >= fft_pages / 2 ? NULL : cm.fetch(p->page[2 * pos + 0]);
                        ckp[2 * i + 1] = pos >= fft_pages / 2 ? NULL : cm.fetch(p->page[2 * pos + 1]);
                        cko[i] = cm.fetch(fo.page[pos]);
                    }

                    for (long i = 0; i < p0 * p1; i++) {
                        long pos = task + i * total_tasks + ctp.ctp_threads;
                        if (pos < fft_pages / 2) cm.prefetch(p->page[2 * pos + 0]);
                        if (pos < fft_pages / 2) cm.prefetch(p->page[2 * pos + 1]);
                    }

                    for (long i = 0; i < p0 * p1; i++) {
                        if (p0 * p1 == 1) cke[i] = cm.fetch(cm.create_chunk());
                        if (!ckp[2 * i + 0] && !ckp[2 * i + 1]) continue;
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_tmp, ckp[2 * i + 0]->data, page_u64 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_tmp + page_u64, ckp[2 * i + 1]->data, page_u64 * 8,
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                        kernel_sort2<<<64, 256, 0, ctp.worker[ct].stream>>>(
                            ctp.worker[ct].d_buf + i * page_u64, d_tmp + 2 * page_u64, d_tmp
                        );
                        CHECK_CUDA_ERR(cudaMemcpyAsync(cko[i]->data, d_tmp + 2 * page_u64, page_u64 * 8, 
                            cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                        if (p0 * p1 == 1) {
                            CHECK_CUDA_ERR(cudaMemcpyAsync(cke[i]->data, ctp.worker[ct].d_buf + i * page_u64, 
                                page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                        }
                    }
                    CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));

                    if (p0 * p1 == 1) {
                        if (ckp[0]) cm.release_del(ckp[0]->id);
                        if (ckp[1]) cm.release_del(ckp[1]->id);
                        fe.page[task] = cke[0]->id;
                        cm.release_sync(cke[0]->id);
                        cm.release_sync(cko[0]->id);
                        continue;
                    } else {
                        for (long i = 0; i < p0 * p1; i++) {
                            if (ckp[2 * i + 0]) holding_chunks[num_holding_chunks++] = ckp[2 * i + 0]->id;
                            if (ckp[2 * i + 1]) holding_chunks[num_holding_chunks++] = ckp[2 * i + 1]->id;
                            if (ckp[2 * i + 0]) cm.release(ckp[2 * i + 0]->id);
                            if (ckp[2 * i + 1]) cm.release(ckp[2 * i + 1]->id);
                        }
                    }

                    for (long i = 0; i < p0 + p1; i++) h_col0[i] = 1;
                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_col0, h_col0, (p0 + p1) * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));

                    /// cke
                    for (long i = 0; i < p0 * p1; i++) {
                        if (task + i * total_tasks >= fft_pages / 2) {
                            CHECK_CUDA_ERR(cudaMemsetAsync(ctp.worker[ct].d_buf + i * page_u64, 0, 
                                page_u64 * 8, ctp.worker[ct].stream));
                        }
                    }
                    
                    for (long i = 0; i < p0; i++) h_multipliers[i] = fft64::pow(wp0, i);
                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers, p0 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    launch_kernel_odd(ctp.worker[ct].d_buf, page_u64 * p1, d_col0, d_multipliers, p0, ctp.worker[ct].stream);

                    if (p1 > 1) {
                        for (long j = 0; j < p0; j++) {
                            for (long i = 0; i < p1; i++) h_multipliers[(j + 1) * p0 + i] = mul(fft64::pow(wpp, j), fft64::pow(wp1, i));
                            CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers + (j + 1) * p0, p1 * 8, 
                                cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                            launch_kernel_odd(ctp.worker[ct].d_buf + j * p1 * page_u64, page_u64, 
                                d_col0, d_multipliers, p1, ctp.worker[ct].stream);
                        }
                    }

                    for (long i = 0; i < p0 * p1; i++) {
                        cke[i] = cm.fetch(holding_chunks[--num_holding_chunks]);
                        CHECK_CUDA_ERR(cudaMemcpyAsync(cke[i]->data, ctp.worker[ct].d_buf + i * page_u64, 
                            page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                    }

                    /// cko
                    for (long i = 0; i < p0 * p1; i++) {
                        if (task + i * total_tasks >= fft_pages / 2) {
                            CHECK_CUDA_ERR(cudaMemsetAsync(ctp.worker[ct].d_buf + i * page_u64, 0, 
                                page_u64 * 8, ctp.worker[ct].stream));
                        } else {
                            CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                                cko[i]->data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream)); 
                        }
                    }
                    
                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers, p0 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    launch_kernel_odd(ctp.worker[ct].d_buf, page_u64 * p1, d_col0, d_multipliers, p0, ctp.worker[ct].stream);

                    for (long j = 0; j < p0 && p1 > 1; j++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_multipliers, h_multipliers + (j + 1) * p0, p1 * 8, 
                            cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                        launch_kernel_odd(ctp.worker[ct].d_buf + j * p1 * page_u64, page_u64, 
                            d_col0, d_multipliers, p1, ctp.worker[ct].stream);
                    }

                    for (long i = 0; i < p0 * p1; i++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(cko[i]->data, ctp.worker[ct].d_buf + i * page_u64, 
                            page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                    }

                    CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));

                    for (long i = 0; i < p0 * p1; i++) {
                        fe.page[task + ((i % p1) * p0 + (i / p1)) * total_tasks] = cke[i]->id;
                        fo.page[task + ((i % p1) * p0 + (i / p1)) * total_tasks] = cko[i]->id;
                        cm.release_sync(cke[i]->id);
                        cm.release_sync(cko[i]->id);
                    }
                }
                free(h_col0);
                free(h_multipliers);
                free(holding_chunks);
                if (num_holding_chunks) lg_err("ct = %ld, num_holding_chunks = %d when exiting", ct, num_holding_chunks);
            });
        }
        ctp.wait();
    }

    /// fft even && tg2 && invfft even
    const uint64_t gp       = pow(g, (modulus - 1) >> e2);
    const uint64_t gg       = pow(g, (modulus - 1) / (p0 * p1 * (1UL << e2)));
    const uint64_t gpinv    = pow(inv(g), (modulus - 1) >> e2);
    const uint64_t gginv    = pow(inv(g), (modulus - 1) / (p0 * p1 * (1UL << e2)));
    const uint64_t GSizeinv = inv((p0 * p1) << e2);
    uint64_t *gp_rev        = gen_gp_rev(gp, e2 - 1);
    uint64_t *gpinv_rev     = gen_gp_rev(gpinv, e2 - 1);
    uint64_t *h_twist       = compute_chunk_twist(gp_rev);
    uint64_t *h_twist_inv   = compute_chunk_twist(gpinv_rev);
    uint64_t *gp_revf       = gen_gp_rev(gp, e2);
    const long Tpages = fft_pages / (p0 * p1);
    
    chunk_t *cke = (chunk_t *) malloc(sizeof(chunk_t) * Tpages);
    chunk_t *cko = (chunk_t *) malloc(sizeof(chunk_t) * Tpages);

    for (long T = 0; T < p0 * p1; T++) {
        for (long i = 0; i < Tpages; i++) {
            cke[i] = *cm.fetch(fe.page[i + T * Tpages]);
            cko[i] = *cm.fetch(fo.page[i + T * Tpages]);
        }
        for (long i = 0; i < Tpages && T != p0 * p1 - 1; i++) {
            cm.prefetch(fe.page[i + (T + 1) * Tpages]);
            cm.prefetch(fo.page[i + (T + 1) * Tpages]);
        }

        for (long ct = 0; ct < ctp.ctp_threads; ct++) {
            ctp.worker[ct].thread.push([=]() {               
                uint64_t *d_bufe = ctp.worker[ct].d_buf + page_u64 * 0;
                uint64_t *d_bufo = ctp.worker[ct].d_buf + page_u64 * 128;
                uint64_t *d_twist = ctp.worker[ct].d_buf + page_u64 * 256;
                const long pages = 1 << (e2 - 25);
                
                uint64_t twist[128];
                int pos = 0;
                for (long E = 25; E < e2; E++) {
                    const uint64_t gE = pow(gg, T << E);
                    for (int l = 0; l < (1 << (e2 - E - 1)); l++) twist[pos++] = mul(gp_rev[l], gE);
                }
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));

                for (long task = ct; task < 128; task += ctp.ctp_threads) {
                    for (long i = 0; i < pages; i++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_bufe + i * page_u64, cke[task + i * 128].data, 
                            page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_bufo + i * page_u64, cko[task + i * 128].data, 
                            page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    }
                    kernel_fft_e32<<<e32_blocks, e32_threads, 256 * pages, ctp.worker[ct].stream>>>(d_bufe, pages, d_twist);
                    kernel_fft_e32<<<e32_blocks, e32_threads, 256 * pages, ctp.worker[ct].stream>>>(d_bufo, pages, d_twist);
                    for (long i = 0; i < pages; i++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(cke[task + i * 128].data, d_bufe + i * page_u64, 
                            page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(cko[task + i * 128].data, d_bufo + i * page_u64,
                            page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                    }
                    CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
                }
            });
        }
        ctp.wait();
        
        for (long ct = 0; ct < ctp.ctp_threads; ct++) {
            ctp.worker[ct].thread.push([=]() {
                uint64_t *d_bufe = ctp.worker[ct].d_buf + page_u64 * 0;
                uint64_t *d_bufo = ctp.worker[ct].d_buf + page_u64 * 128;
                uint64_t *d_twist = ctp.worker[ct].d_buf + page_u64 * 256;
                uint64_t *d_ck_twist = ctp.worker[ct].d_buf + page_u64 * 257;
                uint64_t *d_gp_revf = ctp.worker[ct].d_buf + page_u64 * 258;
                uint64_t *d_gt = ctp.worker[ct].d_buf + page_u64 * 259;
                uint64_t *d_twist_inv = ctp.worker[ct].d_buf + page_u64 * 260;
                uint64_t *d_ck_twist_inv = ctp.worker[ct].d_buf + page_u64 * 261;
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_gp_revf, gp_revf, page_u64 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_ck_twist, h_twist, page_u64 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_ck_twist_inv, h_twist_inv, page_u64 * 8, 
                    cudaMemcpyHostToDevice, ctp.worker[ct].stream));

                uint64_t twist[128], twist_inv[128], twist_bias[2304], twist_bias_inv[2304], gt[128];
                for (long task = ct; task < Tpages / 128; task += ctp.ctp_threads) {
                    for (long i = 0; i < 128; i++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_bufe + i * page_u64, cke[task * 128 + i].data, 
                            page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                        CHECK_CUDA_ERR(cudaMemcpyAsync(d_bufo + i * page_u64, cko[task * 128 + i].data, 
                            page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    }
                    
                    int pos = 0;
                    for (long E = 18; E < 25; E++) {
                        const uint64_t gE = pow(gg, T << E);
                        for (int l = 0; l < (1 << (25 - E - 1)); l++) twist[pos++] = mul(gp_rev[(task << (25 - E - 1)) + l], gE);
                    }

                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    
                    kernel_fft_e32<<<e32_blocks, e32_threads, 256 * 128, ctp.worker[ct].stream>>>(d_bufe, 128, d_twist);
                    kernel_fft_e32<<<e32_blocks, e32_threads, 256 * 128, ctp.worker[ct].stream>>>(d_bufo, 128, d_twist);
                    
                    for (long bias = task * 128; bias < task * 128 + 128; bias++) {
                        for (long E = 0; E < 18; E++) {
                            twist_bias[(bias - task * 128) * 18 + E] = 
                                mul(pow(gp, bit_rev(bias * (page_u64 >> (E + 1)), e2 - 1)), pow(gg, T << E));
                        }
                    }

                    for (long bias = task * 128; bias < task * 128 + 128; bias++) {
                        for (long E = 0; E < 18; E++) {
                            twist_bias_inv[(bias - task * 128) * 18 + E] = 
                                mul(pow(gpinv, bit_rev(bias * (page_u64 >> (E + 1)), e2 - 1)), pow(gginv, T << E));
                        }
                    }

                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist_bias, 128 * 18 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist_inv, twist_bias_inv, 128 * 18 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    
                    kernel_fft_e18<<<128, e18_threads, 0, ctp.worker[ct].stream>>>(d_bufe, d_ck_twist, d_twist);
                    kernel_fft_e18<<<128, e18_threads, 0, ctp.worker[ct].stream>>>(d_bufo, d_ck_twist, d_twist);
                    for (long i = 0; i < 128; i++) {
                        gt[i] = pow(gg, T + p0 * p1 * bit_rev((task * 128 + i) * page_u64, e2));
                    }
                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_gt, gt, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    kernel_tg2<<<128, 256, 0, ctp.worker[ct].stream>>>(d_bufe, d_bufo, d_gp_revf, d_gt);
                    
                    kernel_invfft_e18<<<128, e18_threads, 0, ctp.worker[ct].stream>>>(d_bufe, d_ck_twist_inv, d_twist_inv, GSizeinv);
                    
                    pos = 0;
                    for (long E = 18; E < 25; E++) {
                        const uint64_t gE = pow(gginv, T << E);
                        for (int l = 0; l < (1 << (25 - E - 1)); l++) twist_inv[pos++] = mul(gpinv_rev[(task << (25 - E - 1)) + l], gE);
                    }
                    CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist_inv, twist_inv, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    kernel_invfft_e32<<<e32_blocks, e32_threads, 256 * 128, ctp.worker[ct].stream>>>(
                        d_bufe, 128, d_twist_inv
                    );
                    for (long i = 0; i < 128; i++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(cke[task * 128 + i].data, d_bufe + i * page_u64, 
                            page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                    }
                    CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
                }
            });
        }
        ctp.wait();

        for (long ct = 0; ct < ctp.ctp_threads; ct++) {
            const long pages = 1 << (e2 - 25);
            ctp.worker[ct].thread.push([=]() {
                uint64_t twist[128];
                int pos = 0;
                for (long E = 25; E < e2; E++) {
                    const uint64_t gE = pow(gginv, T << E);
                    for (int l = 0; l < (1 << (e2 - E - 1)); l++) twist[pos++] = mul(gpinv_rev[l], gE);
                }
                uint64_t *d_twist = ctp.worker[ct].d_buf + page_u64 * 128;
                CHECK_CUDA_ERR(cudaMemcpyAsync(d_twist, twist, 128 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                for (long task = ct; task < 128; task += ctp.ctp_threads) {
                    for (long i = 0; i < pages; i++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(ctp.worker[ct].d_buf + i * page_u64, 
                            cke[task + i * 128].data, page_u64 * 8, cudaMemcpyHostToDevice, ctp.worker[ct].stream));
                    }
                    kernel_invfft_e32<<<e32_blocks, e32_threads, 256 * pages, ctp.worker[ct].stream>>>(
                        ctp.worker[ct].d_buf, pages, d_twist
                    );
                    for (long i = 0; i < pages; i++) {
                        CHECK_CUDA_ERR(cudaMemcpyAsync(cke[task + i * 128].data, ctp.worker[ct].d_buf + i * page_u64, 
                            page_u64 * 8, cudaMemcpyDeviceToHost, ctp.worker[ct].stream));
                    }
                    CHECK_CUDA_ERR(cudaStreamSynchronize(ctp.worker[ct].stream));
                }
            });
        }
        ctp.wait();

        for (long i = 0; i < Tpages; i++) {
            cm.release_sync(cke[i].id);
            cm.release_del(cko[i].id);
        }
    }

    free(h_twist);
    free(h_twist_inv);
    free(gp_revf);
    free(gp_rev);
    free(gpinv_rev);
    free(cke);
    free(cko);

    if (p0 * p1 != 1) {
        const long total_tasks = fft_pages / p1 / p0;            
        for (long i = 0; i < p1 * p0; i++) {
            for (long k = 0; k < total_tasks; k++) {
                p->page[k + total_tasks * i] = fe.page[k + total_tasks * ((i % p1) * p0 + (i / p1))];
            }
        }

        gpu_invfft_odd(p, p0, p1, e2);
    } else {
        for (long i = 0; i < fft_pages; i++) {
            p->page[i] = fe.page[i];
        }
    }
}

template void fft64::gpu_eval<Poseidon2_perm64_24b<0>>(poly_t *, long, uint64_t *, uint64_t *);
template void fft64::gpu_eval<Poseidon2_perm64_28b<1>>(poly_t *, long, uint64_t *, uint64_t *);
template void fft64::gpu_eval<Poseidon2_perm64_32b<0>>(poly_t *, long, uint64_t *, uint64_t *);
