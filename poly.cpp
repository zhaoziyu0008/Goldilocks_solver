#include "poly.h"
#include "fft.h"

#include <omp.h>
#include <math.h>

poly_t::poly_t(long max_pages) : _max_pages(max_pages) {
    if (max_pages >= 0) page = (int *) malloc(sizeof(int) * max_pages);
}

poly_t::~poly_t() {
    free_page();
    
    free(page);
    page = NULL;
}

int poly_t::copy_from(const poly_t& other) {
    _max_pages = other._max_pages;
    pages      = other.pages;
    page       = (int *) malloc(sizeof(int) * _max_pages);
    
    #pragma omp parallel for
    for (long i = 0; i < pages; i++) {
        page[i] = cm.create_chunk();
        chunk_t *ck = cm.fetch(page[i]);
        chunk_t *other_ck = cm.fetch(other.page[i]);
        memcpy(ck->data, other_ck->data, resource_holder_t::page_size);
        cm.release_sync(page[i]);
        cm.release(other.page[i]);
    }

    return 0;
}

int poly_t::alloc_page(long target_pages) {
    if (target_pages == -1) target_pages = _max_pages;
    if (target_pages < pages) return 0;
    if (target_pages > _max_pages) {
        lg_err("poly_t::alloc_page: target pages %ld exceeds max pages %ld", target_pages, _max_pages);
        return -1;
    }
    for (long i = pages; i < target_pages; i++) {
        page[i] = cm.create_chunk();
    }
    pages = target_pages;
    return 0;
}

int poly_t::free_page(long target_pages) {
    if (target_pages == -1) target_pages = 0;
    if (target_pages >= pages) return 0;
    for (long i = pages - 1; i >= target_pages; i--) {
        cm.fetch(page[i]);
        cm.release_del(page[i]);
    }
    pages = target_pages;
    return 0;
}

#include <dirent.h>
#include <sys/stat.h>

int poly_t::load(const char *dir, int rm_src) {
    /// dir check
    for (long i = 0; i < 4; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/%ld/", dir, i);
        struct stat st;
        if (stat(path, &st) == -1 || !S_ISDIR(st.st_mode)) {
            lg_err("poly_t::load: directory %s does not exist", path);
            return -1;
        }
    }

    int page_count = -1;

    /// check file names
    {
        char cmd[256];
        snprintf(cmd, sizeof(cmd), "find %s/0/ %s/1/ %s/2 %s/3 -type f -name \".ck*\" | wc -l", dir, dir, dir, dir);
        FILE *fp = popen(cmd, "r");
        if (fp == NULL) {
            lg_err("poly_t::load: failed to run command %s", cmd);
            return -1;
        }
        if (fscanf(fp, "%d", &page_count) != 1) {
            lg_err("poly_t::load: failed to read page count from command %s", cmd);
        }
        pclose(fp);

        for (long i = 0; i < 4; i++) {
            char dir_path[256];
            snprintf(dir_path, sizeof(dir_path), "%s/%ld/", dir, i);

            DIR *dp = opendir(dir_path);
            if (dp == NULL) {
                lg_err("poly_t::load: failed to open directory %s", dir_path);
                return -1;
            }
            
            struct dirent *entry;
            while ((entry = readdir(dp)) != NULL) {
                if (strncmp(entry->d_name, ".ck", 3) == 0) {
                    unsigned int id;
                    int r = sscanf(entry->d_name, ".ck%07x", &id);
                    if (r != 1 || id >= page_count || hash_to_dir(id) != i) {
                        lg_err("poly_t::load: invalid file name %s", entry->d_name);
                        closedir(dp);
                        return -1;
                    }
                }
            }

            closedir(dp);
        }
    }
    
    this->alloc_page(page_count);
    this->free_page(page_count);

    int ret = 0;

    #pragma omp parallel for
    for (long i = 0; i < page_count; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/%d/.ck%07lx", dir, hash_to_dir(i), i);

        int fd = open(filename, O_RDONLY);
        if (fd == -1) {
            lg_err("poly_t::load: open %s failed", filename);
            ret = -1;
            continue;
        }

        chunk_t *ck = cm.fetch(page[i]);

        long nbytes = read(fd, ck->data, resource_holder_t::page_size);
        close(fd);

        cm.release_sync(page[i]);
        
        if (nbytes != resource_holder_t::page_size) {
            lg_err("poly_t::load: read %s returned %ld(expected %ld)", 
                     filename, nbytes, resource_holder_t::page_size);
            ret = -1;
        }

        if (rm_src) remove(filename);
    }

    return ret;
}

int poly_t::sync(const char *dir, int rm_src) {
    if (mkdir(dir, 0755) == -1 && errno != EEXIST) {
        lg_err("poly_t::sync: failed to create parent directory %s", dir);
        return -1;
    }

    for (long i = 0; i < 4; i++) {
        char path[256];
        snprintf(path, sizeof(path), "%s/%ld/", dir, i);
        if (mkdir(path, 0755) == -1 && errno != EEXIST) {
            lg_err("poly_t::sync: failed to create directory %s", path);
            return -1;
        }

        char cmd[512];
        snprintf(cmd, sizeof(cmd), "find %s -type f -name \".ck*\" -delete", path);
        int _ = system(cmd);
    }

    int ret = 0;

    #pragma omp parallel for
    for (long i = 0; i < pages; i++) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/%d/.ck%07lx", dir, hash_to_dir(i), i);

        int fd = open(filename, O_WRONLY | O_CREAT, 0644);
        if (fd == -1) {
            lg_err("poly_t::sync: open %s failed", filename);
            ret = -1;
            continue;
        }

        chunk_t *ck = cm.fetch(page[i]);
        long nbytes = write(fd, ck->data, resource_holder_t::page_size);
        close(fd);

        if (rm_src) cm.release_del(page[i]);
        else cm.release(page[i]);

        if (nbytes != resource_holder_t::page_size) {
            lg_err("poly_t::sync: write %s returned %ld(expected %ld)", 
                     filename, nbytes, resource_holder_t::page_size);
            ret = -1;
        }
    }

    if (rm_src) this->pages = 0;

    return ret;
}

template<class FFT_traits>
uint64_t *roots(poly_t *p, long target_size, long period, uint64_t ratio) {
    period_red<FFT_traits>(p, target_size, period, ratio);

    long num_ret = 0;
    uint64_t *ret = (uint64_t *) malloc(sizeof(uint64_t) * (FFT_traits::roots_limit + 1));
    for (long i = 0; i < FFT_traits::roots_limit + 1; i++) ret[i] = -1ULL;
    
    if (period == 0x100000000L) {
        poly_t f[31];
        for (long i = 0; i < 31; i++) {
            f[i].copy_from(*p);
            if (i < 5) tg2_gpu<FFT_traits>(p, 0x100000000L >> i);
            else tg2<FFT_traits>(p, 0x100000000L >> i);
            period_red<FFT_traits>(p, 0x100000000L >> i, 0x100000000L >> (i + 1), ratio);
        }

        num_ret++;
        ret[0] = ratio;
        enum_roots<FFT_traits>(num_ret, ret, p, 2);
        for (long i = 0; i < 31; i++) {
            enum_roots<FFT_traits>(num_ret, ret, &f[30 - i], 2);
        }
    } else if (period == 0xffffffffL && ratio == 1) {
        poly_t f0, f1, f2, f3;
        f0.copy_from(*p);

        tg3<FFT_traits>(p, 0xffffffff);
        period_red<FFT_traits>(p, 0xffffffff, 0xffffffff / 3, 1);
        f1.copy_from(*p);

        tg5<FFT_traits>(p, 0xffffffff / 3);
        period_red<FFT_traits>(p, 0xffffffff / 3, 0xffffffff / 15, 1);
        f2.copy_from(*p);

        tg17<FFT_traits>(p, 0xffffffff / 15);
        period_red<FFT_traits>(p, 0xffffffff / 15, 0xffffffff / 255, 1);
        f3.copy_from(*p);

        tg257<FFT_traits>(p, 0xffffffff / 255);
        period_red<FFT_traits>(p, 0xffffffff / 255, 0xffffffff / 65535, 1);
        
        num_ret++;
        ret[0] = 1ULL;
        enum_roots<FFT_traits>(num_ret, ret, p, 65537);
        enum_roots<FFT_traits>(num_ret, ret, &f3, 257);
        enum_roots<FFT_traits>(num_ret, ret, &f2, 17);
        enum_roots<FFT_traits>(num_ret, ret, &f1, 5);
        enum_roots<FFT_traits>(num_ret, ret, &f0, 3);
    } else {
        lg_err("roots: not implemented for period %ld and ratio %ld", period, ratio);
    }

    return ret;
}

template <class FFT_traits>
void enum_roots(long &num_ret, uint64_t *ret, poly_t *p, long period) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);

    long num_roots = 0;
    uint64_t *roots_list = (uint64_t *) malloc(sizeof(uint64_t) * num_ret * period);

    for (long i = 0; i < num_ret; i++) {
        uint64_t root = FFT_traits::root(ret[i], period);
        if (root == -1ULL) continue;
        for (long j = 0; j < period; j++) {
            roots_list[num_roots++] = root;
            root = FFT_traits::mul(root, FFT_traits::rou(period));
        }
    }

    num_ret = 0;
    pthread_spinlock_t lock;
    pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED);

    if (p->pages < 4) {
        chunk_t ck_list[4];
        for (long i = 0; i < p->pages; i++) {
            ck_list[i] = *cm.fetch(p->page[i]);
        }

        long total_tasks = (num_roots - 1) / 16L + 1;
        #pragma omp parallel for schedule(static)
        for (long task = 0; task < total_tasks; task++) {
            const long task_size = task == total_tasks - 1 ? num_roots - task * 16 : 16;
            uint64_t res[16] = {};
            uint64_t rts[16] = {};
            for (int i = 0; i < task_size; i++) {
                rts[i] = roots_list[task * 16 + i];
            }

            for (long ck_ptr = p->pages - 1; ck_ptr >= 0; ck_ptr--) {
                uint64_t *data = ck_list[ck_ptr].data;
                for (long j = page_u64 - 1; j >= 0; j--) {
                    uint64_t val = data[j];
                    for (long k = 0; k < task_size; k++) {
                        res[k] = FFT_traits::add(val, FFT_traits::mul(rts[k], res[k]));
                    }
                }
            }

            for (long k = 0; k < task_size; k++) {
                if (res[k] == 0) {
                    pthread_spin_lock(&lock);
                    if (num_ret < FFT_traits::roots_limit) {
                        ret[num_ret++] = rts[k];
                    }
                    pthread_spin_unlock(&lock);
                }
            }
        }

        for (long i = 0; i < p->pages; i++) {
            cm.release_del(p->page[i]);
        }
    } else {
        const long total_tasks = p->pages;
        uint64_t *res = (uint64_t *) calloc(num_roots, sizeof(uint64_t));
        #pragma omp parallel
        {
            const long thread_id = omp_get_thread_num();
            const long num_threads = omp_get_num_threads();
            uint64_t *thread_res = (uint64_t *) calloc(num_roots, sizeof(uint64_t));
            uint64_t *tmp = (uint64_t *) malloc(num_roots * sizeof(uint64_t));
            for (long i = thread_id; i < total_tasks; i += num_threads) {
                chunk_t *ck = cm.fetch(p->page[i]);
                memset(tmp, 0, num_roots * sizeof(uint64_t));
                for (long j = page_u64 - 1; j >= 0; j--) {
                    uint64_t val = ck->data[j];
                    for (long k = 0; k < num_roots; k++) {
                        tmp[k] = FFT_traits::add(val, FFT_traits::mul(roots_list[k], tmp[k]));
                    }
                }
                for (long k = 0; k < num_roots; k++) {
                    tmp[k] = FFT_traits::mul(tmp[k], FFT_traits::pow(roots_list[k], i * page_u64));
                    thread_res[k] = FFT_traits::add(thread_res[k], tmp[k]);
                }
                cm.release_del(p->page[i]);
            }

            pthread_spin_lock(&lock);
            for (long i = 0; i < num_roots; i++) {
                res[i] = FFT_traits::add(res[i], thread_res[i]);
            }
            pthread_spin_unlock(&lock);
            free(thread_res);
            free(tmp);
        }

        for (long i = 0; i < num_roots; i++) {
            if (res[i] == 0 && num_ret < FFT_traits::roots_limit) {
                ret[num_ret++] = roots_list[i];
            }
        }
        
        free(res);
    }

    ret[num_ret] = -1ULL;
    pthread_spin_destroy(&lock);
    
    free(roots_list);
    p->pages = 0;
}

template <class FFT_traits>
void period_red(poly_t *p, long target_size, long period, uint64_t ratio) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);

    const long fft_pages = FFT_traits::fft_pages(target_size);

    const long total_tasks = (period - 1) / page_u64 + 1;
    int *task_list = (int *) malloc(sizeof(int) * total_tasks);
    for (long i = 0; i < total_tasks; i++) {
        task_list[i] = i;
    }
    for (long i = 0; i < total_tasks; i++) {
        int src = rand() % total_tasks;
        int tmp = task_list[i];
        task_list[i] = task_list[src];
        task_list[src] = tmp;
    }

    int *chunk_used = (int *) calloc(fft_pages, sizeof(int));

    #pragma omp parallel for schedule(guided)
    for (long task_id = 0; task_id < total_tasks; task_id++) {
        const long task = task_list[task_id];
        const long task_size = task == total_tasks - 1 ? period - task * page_u64 : page_u64;

        chunk_t *ck_acc = cm.fetch(p->page[task]);
        uint64_t seg_ratio = ratio;
        
        for (long seg = task * page_u64 + period; seg < fft_pages * page_u64; seg += period) {
            const long seg_id = seg / page_u64;
            const long seg_pos = seg % page_u64;
            
            chunk_t   *ck_seg0 = seg_id != task ? cm.fetch(p->page[seg_id]) : ck_acc;
            uint64_t *seg0_ptr = ck_seg0->data + seg_pos;
            long      seg0_num = page_u64 - seg_pos > task_size ? task_size : page_u64 - seg_pos;
            
            for (long i = 0; i < seg0_num; i++) {
                ck_acc->data[i] = FFT_traits::add(ck_acc->data[i], FFT_traits::mul(seg0_ptr[i], seg_ratio));
            }

            chunk_used[seg_id] += seg0_num;
            if (seg_id != task) {
                if (chunk_used[seg_id] == page_u64) cm.release_del(p->page[seg_id]);
                else cm.release(p->page[seg_id]);
            }

            if (seg0_num < task_size && seg_id != fft_pages - 1) {
                chunk_t  *ck_seg1 = cm.fetch(p->page[seg_id + 1]);
                uint64_t *seg1_ptr = ck_seg1->data;
                long      seg1_num = task_size - seg0_num;
                for (long i = 0; i < seg1_num; i++) {
                    ck_acc->data[i + seg0_num] = FFT_traits::add(ck_acc->data[i + seg0_num], FFT_traits::mul(seg1_ptr[i], seg_ratio));
                }
                chunk_used[seg_id + 1] += seg1_num;
                if (chunk_used[seg_id + 1] == page_u64) cm.release_del(p->page[seg_id + 1]);
                else cm.release(p->page[seg_id + 1]);
            }
            
            seg_ratio = FFT_traits::mul(seg_ratio, ratio);   
        }

        cm.release_sync(p->page[task]);
    }
    
    if (period % page_u64) {
        chunk_t *ck_acc = cm.fetch(p->page[total_tasks - 1]);
        memset(ck_acc->data + (period % page_u64), 0, (page_u64 - period % page_u64) * sizeof(uint64_t));
        cm.release_sync(p->page[total_tasks - 1]);
    }

    p->pages = total_tasks;
    p->alloc_page(FFT_traits::fft_pages(period));

    #if 1
    chunk_used[total_tasks - 1] += period % page_u64;
    for (long i = total_tasks; i < fft_pages; i++) {
        if (chunk_used[i] != page_u64) {
            lg_err("period_red: chunk %ld used only %d elements", i, chunk_used[i]);
        }
    }
    #endif

    free(task_list);
    free(chunk_used);
}

template <class FFT_traits>
void tg2(poly_t *p, long target_size) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);
    
    const long fft_pages = FFT_traits::fft_pages(target_size);
    
    poly_t f0(fft_pages), f1(fft_pages);
    f0.alloc_page();
    f1.alloc_page();

    #pragma omp parallel for schedule(guided)
    for (long task = 0; task < fft_pages; task += 2) {
        const long task_pages = fft_pages - task < 2 ? 1 : 2;
        chunk_t *ck_p0 = cm.fetch(p->page[task]);
        chunk_t *ck_p1 = task_pages == 2 ? 
                         cm.fetch(p->page[task + 1]) : NULL;
        chunk_t *ck_f0 = cm.fetch(f0.page[task / 2]);
        chunk_t *ck_f1 = cm.fetch(f1.page[task / 2]);

        for (long i = 0; i < page_u64 / 2; i++) {
            ck_f0->data[i] = ck_p0->data[2 * i + 0];
            ck_f1->data[i] = ck_p0->data[2 * i + 1];
        }

        if (ck_p1) {
            for (long i = 0; i < page_u64 / 2; i++) {
                ck_f0->data[i + page_u64 / 2] = ck_p1->data[2 * i + 0];
                ck_f1->data[i + page_u64 / 2] = ck_p1->data[2 * i + 1];
            }
            cm.release_del(ck_p1->id);
        }

        cm.release_del(ck_p0->id);
        cm.release_sync(ck_f0->id);
        cm.release_sync(ck_f1->id);
    }

    fft<FFT_traits>(&f0, target_size);
    fft<FFT_traits>(&f1, target_size, 2, &f0);
    invfft<FFT_traits>(&f0, target_size);

    for (long i = 0; i < fft_pages; i++) {
        p->page[i] = f0.page[i];
    }

    f0.pages = 0;
}

template <class FFT_traits>
void compose(poly_t *dst, poly_t *p, long target_size, uint64_t ratio) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);

    const long fft_pages = FFT_traits::fft_pages(target_size);
    const uint64_t page_bias = FFT_traits::pow(ratio, page_u64);

    #pragma omp parallel for schedule(guided)
    for (long task = 0; task < fft_pages; task++) {
        chunk_t *ck_p = cm.fetch(p->page[task]);
        chunk_t *ck_dst = cm.fetch(dst->page[task]);
        uint64_t bias = FFT_traits::pow(page_bias, task);
        for (long i = 0; i < page_u64; i++) {
            ck_dst->data[i] = FFT_traits::mul(ck_p->data[i], bias);
            bias = FFT_traits::mul(bias, ratio);
        }
        cm.release(ck_p->id);
        cm.release_sync(ck_dst->id);
    }
}

template <class FFT_traits>
void collect(poly_t *dst, poly_t *p, long target_size, long tg) {
    constexpr long page_u64 = poly_t::page_size / sizeof(uint64_t);
    
    const long fft_pages = FFT_traits::fft_pages(target_size);
    dst->alloc_page(fft_pages);
    
    const long total_tasks = (target_size - 1) / page_u64 + 1;
    int *task_list = (int *) malloc(sizeof(int) * total_tasks);
    for (long i = 0; i < total_tasks; i++) {
        task_list[i] = i;
    }
    for (long i = 0; i < total_tasks; i++) {
        int src = rand() % total_tasks;
        int tmp = task_list[i];
        task_list[i] = task_list[src];
        task_list[src] = tmp;
    }

    #pragma omp parallel for schedule(guided)
    for (long task_id = 0; task_id < total_tasks; task_id++) {
        const long task = task_list[task_id];
        const long task_size = task < total_tasks - 1 ? page_u64 : target_size - task * page_u64;
        int pos = 0;
        chunk_t *ck_dst = cm.fetch(dst->page[task]);
        for (long src_ptr = task * tg; src_ptr < task * tg + tg; src_ptr++) {
            chunk_t *ck_src = cm.fetch(p->page[src_ptr]);
            long j_start = ((-src_ptr * page_u64) % tg + tg) % tg;
            for (long j = j_start; j < page_u64; j += tg) {
                ck_dst->data[pos++] = ck_src->data[j];
            }
            cm.release_del(ck_src->id);
            if (pos >= task_size) break;
        }
        cm.release_sync(ck_dst->id);
    }

    free(task_list);

    #pragma omp parallel for schedule(guided)
    for (long i = (tg * target_size - 1) / page_u64 + 1; i < FFT_traits::fft_pages(target_size * tg); i++) {
        cm.fetch(p->page[i]);
        cm.release_del(p->page[i]);
    }
    p->pages = 0;
}

template <class FFT_traits>
void tg3(poly_t *p, long target_size) {
    const long x1pages = FFT_traits::fft_pages(target_size * 1L);
    const long x3pages = FFT_traits::fft_pages(target_size * 3L);
    poly_t f(x3pages), g(x3pages);
    f.alloc_page(x3pages);
    g.alloc_page(x3pages);
    compose<FFT_traits>(&f, p, target_size, FFT_traits::pow(FFT_traits::rou3, 1));
    compose<FFT_traits>(&g, p, target_size, FFT_traits::pow(FFT_traits::rou3, 2));
    fft<FFT_traits>(&f, target_size * 3L);
    fft<FFT_traits>(&g, target_size * 3L, 1, &f);
    g.pages = p->pages;
    p->pages = 0;
    for (long i = 0; i < x1pages; i++) {
        g.page[i] = p->page[i];
    }
    g.alloc_page(x3pages);
    fft<FFT_traits>(&g, target_size * 3L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 3L);
    collect<FFT_traits>(p, &f, target_size, 3L);
}

template <class FFT_traits>
void tg5(poly_t *p, long target_size) {
    const long x1pages = FFT_traits::fft_pages(target_size * 1L);
    const long x2pages = FFT_traits::fft_pages(target_size * 2L);
    const long x5pages = FFT_traits::fft_pages(target_size * 5L);
    poly_t f(x5pages), g(x5pages);
    f.alloc_page(x2pages);
    g.alloc_page(x2pages);
    compose<FFT_traits>(&f, p, target_size, FFT_traits::pow(FFT_traits::rou5, 1));
    compose<FFT_traits>(&g, p, target_size, FFT_traits::pow(FFT_traits::rou5, 3));
    fft<FFT_traits>(&f, target_size * 2L);
    fft<FFT_traits>(&g, target_size * 2L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 2L);
    g.alloc_page(x5pages);
    compose<FFT_traits>(&g, &f, target_size * 2, FFT_traits::pow(FFT_traits::rou5, 1));
    f.alloc_page(x5pages);
    fft<FFT_traits>(&f, target_size * 5L);
    fft<FFT_traits>(&g, target_size * 5L, 1, &f);
    g.pages = p->pages;
    p->pages = 0;
    for (long i = 0; i < x1pages; i++) {
        g.page[i] = p->page[i];
    }
    g.alloc_page(x5pages);
    fft<FFT_traits>(&g, target_size * 5L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 5L);
    collect<FFT_traits>(p, &f, target_size, 5L);
}

template <class FFT_traits>
void tg17(poly_t *p, long target_size) {
    const long x1pages = FFT_traits::fft_pages(target_size * 1L);
    const long x2pages = FFT_traits::fft_pages(target_size * 2L);
    const long x4pages = FFT_traits::fft_pages(target_size * 4L);
    const long x8pages = FFT_traits::fft_pages(target_size * 8L);
    const long x17pages = FFT_traits::fft_pages(target_size * 17L);
    poly_t f(x17pages), g(x17pages);
    f.alloc_page(x2pages);
    g.alloc_page(x2pages);
    compose<FFT_traits>(&f, p, target_size, FFT_traits::pow(FFT_traits::rou17, 1));
    compose<FFT_traits>(&g, p, target_size, FFT_traits::pow(FFT_traits::rou17, 9));
    fft<FFT_traits>(&f, target_size * 2L);
    fft<FFT_traits>(&g, target_size * 2L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 2L);
    g.alloc_page(x4pages);
    compose<FFT_traits>(&g, &f, target_size * 2L, FFT_traits::pow(FFT_traits::rou17, 4));
    f.alloc_page(x4pages);
    fft<FFT_traits>(&f, target_size * 4L);
    fft<FFT_traits>(&g, target_size * 4L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 4L);
    g.alloc_page(x8pages);
    compose<FFT_traits>(&g, &f, target_size * 4L, FFT_traits::pow(FFT_traits::rou17, 2));
    f.alloc_page(x8pages);
    fft<FFT_traits>(&f, target_size * 8L);
    fft<FFT_traits>(&g, target_size * 8L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 8L);
    g.alloc_page(x17pages);
    compose<FFT_traits>(&g, &f, target_size * 8L, FFT_traits::pow(FFT_traits::rou17, 1));
    f.alloc_page(x17pages);
    fft<FFT_traits>(&f, target_size * 17L);
    fft<FFT_traits>(&g, target_size * 17L, 1, &f);
    g.pages = p->pages;
    p->pages = 0;
    for (long i = 0; i < x1pages; i++) {
        g.page[i] = p->page[i];
    }
    g.alloc_page(x17pages);
    fft<FFT_traits>(&g, target_size * 17L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 17L);
    collect<FFT_traits>(p, &f, target_size, 17L);
}

template <class FFT_traits>
void tg257(poly_t *p, long target_size) {
    const long x1pages = FFT_traits::fft_pages(target_size * 1L);
    const long x2pages = FFT_traits::fft_pages(target_size * 2L);
    const long x4pages = FFT_traits::fft_pages(target_size * 4L);
    const long x8pages = FFT_traits::fft_pages(target_size * 8L);
    const long x16pages = FFT_traits::fft_pages(target_size * 16L);
    const long x32pages = FFT_traits::fft_pages(target_size * 32L);
    const long x64pages = FFT_traits::fft_pages(target_size * 64L);
    const long x128pages = FFT_traits::fft_pages(target_size * 128L);
    const long x257pages = FFT_traits::fft_pages(target_size * 257L);
    poly_t f(x257pages), g(x257pages);
    f.alloc_page(x2pages);
    g.alloc_page(x2pages);
    compose<FFT_traits>(&f, p, target_size, FFT_traits::pow(FFT_traits::rou257, 1));
    compose<FFT_traits>(&g, p, target_size, FFT_traits::pow(FFT_traits::rou257, 129));
    fft<FFT_traits>(&f, target_size * 2L);
    fft<FFT_traits>(&g, target_size * 2L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 2L);
    g.alloc_page(x4pages);
    compose<FFT_traits>(&g, &f, target_size * 2L, FFT_traits::pow(FFT_traits::rou257, 64));
    f.alloc_page(x4pages);
    fft<FFT_traits>(&f, target_size * 4L);
    fft<FFT_traits>(&g, target_size * 4L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 4L);
    g.alloc_page(x8pages);
    compose<FFT_traits>(&g, &f, target_size * 4L, FFT_traits::pow(FFT_traits::rou257, 32));
    f.alloc_page(x8pages);
    fft<FFT_traits>(&f, target_size * 8L);
    fft<FFT_traits>(&g, target_size * 8L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 8L);
    g.alloc_page(x16pages);
    compose<FFT_traits>(&g, &f, target_size * 8L, FFT_traits::pow(FFT_traits::rou257, 16));
    f.alloc_page(x16pages);
    fft<FFT_traits>(&f, target_size * 16L);
    fft<FFT_traits>(&g, target_size * 16L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 16L);
    g.alloc_page(x32pages);
    compose<FFT_traits>(&g, &f, target_size * 16L, FFT_traits::pow(FFT_traits::rou257, 8));
    f.alloc_page(x32pages);
    fft<FFT_traits>(&f, target_size * 32L);
    fft<FFT_traits>(&g, target_size * 32L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 32L);
    g.alloc_page(x64pages);
    compose<FFT_traits>(&g, &f, target_size * 32L, FFT_traits::pow(FFT_traits::rou257, 4));
    f.alloc_page(x64pages);
    fft<FFT_traits>(&f, target_size * 64L);
    fft<FFT_traits>(&g, target_size * 64L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 64L);
    g.alloc_page(x128pages);
    compose<FFT_traits>(&g, &f, target_size * 64L, FFT_traits::pow(FFT_traits::rou257, 2));
    f.alloc_page(x128pages);
    fft<FFT_traits>(&f, target_size * 128L);
    fft<FFT_traits>(&g, target_size * 128L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 128L);
    g.alloc_page(x257pages);
    compose<FFT_traits>(&g, &f, target_size * 128L, FFT_traits::pow(FFT_traits::rou257, 1));
    f.alloc_page(x257pages);
    fft<FFT_traits>(&f, target_size * 257L);
    fft<FFT_traits>(&g, target_size * 257L, 1, &f);
    g.pages = p->pages;
    p->pages = 0;
    for (long i = 0; i < x1pages; i++) {
        g.page[i] = p->page[i];
    }
    g.alloc_page(x257pages);
    fft<FFT_traits>(&g, target_size * 257L, 1, &f);
    invfft<FFT_traits>(&f, target_size * 257L);
    collect<FFT_traits>(p, &f, target_size, 257L);
}

// Explicit instantiation for fft64
template uint64_t* roots<fft64>(poly_t *p, long target_size, long period, uint64_t ratio);
template void period_red<fft64>(poly_t *p, long target_size, long period, uint64_t ratio);
template void enum_roots<fft64>(long &num_ret, uint64_t *ret, poly_t *p, long period);
template void compose<fft64>(poly_t *dst, poly_t *p, long target_size, uint64_t ratio);
template void collect<fft64>(poly_t *dst, poly_t *p, long target_size, long tg);
template void tg2<fft64>(poly_t *p, long target_size);
template void tg3<fft64>(poly_t *p, long target_size);
template void tg5<fft64>(poly_t *p, long target_size);
template void tg17<fft64>(poly_t *p, long target_size);
template void tg257<fft64>(poly_t *p, long target_size);