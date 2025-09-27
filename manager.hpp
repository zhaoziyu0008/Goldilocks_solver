#ifndef __MANAGER_HPP
#define __MANAGER_HPP

#include <atomic>
#include <queue>

#include "resource.hpp"
#include "log.hpp"

template<typename T>
inline volatile T& vol(T& x) {
    return *reinterpret_cast<volatile T*>(&x);
}

inline int hash_to_dir(uint32_t id) {
    long ret = 0;
    ret = __builtin_popcount(id) + __builtin_popcount(id * 3) + (id & 3);
    ret %= 4;
    return ret;
}

template <class Chunk, class ThreadPool, class Lock>
struct cache_manager_t {
    public:
    typedef uint32_t Status;
    static constexpr Status _ck_loading       = 0x80000000;         // loading from disk
    static constexpr Status _ck_syncing       = 0x40000000;         // syncing to disk
    static constexpr Status _ck_caching       = 0x20000000;         // cached in memory
    static constexpr Status _ck_reading       = 0x10000000;         // cache is being used (read-only)
    static constexpr Status _ck_writing       = 0x08000000;         // cache is being used (read & write)
    static constexpr Status _ck_to_sync       = 0x04000000;         // need to sync to disk
    static constexpr Status _ck_size_mask     = 0x00003fff;
    static constexpr Status _ck_cache_id_mask = 0x03ffffff;

    static cache_manager_t& instance() {
        static cache_manager_t instance;
        return instance;
    }

    cache_manager_t(const cache_manager_t&) = delete;
    cache_manager_t& operator=(const cache_manager_t&) = delete;
    cache_manager_t(cache_manager_t&&) = delete;
    cache_manager_t& operator=(cache_manager_t&&) = delete;

    inline long num_chunks() const { return _num_chunks; }
    inline long num_empty() const { return _num_deleted_ids; }
    inline long max_cached_chunks() const { return _max_cached_chunks; }
    inline Status chunk_status(long chunk_id) const { return _chunk_status[chunk_id]; }

    /// @brief create a empty chunk, return the chunk_id
    long create_chunk();
    /// @brief try to prefetch a chunk
    long prefetch(long chunk_id);
    /// @note seems we will never read a single chunk through two threads
    ///       simultaneously, so do not distinguish read and write now
    /// @brief will stall until the chunk is ready
    Chunk *fetch(long chunk_id);
    /// @brief unlock the chunk only
    long release(long chunk_id);
    /// @brief first add the chunk to the list of chunks to sync, then unlock the chunk
    long release_sync(long chunk_id);
    /// @brief sync an in-writing chunk to disk and release it
    long release_del(long chunk_id);
    /// @brief wait until all tasks are done
    long wait_work();
    /// @warning only one thread can call these setting functions in the same time
    /// @warning no allocation will happen in it, but if many threads 
    ///          keeping accessing the same cached chunk, it may fail
    ///          to evict this chunk, I choose to give up after 10s
    long set_max_cached_chunks(long max_cached_chunks);
    
    
    ThreadPool loading_pool;
    ThreadPool syncing_pool;

    struct LockHolder {
        LockHolder(Lock *lk) : _lk(lk) { _lk->lock(); }
        
        LockHolder(cache_manager_t *cm, long id) : _lk(&cm->_locks[id % 256]) { _lk->lock(); }
        
        ~LockHolder() { if (_lk) _lk->unlock(); }
        
        private:
        Lock *_lk = NULL;
    };

    private:
    cache_manager_t();
    ~cache_manager_t();

    long    _num_chunks         = 0;
    Status *_chunk_status       = NULL;
    Lock    _locks[256];

    /// deleted chunks
    long    _num_deleted_ids    = 0;
    long   *_deleted_ids        = NULL;
    Lock    _deleted_ids_lock;

    /// cache
    long    _max_cached_chunks  = 0;
    long    _last_cache         = 0;
    Chunk  *_cached_chunks      = NULL;
    Lock    _cached_chunks_lock;

    /// when a chunk is asked to load, it will be use immediately so 
    /// I directly push the loading task to the thread pool, but syncing
    /// is not so urgent, I keep a list of chunks to sync
    std::atomic<int32_t> _num_loading_chunks{0};
    std::atomic<int32_t> _num_syncing_chunks{0};
    std::queue<int32_t>  _to_sync_chunks;
    Lock                 _to_sync_chunks_lock;

    int32_t __fetch_cache_for(long chunk_id);
    void __signal_sync_done();
    void __free_all();
    void __load_chunk(long chunk_id);
    void __sync_chunk(long chunk_id);
};

#include <fcntl.h>
#include <unistd.h>

struct chunk_t {
    static constexpr long max_cached_chunks = resource_holder_t::ram_pages;
    static constexpr long cache_only = resource_holder_t::ram_only;
    static constexpr long max_chunks = cache_only ? max_cached_chunks : resource_holder_t::ssd_pages;
    
    /// @brief mark the chunk as empty chunk but keep the memory
    long reset() {
        id = -1;
        return 0;
    }

    /// @brief allocate the buffer of the chunk, do nothing if already allocated
    long alloc() {
        if (data) return 0;
        return resource.alloc_page((void **)&data);
    }

    /// @brief free the buffer of the chunk, do nothing if not allocated
    long free() {
        if (!data) return 0;
        return resource.free_page(data);
    }

    /// @brief load the chunk from disk according to the current id
    long load() {
        if (cache_only) {
            memset(data, 0, resource_holder_t::page_size);
            return 0;
        }

        if (id < 0 || id >= max_chunks) {
            lg_err("chunk_t::load: id %d out of range", id);
            return -1;
        }

        char filename[256];
        snprintf(filename, sizeof(filename), ".data/%d/.ck%07x", hash_to_dir(id), id);
        int fd = open(filename, O_RDONLY);
        if (fd == -1) {
            if (errno != ENOENT) {
                lg_err("chunk_t::load: open %s failed", filename);
                return -1;
            }
            memset(data, 0, resource_holder_t::page_size);
            return 0;
        }
        
        long nbytes = read(fd, data, resource_holder_t::page_size);
        close(fd);

        if (nbytes != resource_holder_t::page_size) {
            lg_err("chunk_t::load: read %s returned %ld(expected %ld)", 
                     filename, nbytes, resource_holder_t::page_size);
            return -1;
        }
        
        return 0;
    }

    /// @brief sync the chunk to disk according to the current id
    long sync() {
        if (cache_only) return 0;

        if (id < 0 || id >= max_chunks) {
            lg_err("chunk_t::sync: id %d out of range", id);
            return -1;
        }

        char filename[256];
        snprintf(filename, sizeof(filename), ".data/%d/.ck%07x", hash_to_dir(id), id);

        int fd = open(filename, O_WRONLY | O_CREAT, 0644);
        if (fd == -1) {
            lg_err("chunk_t::sync: open %s failed", filename);
            return -1;
        }

        long nbytes = write(fd, data, resource_holder_t::page_size);
        if (nbytes != resource_holder_t::page_size) {
            lg_err("chunk_t::sync: write %s returned %ld(expected %ld)", 
                     filename, nbytes, resource_holder_t::page_size);
        }

        if (ftruncate(fd, nbytes) == -1) {
            lg_err("chunk_t::sync: ftruncate %s failed", filename);
            close(fd);
            return -1;
        }

        close(fd);
        return 0;
    }

    /// @brief remove the corresponding chunk file from disk
    static long del(long chunk_id) {
        if (cache_only) return 0;

        if (chunk_id < 0 || chunk_id >= max_chunks) {
            lg_err("chunk_t::del: id %d out of range", chunk_id);
            return -1;
        }

        char filename[256];
        snprintf(filename, sizeof(filename), ".data/%d/.ck%07lx", hash_to_dir(chunk_id), chunk_id);
        remove(filename);
        
        return 0;
    }
    
    static long del_all() {
        int ret0 = system("find .data/0/ -type f -name \".ck*\" -delete 2>/dev/null");
        int ret1 = system("find .data/1/ -type f -name \".ck*\" -delete 2>/dev/null");
        int ret2 = system("find .data/2/ -type f -name \".ck*\" -delete 2>/dev/null");
        int ret3 = system("find .data/3/ -type f -name \".ck*\" -delete 2>/dev/null");
        return ret0 == 0 && ret1 == 0 && ret2 == 0 && ret3 == 0 ? 0 : -1;
    }
    
    int32_t     id = -1;
    uint64_t *data = NULL;
};

#include <pthread.h>

struct lock_t {
    public:
    lock_t() { 
        pthread_spin_init(&lk, PTHREAD_PROCESS_SHARED); 
    }
    
    ~lock_t() { 
        pthread_spin_destroy(&lk); 
    }
    
    void lock() { 
        pthread_spin_lock(&lk); 
    }

    void try_lock() { 
        pthread_spin_trylock(&lk); 
    }

    void unlock() { 
        pthread_spin_unlock(&lk); 
    }

    pthread_spinlock_t lk;
};

template <class Chunk, class ThreadPool, class Lock>
cache_manager_t<Chunk, ThreadPool, Lock>::cache_manager_t() {
    _chunk_status  = (Status *) malloc((_ck_cache_id_mask + 1UL) * sizeof(Status));
    _cached_chunks = (Chunk *)  malloc((_ck_cache_id_mask + 1UL) * sizeof(Chunk));
    _deleted_ids   = (long *)   malloc((_ck_cache_id_mask + 1UL) * sizeof(long));

    loading_pool.resize(16);
    syncing_pool.resize(16);

    Chunk::del_all();

    set_max_cached_chunks(Chunk::max_cached_chunks);
}

template <class Chunk, class ThreadPool, class Lock>
cache_manager_t<Chunk, ThreadPool, Lock>::~cache_manager_t() {
    syncing_pool.wait_work();
    loading_pool.wait_work();
    __free_all();

    free(_chunk_status);
    free(_cached_chunks);
    free(_deleted_ids);
}

template <class Chunk, class ThreadPool, class Lock>
long cache_manager_t<Chunk, ThreadPool, Lock>::create_chunk() {
    if (_num_deleted_ids) {
        LockHolder lock(&_deleted_ids_lock);
        if (vol(_num_deleted_ids)) {
            long id = _deleted_ids[--_num_deleted_ids];
            _chunk_status[id] = 0;
            return id;
        }
    }

    for (;;) {
        long id = _num_chunks;
        if (id >= Chunk::max_chunks) break;
        LockHolder lock(this, id);
        if (vol(_num_chunks) == id) {
            _chunk_status[id] = 0;
            _num_chunks++;
            return id;
        }
    }

    LockHolder lock(&_deleted_ids_lock);
    if (vol(_num_deleted_ids)) {
        long id = _deleted_ids[--_num_deleted_ids];
        _chunk_status[id] = 0;
        return id;
    }

    lg_err("no more chunk id available, aborting");
    abort();
}

template <class Chunk, class ThreadPool, class Lock>
long cache_manager_t<Chunk, ThreadPool, Lock>::prefetch(long chunk_id) {
    constexpr Status _ck_already_loaded = _ck_caching | _ck_loading;

    if (chunk_id < 0 || chunk_id >= _num_chunks) return 0;
    if (_chunk_status[chunk_id] == _ck_size_mask) return 0;
    if (_chunk_status[chunk_id] & _ck_already_loaded) return 0;

    LockHolder lock(this, chunk_id);
    Status status = vol(_chunk_status[chunk_id]);
    if (!(status & _ck_already_loaded) && (status != _ck_size_mask) && !(status & (_ck_writing | _ck_reading))) {
        _chunk_status[chunk_id] |= _ck_loading;
        loading_pool.push([=]() { __load_chunk(chunk_id); });
        _num_loading_chunks++;
    }

    return 0;
}

template <class Chunk, class ThreadPool, class Lock>
Chunk *cache_manager_t<Chunk, ThreadPool, Lock>::fetch(long chunk_id) {
    constexpr Status _ck_already_loaded = _ck_caching | _ck_loading;
    constexpr Status _ck_busy = _ck_loading | _ck_syncing | _ck_reading | _ck_writing;
    
    if (_chunk_status[chunk_id] == _ck_size_mask) return NULL;

    for (;;) {
        if (!(vol(_chunk_status[chunk_id]) & _ck_already_loaded)) {
            LockHolder lock(this, chunk_id);
            Status status = vol(_chunk_status[chunk_id]);
            if (!(status & _ck_already_loaded) && !(status & _ck_busy)) {
                if (status == _ck_size_mask) return NULL;
                _chunk_status[chunk_id] |= _ck_loading;
                loading_pool.push([=]() { __load_chunk(chunk_id); });
                _num_loading_chunks++;
            }
        }
        
        if ((vol(_chunk_status[chunk_id]) & (_ck_busy | _ck_caching)) != _ck_caching) continue;
        
        LockHolder lock(this, chunk_id);
        Status status = vol(_chunk_status[chunk_id]);
        if ((status & (_ck_busy | _ck_caching)) != _ck_caching) continue;
        if (status == _ck_size_mask) return NULL;

        _chunk_status[chunk_id] |= _ck_writing;
        return &_cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask];
    }
}

template <class Chunk, class ThreadPool, class Lock>
long cache_manager_t<Chunk, ThreadPool, Lock>::release(long chunk_id) {
    _chunk_status[chunk_id] &= ~(_ck_writing | _ck_reading);

    return 0;
}

template <class Chunk, class ThreadPool, class Lock>
long cache_manager_t<Chunk, ThreadPool, Lock>::release_sync(long chunk_id) {
    if (_chunk_status[chunk_id] & _ck_to_sync) {
        _chunk_status[chunk_id] &= ~(_ck_writing | _ck_reading);
        return 0;
    }

    LockHolder lock(&_to_sync_chunks_lock);
    if (_num_syncing_chunks.load() < 6) {
        Status new_status = _chunk_status[chunk_id];
        new_status &= ~(_ck_writing | _ck_reading | _ck_to_sync);
        new_status |= _ck_syncing;
        _chunk_status[chunk_id] = new_status;
        syncing_pool.push([=]() { __sync_chunk(chunk_id); });
        _num_syncing_chunks++;
    } else {
        _chunk_status[chunk_id] |= _ck_to_sync;
        _to_sync_chunks.push(chunk_id);
        _chunk_status[chunk_id] &= ~(_ck_writing | _ck_reading);
    }

    return 0;
}

template <class Chunk, class ThreadPool, class Lock>
long cache_manager_t<Chunk, ThreadPool, Lock>::release_del(long chunk_id) {
    if ((_chunk_status[chunk_id] & _ck_syncing) == 0) {
        if (_chunk_status[chunk_id] & _ck_caching) {
            LockHolder lock(&_cached_chunks_lock);
            _cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask].reset();
        }

        Chunk::del(chunk_id);
        
        {
            LockHolder lock(this, chunk_id);
            _chunk_status[chunk_id] = _ck_size_mask;
        }

        {
            LockHolder lock(&_deleted_ids_lock);
            _deleted_ids[_num_deleted_ids++] = chunk_id;
        }

        return 0;
    }

    loading_pool.push([this, chunk_id]() {        
        do {} while (vol(_chunk_status[chunk_id]) & _ck_syncing);

        if (_chunk_status[chunk_id] & _ck_caching) {
            LockHolder lock(&_cached_chunks_lock);
            _cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask].reset();
        }
        
        Chunk::del(chunk_id);
        
        {
            LockHolder lock(this, chunk_id);
            _chunk_status[chunk_id] = _ck_size_mask;
        }
        
        {
            LockHolder lock(&_deleted_ids_lock);
            _deleted_ids[_num_deleted_ids++] = chunk_id;
        }
    });

    return 0;
}

template <class Chunk, class ThreadPool, class Lock>
long cache_manager_t<Chunk, ThreadPool, Lock>::wait_work() {
    syncing_pool.wait_work();
    loading_pool.wait_work();

    long ret = 0;

    long queue_size = _to_sync_chunks.size();
    if (queue_size) {
        lg_warn("%d chunks to sync after wait done?", queue_size);
        ret = -1;
    }

    if (_num_loading_chunks.load() || _num_syncing_chunks.load()) {
        lg_warn("wait_work: %d loading and %d syncing after wait done?", 
                _num_loading_chunks.load(), _num_syncing_chunks.load());
        ret = -1;
    }

    return ret;
}

template <class Chunk, class ThreadPool, class Lock>
long cache_manager_t<Chunk, ThreadPool, Lock>::set_max_cached_chunks(long max_cached_chunks) {
    constexpr Status _ck_busy = _ck_loading | _ck_syncing | _ck_reading | _ck_writing | _ck_to_sync;

    // need ptr
    #define TRY_LOCK_THEN_FREE_CHUNK(_ck_id)                                            \
        int32_t _id = _ck_id;                                                           \
        if (_id < 0) {                                                                  \
            LockHolder lock(&_cached_chunks_lock);                                      \
            if (vol(_ck_id) == -1) {                                                    \
                _ck_id = -2;                                                            \
                _cached_chunks[ptr].free();                                             \
                continue;                                                               \
            }                                                                           \
        }                                                                               \
        if ((_chunk_status[_id] & _ck_busy) == 0) {                                     \
            _locks[_id % 256].lock();                                                   \
            if ((vol(_chunk_status[_id]) & _ck_busy) == 0) {                            \
                _chunk_status[_id] |= _ck_busy;                                         \
                _locks[_id % 256].unlock();                                             \
                LockHolder lock(&_cached_chunks_lock);                                  \
                if (_id == vol(_ck_id)) {                                               \
                    _ck_id = -2;                                                        \
                    _chunk_status[_id] &= ~(_ck_cache_id_mask | _ck_busy | _ck_caching);\
                    _cached_chunks[ptr].free();                                         \
                    continue;                                                           \
                } else _chunk_status[_id] &= ~_ck_busy;                                 \
            } else _locks[_id % 256].unlock();                                          \
        }


    if (max_cached_chunks == this->_max_cached_chunks) return 0;

    if (max_cached_chunks < this->_max_cached_chunks) {
        long old_max_cached_chunks = this->_max_cached_chunks;
        this->_max_cached_chunks = max_cached_chunks;
        this->_last_cache = _last_cache % max_cached_chunks;
        
        int32_t *cache_to_evict = (int32_t *) malloc((old_max_cached_chunks - max_cached_chunks) * sizeof(int32_t));
        long num_cache_to_evict = 0;

        for (int32_t ptr = max_cached_chunks; ptr < old_max_cached_chunks; ptr++) {
            TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id);
            cache_to_evict[num_cache_to_evict++] = ptr;
        }

        struct timeval start, end;
        gettimeofday(&start, NULL);
        while (num_cache_to_evict) {
            int still_using = 0;
            for (long j = 0; j < num_cache_to_evict; j++) {
                int32_t ptr = cache_to_evict[j];
                TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id);
                cache_to_evict[still_using++] = ptr;
            }

            num_cache_to_evict = still_using;

            gettimeofday(&end, NULL);
            if ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6 > 10.0) {
                long k = 0;
                for (int32_t ptr = max_cached_chunks; ptr < old_max_cached_chunks; ptr++) {
                    if (ptr == cache_to_evict[k]) {
                        k++;
                        continue;
                    } else _cached_chunks[ptr] = Chunk();
                }
                
                this->_max_cached_chunks = old_max_cached_chunks;
                lg_err("some chunks are still in use after 10 seconds, aborted");
                if (k != num_cache_to_evict) lg_err("assertion k == num_cache_to_evict failed");
                free(cache_to_evict);

                return -1;
            }
        }
        free(cache_to_evict);
    } else {
        for (int32_t ptr = this->_max_cached_chunks; ptr < max_cached_chunks; ptr++) {
            _cached_chunks[ptr] = Chunk();
        }
        this->_max_cached_chunks = max_cached_chunks;
    }

    return 0;
}
    
template <class Chunk, class ThreadPool, class Lock>
int32_t cache_manager_t<Chunk, ThreadPool, Lock>::__fetch_cache_for(long chunk_id) {
    constexpr Status _ck_busy = _ck_loading | _ck_syncing | _ck_reading | _ck_writing | _ck_to_sync;

    for (int32_t cache_id = _last_cache + 1;; cache_id++) {
        if (cache_id >= _max_cached_chunks) cache_id %= _max_cached_chunks;

        int32_t old_chunk_id = _cached_chunks[cache_id].id;
        if (chunk_t::cache_only && old_chunk_id != -1) continue;

        if (old_chunk_id >= 0) {
            if (_chunk_status[old_chunk_id] & _ck_busy) continue;
        }
        
        if (old_chunk_id == -1) {
            LockHolder lock(&_cached_chunks_lock);
            if (vol(_cached_chunks[cache_id].id) == -1) {
                _cached_chunks[cache_id].id = chunk_id;
                _last_cache = cache_id;
                _cached_chunks[cache_id].alloc();
                return cache_id;
            }
        }

        if (old_chunk_id >= 0) {
            {
                LockHolder lock(this, old_chunk_id);
                if (vol(_chunk_status[old_chunk_id]) & _ck_busy) continue;
                _chunk_status[old_chunk_id] |= _ck_writing | _ck_reading;
            }
            
            _cached_chunks_lock.lock();
            if (old_chunk_id != vol(_cached_chunks[cache_id].id)) {
                _cached_chunks_lock.unlock();
                LockHolder lock(this, old_chunk_id);
                _chunk_status[old_chunk_id] &= ~(_ck_writing | _ck_reading);
                continue;
            }

            _cached_chunks[cache_id].id = chunk_id;
            _last_cache = cache_id;
            _cached_chunks_lock.unlock();
            
            {
                LockHolder lock(this, old_chunk_id);
                _chunk_status[old_chunk_id] &= ~(_ck_cache_id_mask | _ck_caching | _ck_writing | _ck_reading);
            }

            return cache_id;
        }
    }
}

template <class Chunk, class ThreadPool, class Lock>
void cache_manager_t<Chunk, ThreadPool, Lock>::__signal_sync_done() {
    constexpr Status _ck_busy = _ck_loading | _ck_syncing | _ck_reading | _ck_writing;

    long queue_size, num_try = 0;
    {
        LockHolder lock(&_to_sync_chunks_lock);
        queue_size = _to_sync_chunks.size();
    }

    if (queue_size > _max_cached_chunks * 5) {
        for (int i = 0; i < queue_size; i++) {
            int32_t chunk_id;

            {
                LockHolder lock(&_to_sync_chunks_lock);
                if (_to_sync_chunks.empty()) break;

                chunk_id = _to_sync_chunks.front();
                _to_sync_chunks.pop();

                if (_chunk_status[chunk_id] & (_ck_busy | _ck_to_sync)) {
                    _to_sync_chunks.push(chunk_id);
                    continue;
                }
            }
            _locks[chunk_id % 256].lock();
            if (vol(_chunk_status[chunk_id]) & (_ck_busy | _ck_to_sync)) {
                _locks[chunk_id % 256].unlock();
                {
                    LockHolder lock(&_to_sync_chunks_lock);
                    _to_sync_chunks.push(chunk_id);
                }
            } else _locks[chunk_id % 256].unlock();
        }
    }

    while (num_try < queue_size && _num_syncing_chunks.load() < 6) {
        int32_t chunk_id;
        {
            LockHolder lock(&_to_sync_chunks_lock);
            if (_to_sync_chunks.empty()) break;
            chunk_id = _to_sync_chunks.front();
            _to_sync_chunks.pop();
            
            num_try++;

            if (_chunk_status[chunk_id] & _ck_busy) {
                _to_sync_chunks.push(chunk_id);
                continue;
            }
        }
        
        _locks[chunk_id % 256].lock();
        Status status = vol(_chunk_status[chunk_id]);
        if (status & _ck_busy) {
            _locks[chunk_id % 256].unlock();
            {
                LockHolder lock(&_to_sync_chunks_lock);
                _to_sync_chunks.push(chunk_id);
            }
        } else if (!(status & _ck_to_sync)) {
            _locks[chunk_id % 256].unlock();
        } else {
            _chunk_status[chunk_id] = (status & (~_ck_to_sync)) | _ck_syncing;
            _locks[chunk_id % 256].unlock();
            syncing_pool.push([=]() { __sync_chunk(chunk_id); });
            _num_syncing_chunks++;
        }
    }
}

template <class Chunk, class ThreadPool, class Lock>
void cache_manager_t<Chunk, ThreadPool, Lock>::__free_all() {
    constexpr Status _ck_busy = _ck_loading | _ck_syncing | _ck_reading | _ck_writing | _ck_to_sync;
    
    int32_t *cache_to_evict = (int32_t *) malloc(_max_cached_chunks * sizeof(int32_t));
    long num_cache_to_evict = 0;

    for (int32_t ptr = 0; ptr < _max_cached_chunks; ptr++) {
        TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id);
        cache_to_evict[num_cache_to_evict++] = ptr;
    }

    while (num_cache_to_evict) {
        int still_using = 0;
        for (long j = 0; j < num_cache_to_evict; j++) {
            int32_t ptr = cache_to_evict[j];
            TRY_LOCK_THEN_FREE_CHUNK(_cached_chunks[ptr].id);
            cache_to_evict[still_using++] = ptr;
        }
        num_cache_to_evict = still_using;
    }
    free(cache_to_evict);

    #undef TRY_LOCK_THEN_FREE_CHUNK
}

template <class Chunk, class ThreadPool, class Lock>
void cache_manager_t<Chunk, ThreadPool, Lock>::__load_chunk(long chunk_id) {
    int32_t cache_id = __fetch_cache_for(chunk_id);
    _cached_chunks[cache_id].load();

    Status new_status = _chunk_status[chunk_id];
    new_status |= _ck_caching;
    new_status &= ~_ck_loading;
    new_status &= ~_ck_cache_id_mask;
    new_status |= cache_id;

    {
        LockHolder lock(this, chunk_id);
        _chunk_status[chunk_id] = new_status;
    }
    
    _num_loading_chunks--;
}

template <class Chunk, class ThreadPool, class Lock>
void cache_manager_t<Chunk, ThreadPool, Lock>::__sync_chunk(long chunk_id) {
    _cached_chunks[_chunk_status[chunk_id] & _ck_cache_id_mask].sync();

    {
        LockHolder lock(this, chunk_id);
        _chunk_status[chunk_id] &= ~_ck_syncing;
    }

    _num_syncing_chunks--;
    __signal_sync_done();
}

typedef cache_manager_t<chunk_t, thread_pool::thread_pool, lock_t> CM;

inline CM& cm = CM::instance();

#endif