# Cache Manager

A compact, header-only cache manager for fixed-size data pages (chunks). It provides in-memory caching plus asynchronous load and sync to disk. 

## Key API:
- `long create_chunk()` — allocate or reuse a chunk id.
- `long prefetch(long id)` — async load.
- `Chunk* fetch(long id)` — block until cached and return pointer.
- `long release(long id)`, `long release_sync(long id)`, `long release_del(long id)`.
- `long wait_work()` — wait for all async tasks.
- `long set_max_cached_chunks(long n)` — adjust cache size (may fail on timeout).
