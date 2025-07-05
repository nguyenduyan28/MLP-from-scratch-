## Dataloader note
* To make classes can be iterable, we should define `__iter()__` and `__next__()` 
  * `__iter()__`: Return a iterator which has `__next__()`
  * `__next()__`: Return  next element in class, self custom
* To make the `num_workers` available, we using `multiprocessing.Pool` with init to share a large ds for each worker.
* Current workflow: init a pool, share ds for worker, each worker load `batch_size / num_worker` range in each session, at the end merge and return img_arr
* **TODO**: Make it faster than current single load (0.07)
* Solution by GPT:
  1. Profiling : Debug I/O time, CPU transform, ... to know where it is has bottleneck
  2. Reduce IPC overhead & pickle : Worker persitent, only copy ds 1 times (multiprocessing, persistent workers, fork vs spawn, copy-on-write)
  3. Remove copy when return batch : Worker save array into shared memory and send to main process or memory map. (shared_memory, mmamp, memmap numpy, posix_ipc)
  4. concat batch in worker : worker collate / stack before sending instead of concatenate (collate_fn, zero-copy tensor, default_collate)