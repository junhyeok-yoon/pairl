"""Parallel CPU rollouts for diagnostics and evaluations (no training): a drop-in replacement for the function returned by l2.make_batch_fn.
The episode axis is split into chunks of one size that run in worker processes, one per core, each pinned to its core and single-threaded.
Allowed sizes are 2, 4, 8, 16, 32, 64 (bit-identical to the unsplit batch on the port-check episodes of this PC; sizes 1, 3 and 7 were not,
records/parallel_check.log). chunk='auto' takes the smallest allowed size that gives at most one chunk per worker, capped at 32. The last chunk is padded
with copies of its last episode and the padding is dropped. Workers share a persistent compilation cache, so each chunk size is compiled about once.
Memory: workers are limited to what fits the memory budget (cgroup limit or MemAvailable minus 4 GB) after the parent and max_out_gb; chunk results go into
one preallocated output (numpy arrays, no second copy) and a call whose output would exceed max_out_gb is refused
(on 2026-11-15 a 4096-episode call held several copies, about 12 GB, and the kernel OOM killer also took the desktop editor).
Module state that shapes the trace (env.COST_V, l2.ACT_MODE, l2.OBS_TIP) and cfg are read in the parent at every call and set in the worker before tracing."""
import os, pickle, hashlib, multiprocessing as mp
import numpy as np

WORKER_ENV = {'JAX_PLATFORMS': 'cpu', 'XLA_FLAGS': '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1',
              'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
_W = {}
WORKER_GB = float(os.environ.get("PAIRL_WORKER_GB", 0.8))   # per-worker memory charge (measured 0.7 GB per worker, records/parallel_check.log)
PARENT_GB = 3.0                                                # parent process before the output (measured 2.8 GB)


def memory_budget_gb():
    """Memory this job may use: the cgroup limit of this process (script/safe_run.sh sets one) or MemAvailable minus 4 GB for the desktop, whichever is smaller."""
    avail = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable')) / 1e6 - 4.0
    try:
        cg = open('/proc/self/cgroup').read().strip().split('::')[-1]
        lim = open(f'/sys/fs/cgroup{cg}/memory.max').read().strip()
        if lim != 'max': avail = min(avail, int(lim) / 1e9)
    except OSError: pass
    return avail


ALLOWED = (2, 4, 8, 16, 32, 64)
from pairl import paths
CACHE = str(paths.TMP / 'jax_cache')


def _init(cores):
    os.sched_setaffinity(0, {cores.get()})
    import jax; jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_compilation_cache_dir', CACHE); jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    _W['fns'] = {}


def _run(args):
    import jax
    from pairl.learn import l2
    from pairl.stageb import env
    i, cfg, ck, pb, tb, keys, bp, hp, mode, blind, cost_v, act_mode, obs_tip = args
    env.COST_V = cost_v; l2.ACT_MODE = act_mode; l2.OBS_TIP = obs_tip
    k = (cost_v, act_mode, obs_tip, ck)
    if k not in _W['fns']: _W['fns'][k] = l2.make_batch_fn(cfg)
    return i, jax.tree_util.tree_map(np.asarray, _W['fns'][k](pb, tb, keys, bp, hp, mode, blind))


class ParallelBatch:
    """Callable with the signature of l2.make_batch_fn(cfg): (pb, tb, keys, bp, hp, mode, blind) -> (st, m, dec, rec), plus cfg as the first argument."""

    def __init__(self, workers=None, chunk='auto', cores=None, max_out_gb=4.0):
        assert chunk == 'auto' or int(chunk) in ALLOWED, f'chunk must be auto or one of {ALLOWED}'
        cores = list(cores) if cores is not None else sorted(os.sched_getaffinity(0))
        cap = max(1, int((memory_budget_gb() - PARENT_GB - max_out_gb) / WORKER_GB))
        self.workers = min(workers or len(cores), len(cores), cap); self.chunk = chunk; self.max_out = max_out_gb * 1e9
        if workers and self.workers < workers: print(f'parallel: {workers} workers requested, {self.workers} fit the memory budget', flush=True)
        ctx = mp.get_context('spawn'); q = ctx.Queue()
        for c in cores[:self.workers]: q.put(c)
        old = {k: os.environ.get(k) for k in WORKER_ENV}
        os.environ.update(WORKER_ENV)
        try: self.pool = ctx.Pool(self.workers, initializer=_init, initargs=(q,))
        finally:
            for k, v in old.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v

    def __call__(self, cfg, pb, tb, keys, bp, hp, mode, blind=0.0):
        import jax, jax.numpy as jnp
        from pairl.learn import l2
        from pairl.stageb import env
        tonp = lambda t: jax.tree_util.tree_map(np.asarray, t)
        cfg = {k: (np.asarray(v) if not isinstance(v, (int, float)) else v) for k, v in cfg.items()}
        ck = hashlib.sha256(pickle.dumps(sorted((k, v.tobytes() if isinstance(v, np.ndarray) else v) for k, v in cfg.items()))).hexdigest()
        pb, tb, keys, bp, hp = tonp(pb), tonp(tb), np.asarray(keys), tonp(bp), tonp(hp)
        n = keys.shape[0]; tasks = []
        S = int(self.chunk) if self.chunk != 'auto' else next((c for c in ALLOWED if c * self.workers >= n), 32)
        S = min(S, 32) if self.chunk == 'auto' else S; self.last_chunk = S
        for i, a in enumerate(range(0, n, S)):
            idx = np.minimum(np.arange(a, a + S), min(a + S, n) - 1)
            sl = lambda t: jax.tree_util.tree_map(lambda x: x[idx], t)
            tasks.append((i, cfg, ck, sl(pb), sl(tb), keys[idx], bp, hp, float(mode), float(blind), env.COST_V, l2.ACT_MODE, l2.OBS_TIP))
        out = None
        for i, part in self.pool.imap_unordered(_run, tasks):
            if out is None:
                need = sum(x.nbytes for x in jax.tree_util.tree_leaves(part)) / S * n
                if need > self.max_out:
                    self.pool.terminate(); raise MemoryError(f'output of {n} episodes needs {need / 1e9:.1f} GB > max_out_gb {self.max_out / 1e9:.1f}; use smaller batches')
                out = jax.tree_util.tree_map(lambda x: np.empty((n,) + x.shape[1:], x.dtype), part)
            a = i * S; b = min(a + S, n)
            jax.tree_util.tree_map(lambda o, x: o.__setitem__(slice(a, b), x[:b - a]), out, part); del part
        return out

    def batch_fn(self, cfg):
        return lambda pb, tb, keys, bp, hp, mode, blind=0.0: self(cfg, pb, tb, keys, bp, hp, mode, blind)

    def close(self):
        self.pool.close(); self.pool.join()
