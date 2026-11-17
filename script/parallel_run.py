"""Launcher: runs a non-training diagnostic or evaluation script with its rollouts spread over CPU worker processes (one per core, single-threaded).
l2.make_batch_fn is replaced by pairl.tools.parallel before the target script runs; everything else of the target runs unchanged in this process
(on the CPU). Training (script/l2_train.py) is refused.
Usage: python script/parallel_run.py [workers=N] [chunk=auto|2|4|8|16|32|64] TARGET.py [target arguments ...]
  example: python script/parallel_run.py workers=24 script/port_check.py ref=pc"""
import os, sys, runpy
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

if __name__ == '__main__':
    args = sys.argv[1:]; opt = {}
    while args and '=' in args[0] and not args[0].endswith('.py'): k, v = args.pop(0).split('=', 1); opt[k] = v
    assert args and args[0].endswith('.py'), __doc__
    target = os.path.abspath(args[0])
    assert os.path.basename(target) != 'l2_train.py', 'training runs are not launched in parallel'
    import jax; jax.config.update('jax_enable_x64', True)
    from pairl.learn import l2
    from pairl.tools.parallel import ParallelBatch
    pb = ParallelBatch(workers=int(opt['workers']) if 'workers' in opt else None, chunk=opt.get('chunk', 'auto'))
    print(f'parallel_run: {pb.workers} worker processes, chunk {pb.chunk}, target {os.path.relpath(target)}', flush=True)
    l2.make_batch_fn = pb.batch_fn
    sys.argv = [target] + args[1:]
    try: runpy.run_path(target, run_name='__main__')
    finally: pb.close()
