import subprocess, sys, os

# Regenerates every record and figure, then runs the consistency gate. Run from the code/ directory.
# Total run time is about 17 minutes on one core; the two closed-loop checks append one record per run.
os.makedirs('../records', exist_ok=True); os.makedirs('../figures', exist_ok=True)
for f in ('../records/closed_loop.jsonl', '../records/adapt_yield.jsonl', '../records/mech_arm.jsonl'):
    if os.path.exists(f):
        os.remove(f)
steps = [('check_decrease.py', [], '../records/log_check_decrease.txt'),
         ('check_closed_loop.py', [], None),
         ('check_adapt_yield.py', [], None),
         ('check_authority_panda.py', [], None),
         ('check_mech_arm.py', [], None),
         ('fig_yield_core.py', [], '../records/log_fig_yield_core.txt'),
         ('check_sync.py', [], None)]
for script, args, log in steps:
    out = subprocess.run([sys.executable, script, *args], capture_output=True, text=True)
    print(f'[{script}] exit {out.returncode}\n{out.stdout}{out.stderr}', flush=True)
    if log:
        open(log, 'w').write(out.stdout)
    if out.returncode != 0:
        sys.exit(out.returncode)
