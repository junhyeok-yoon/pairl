"""Experiment groups of data/runs (rule of 2026-11-16): data/runs/<YYYYMMDDHHMMSS>/<run>/, manifest.json per group, data/runs/INDEX.md.
A launcher calls open_run() before training and close_run() at the end; the manifest records the plan and its hash, the git commit (and a hash of the
diff if the tree is dirty), the exact command, the platform, the start and end times, the status and the purpose of the group."""
import os, sys, json, hashlib, platform, subprocess, datetime
from pairl import paths

now = lambda: datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')


CODE = ('script/l2_train.py', 'src/pairl/learn', 'src/pairl/stageb', 'src/pairl/paths.py', 'pyproject.toml')   # code that determines the numerics of a run;
                                                                 # bookkeeping (src/pairl/tools), evaluation, figure and report scripts are not part of it


def _git():
    """Code state: HEAD commit and, if the code paths differ from HEAD, a hash of their diff and of the untracked code files (names and contents).
    Only the code paths count; records, figures, run outputs and documents do not."""
    run = lambda *a: subprocess.run(['git', '-C', str(paths.ROOT), *a], capture_output=True, text=True).stdout
    commit = run('rev-parse', 'HEAD').strip() or None
    diff = run('diff', 'HEAD', '--', *CODE)
    untracked = sorted(f for f in run('ls-files', '--others', '--exclude-standard', '--', *CODE).split('\n') if f.endswith(('.py', '.toml')))
    h = hashlib.sha256(diff.encode())
    for f in untracked: h.update(f.encode() + b'\0' + open(paths.ROOT / f, 'rb').read())
    dirty = bool(diff.strip() or untracked)
    return dict(git_commit=commit, dirty=dirty, diff_sha256=h.hexdigest() if dirty else None,
                note='hash of the diff against HEAD and of the untracked files of the training path (CODE in pairl.tools.runs)' if dirty else None)


def _platform():
    import jax
    cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor())
    d = jax.devices()[0]
    return dict(device=d.platform.upper(), jax_device=str(d), device_kind=getattr(d, 'device_kind', None), cpu=cpu, host=platform.node(),
                python=platform.python_version(), jax=jax.__version__, x64=bool(jax.config.jax_enable_x64), xla_flags=os.environ.get('XLA_FLAGS'))


def _index_line(m):
    ts = m['timestamp']; link = f"[{ts}]({ts}/summary.md)" if (paths.RUNS / ts / 'summary.md').exists() else ts
    sc = m.get('success'); sx = f"{sc['P1']:.3f} / {sc['full_chain']:.3f} ({sc['checkpoint']})" if sc else '-'
    return f"| {link} | {m['purpose']} | {m['plan']} | {m['status']} | {m.get('key_result') or ''} | {sx} |"


def _write_index(m):
    f = paths.RUNS / 'INDEX.md'; L = open(f).read().rstrip('\n').split('\n')
    for i, l in enumerate(L):                                         # column of the phase success (P1 / full chain), added 2026-11-16
        if l.startswith('| timestamp |') and 'P1 / full chain' not in l: L[i] = l + ' P1 / full chain (deterministic, CPU) |'
        elif l.startswith('|---') and l.count('|') == 6: L[i] = l + '---|'
        elif l.startswith('| ') and not l.startswith('| timestamp') and l.count(' | ') == 4: L[i] = l + ' - |'
    mine = lambda l: l.startswith(f"| {m['timestamp']} |") or l.startswith(f"| [{m['timestamp']}]")
    i = next((k for k, l in enumerate(L) if mine(l)), None)
    if i is None: L.append(_index_line(m))
    else: L = [l for k, l in enumerate(L) if k == i or not mine(l)]; L[i] = _index_line(m)       # one line per group
    open(f, 'w').write('\n'.join(L) + '\n')


def open_run(group, run, plan, purpose, argv):
    """group='new' creates data/runs/<now>/; group=<timestamp> adds a run to an existing group. Returns the run directory."""
    assert plan and os.path.exists(paths.ROOT / plan), 'plan= must name an existing plan file (records/<name>_plan.md)'
    hf = paths.ROOT / plan.replace('_plan.md', '_plan.hash').replace('.md', '.hash')
    assert hf.exists() and open(hf).read().split(), f'{hf.name}: no hash line (register the plan before running)'
    ph = hashlib.sha256(open(paths.ROOT / plan, 'rb').read()).hexdigest()[:8]
    assert ph == open(hf).read().split()[-1], f'plan {plan} changed since its last hash line ({ph} vs {open(hf).read().split()[-1]})'
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S') if group == 'new' else group
    gd = paths.RUNS / ts; mf = gd / 'manifest.json'
    if group == 'new': gd.mkdir(parents=False, exist_ok=False); m = dict(timestamp=ts, purpose=purpose, plan=plan, plan_sha256_prefix=ph, plan_hash_file=str(hf.relative_to(paths.ROOT)),
                                                                     code=_git(), platform=_platform(), commands=[], runs={}, start=now(), end=None, status='running', key_result=None)
    else:
        assert mf.exists(), f'group {ts} does not exist'; m = json.load(open(mf))
        chain = open(hf).read().split()                                    # an amendment appends a hash line; the group's version must be in the chain
        assert m['plan'] == plan and m['plan_sha256_prefix'] in chain, 'a group holds one plan (amendments of it allowed)'
        assert m['code']['diff_sha256'] == _git()['diff_sha256'] and m['code']['git_commit'] == _git()['git_commit'], 'a group holds one code state'
    rd = gd / run; assert not rd.exists(), f'{rd} exists'; rd.mkdir()
    code_root = os.path.dirname(os.path.dirname(os.path.abspath(argv[0])))
    m['commands'].append(' '.join(['python', os.path.relpath(os.path.abspath(argv[0]), code_root)] + [a if ' ' not in a else repr(a) for a in argv[1:]]))
    m['runs'][run] = dict(start=now(), end=None, status='running', plan_sha256_prefix=ph); m['status'] = 'running'
    json.dump(m, open(mf, 'w'), indent=1); _write_index(m)
    return rd


def close_run(rd, status):
    """status: complete, stopped early or failed. The group is complete when all its runs are complete."""
    mf = rd.parent / 'manifest.json'; m = json.load(open(mf))
    m['runs'][rd.name].update(end=now(), status=status); m['end'] = now()
    st = {r['status'] for r in m['runs'].values()}
    m['status'] = 'running' if 'running' in st else ('complete' if st == {'complete'} else ('failed' if 'failed' in st else 'stopped early'))
    json.dump(m, open(mf, 'w'), indent=1); _write_index(m)
