"""Single source of the repository paths. The root is $PAIRL_ROOT if set, else the repository that contains this package."""
import os
from pathlib import Path

ROOT = Path(os.environ['PAIRL_ROOT']).resolve() if os.environ.get('PAIRL_ROOT') else Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RUNS = DATA / 'runs'
POOLS = DATA / 'pools'
RECORDS = ROOT / 'records'
DOCS = ROOT / 'docs'
NOTES = DOCS / 'notes'
FIGURES = DOCS / 'figures'
ROLLOUTS = FIGURES / 'rollouts'
GUIDELINES = DOCS / 'guidelines'
HANDOFF = DOCS / '20261115_handoff.md'
BUILD = ROOT / 'build'
LOCAL = ROOT / '_local'                    # git-ignored local material (bundle extract, baselines, Python base)
TMP = LOCAL / 'tmp'                         # temp files of this project (instead of /tmp); clean up after use


def run_dir(name):
    """Directory of a training run by name: data/runs/<group>/<name> (groups are timestamps), or data/runs/<name> if ungrouped. Exactly one match."""
    hits = [p for p in RUNS.glob(f'*/{name}') if p.is_dir() and p.parent.name.isdigit()] + ([RUNS / name] if (RUNS / name).is_dir() else [])
    if len(hits) != 1: raise FileNotFoundError(f'run {name!r}: {len(hits)} matches under {RUNS}')
    return hits[0]
