#!/bin/sh
# usage: bash script/build_docs.sh <name>   (renders docs/notes/<name>.tex in build/ with fontspec; the source keeps kotex; PDF copied to docs/notes/<name>.pdf)
set -e
N=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT/build"
python3 - "$N" "$ROOT" <<'PY'
import sys
n, root = sys.argv[1], sys.argv[2]
s = open(f'{root}/docs/notes/{n}.tex').read()
k = r'\usepackage{kotex}'
assert s.count(k) == 1
s = s.replace(k, r'\usepackage{fontspec}\setmainfont{Noto Serif CJK KR}[ItalicFont=Noto Serif CJK KR, ItalicFeatures={FakeSlant=0.18}, BoldItalicFont=Noto Serif CJK KR Bold, BoldItalicFeatures={FakeSlant=0.18}]\XeTeXlinebreaklocale ko')
s = s.replace(r'\c{S}', r'{\fontspec{DejaVu Serif}\symbol{"015E}}')  # glyph missing in the CJK font
open(f'{root}/build/{n}.tex', 'w').write(s)
PY
cd "$ROOT/build"
export TEXINPUTS=".:$ROOT/docs/figures:"   # figures of the notes (graphicspath is relative to docs/notes, the build runs in build/)
for i in 1 2 3; do xelatex -interaction=nonstopmode -halt-on-error "$N.tex" > "$N.build.log" 2>&1 || { tail -30 "$N.build.log"; exit 1; }; done
echo "errors: $(grep -c '^!' $N.log)  undefined: $(grep -c 'undefined' $N.log)  multiply-defined: $(grep -c 'multiply' $N.log)  overfull: $(grep -c 'Overfull' $N.log)  missing-char: $(grep -c 'Missing character' $N.log)  pages: $(pdfinfo $N.pdf | grep Pages | awk '{print $2}')"
cp "$N.pdf" "$ROOT/docs/notes/$N.pdf"
