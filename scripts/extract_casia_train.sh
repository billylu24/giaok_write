#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# The public mirror wraps the original training ALZ in a ZIP.
python3 - <<'PY'
from pathlib import Path
import zipfile
src=Path('data/casia/raw/HWDB1.1trn_gnt.zip')
with zipfile.ZipFile(src) as z:
    if z.namelist()!=['HWDB1.1trn_gnt.alz']:
        raise ValueError('Unexpected training wrapper format')
    dst=src.parent/'HWDB1.1trn_gnt.alz'
    if not dst.exists():z.extract('HWDB1.1trn_gnt.alz',src.parent)
PY
mkdir -p vendor/tools data/casia/raw/train_gnt
if [ ! -d vendor/tools/unalz ]; then
  git clone https://github.com/kippler/unalz.git vendor/tools/unalz
fi
git -C vendor/tools/unalz checkout 457884d21962caa12085bfb6ec5bd12f3eb93c00
(
 cd vendor/tools/unalz/unalz-0.65
 gcc -c UnAlzBz2decompress.c UnAlzbzlib.c -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64
 g++ -std=c++11 -D_UNALZ_ICONV -D_UNALZ_UTF8 -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 \
  main.cpp UnAlz.cpp UnAlzUtils.cpp UnAlzBzip2.cpp UnAlzBz2decompress.o UnAlzbzlib.o \
  -lz -lbz2 -o unalz
)
vendor/tools/unalz/unalz-0.65/unalz -d data/casia/raw/train_gnt data/casia/raw/HWDB1.1trn_gnt.alz
python3 - <<'PY'
from pathlib import Path
files=list(Path('data/casia/raw/train_gnt').rglob('*.gnt'))
if len(files)!=240:raise ValueError(f'Expected 240 training writer files, got {len(files)}')
print('Training archive: 240 writer GNT files extracted')
PY
