#!/usr/bin/env python3
"""Bring every raw day Parquet to one canonical schema (the newest file's).
Missing columns -> typed nulls, column order -> canonical. Extra columns or type conflicts -> refuse."""
import sys
from pathlib import Path
import polars as pl

PQ_DIR = Path("/data/swiss-stats/parquet/istdaten")
files = sorted(PQ_DIR.glob("*.parquet"))
canon = pl.read_parquet_schema(files[-1])
fixed = refused = 0
for f in files:
    sch = pl.read_parquet_schema(f)
    if list(sch.items()) == list(canon.items()):
        continue
    extra = [c for c in sch if c not in canon]
    conflict = [c for c in sch if c in canon and sch[c] != canon[c]]
    if extra or conflict:
        print(f"REFUSE {f.name}: extra {extra}, type conflict {conflict}", file=sys.stderr)
        refused += 1
        continue
    df = pl.read_parquet(f)
    df = df.with_columns([pl.lit(None, dtype=t).alias(c) for c, t in canon.items() if c not in sch]).select(list(canon))
    tmp = f.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True, row_group_size=500_000)
    tmp.rename(f)
    fixed += 1
print(f"files {len(files)} | normalized {fixed} | refused {refused}")
sys.exit(1 if refused else 0)
