#!/usr/bin/env python3
"""Recompute is_ch + station_uic in all day files (BPUIC 7 or 9 digits)."""
from pathlib import Path
import polars as pl

PQ_DIR = Path("/data/swiss-stats/parquet/istdaten")
for f in sorted(PQ_DIR.glob("*.parquet")):
    df = pl.read_parquet(f).with_columns(
        pl.col("BPUIC").cast(pl.Utf8).str.starts_with("85").alias("is_ch"),
        pl.when(pl.col("BPUIC") >= 10_000_000).then(pl.col("BPUIC") // 100)
          .otherwise(pl.col("BPUIC")).cast(pl.Int32).alias("station_uic"),
    )
    tmp = f.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True, row_group_size=500_000)
    tmp.rename(f)
    print(f.name, f"is_ch={100*df['is_ch'].mean():.1f}%", f"stations={df['station_uic'].n_unique()}")
