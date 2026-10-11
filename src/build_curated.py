#!/usr/bin/env python3
"""swiss-stats curated layer: raw day Parquet -> slim integer arrivals + append-only dimensions."""
import sys, time
from pathlib import Path
import polars as pl
import yaml

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config" / "config.yaml"))
ROOT = Path(CFG["data_root"])
RAW = ROOT / "parquet" / "istdaten"
CUR = ROOT / "curated"
ARR = CUR / "arrivals"
DIM_PROD = CUR / "dim_product.parquet"
DIM_LINE = CUR / "dim_line.parquet"
DIM_STATION = CUR / "dim_station.parquet"
BATCH = 20                                  # days per batch: RAM independent of history length

PROD = pl.col("PRODUKT_ID").fill_null("unknown").str.to_lowercase().str.to_titlecase().replace({"Bua": "Bus"}).alias("product")
OPER = pl.col("BETREIBER_ABK").fill_null("?").alias("operator")
LINE = pl.col("LINIEN_TEXT").fill_null("?").alias("line_text")


def arrivals(lf):
    return lf.filter(
        pl.col("is_ch")
        & (pl.col("AN_PROGNOSE_STATUS") == "REAL")
        & ~pl.col("FAELLT_AUS_TF")
        & ~pl.col("DURCHFAHRT_TF")
        & pl.col("arr_delay_s").is_not_null()
        & (pl.col("arr_delay_s").abs() <= 86_400)   # |delay| > 1 day = broken timestamp
    )


def update_coded_dim(path, new_vals, keys, code):
    """Append-only dimension: existing codes never change, new keys get next codes."""
    if path.exists():
        old = pl.read_parquet(path)
    else:
        old = pl.DataFrame(schema={**dict(new_vals.schema), code: pl.Int32})
    add = new_vals.join(old.select(keys), on=keys, how="anti").sort(keys)
    add = add.with_row_index(code, offset=old.height).with_columns(pl.col(code).cast(pl.Int32))
    dim = pl.concat([old, add.select(old.columns)])
    dim.write_parquet(path)
    return dim, add.height


def update_station_dim(new_vals):
    if DIM_STATION.exists():
        old = pl.read_parquet(DIM_STATION)
    else:
        old = pl.DataFrame(schema={"station_uic": pl.Int32, "name": pl.Utf8})
    add = new_vals.join(old.select("station_uic"), on="station_uic", how="anti")
    dim = pl.concat([old, add.select(old.columns)])
    dim.write_parquet(DIM_STATION)
    return dim, add.height


def main():
    rebuild = "--rebuild" in sys.argv
    CUR.mkdir(parents=True, exist_ok=True)
    ARR.mkdir(exist_ok=True)
    if rebuild:
        for p in (DIM_PROD, DIM_LINE, DIM_STATION, *ARR.glob("*.parquet")):
            p.unlink()
    days = sorted(RAW.glob("*.parquet"))
    todo = [f for f in days if not (ARR / f.name).exists()]
    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    total = 0
    for b0 in range(0, len(todo), BATCH):
        total += curate_batch(todo[b0:b0 + BATCH])
    print(f"\ncurated {len(todo)} day(s), {total:,} rows, {time.time() - t0:.1f}s")


def curate_batch(todo):
    src = arrivals(pl.scan_parquet([str(f) for f in todo]))
    prods = src.select(PROD).unique().collect(engine="streaming")
    lines = src.select(OPER, LINE).unique().collect(engine="streaming")
    stations = (src.group_by("station_uic", "HALTESTELLEN_NAME").len()
                .sort("len", descending=True).unique("station_uic", keep="first")
                .select(pl.col("station_uic").cast(pl.Int32), pl.col("HALTESTELLEN_NAME").alias("name"))
                .collect(engine="streaming"))
    dprod, n_p = update_coded_dim(DIM_PROD, prods, ["product"], "pcode")
    dline, n_l = update_coded_dim(DIM_LINE, lines, ["operator", "line_text"], "lcode")
    dst, n_s = update_station_dim(stations)
    assert dprod.height <= 16, dprod
    print(f"dims: products {dprod.height} (+{n_p}) | lines {dline.height} (+{n_l}) | "
          f"stations {dst.height} (+{n_s})")

    total = 0
    for f in todo:
        df = (arrivals(pl.scan_parquet(f))
              .select(pl.col("BETRIEBSTAG").alias("date"),
                      pl.col("station_uic").cast(pl.Int32),
                      PROD, OPER, LINE,
                      pl.col("ANKUNFTSZEIT").dt.hour().cast(pl.Int8).alias("hour"),
                      (pl.col("BETRIEBSTAG").dt.weekday() - 1).cast(pl.Int8).alias("wd"),   # 0 = Monday
                      pl.col("arr_delay_s").cast(pl.Int32).alias("delay"))
              .join(dprod.lazy(), on="product", how="left")
              .join(dline.lazy(), on=["operator", "line_text"], how="left")
              .select("date", "station_uic", pl.col("pcode").cast(pl.Int8), "lcode", "hour", "wd", "delay")
              .sort("station_uic")
              .collect())
        assert df["pcode"].null_count() == 0 and df["lcode"].null_count() == 0, f"unmapped codes in {f.name}"
        tmp = (ARR / f.name).with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True, row_group_size=500_000)
        tmp.rename(ARR / f.name)
        total += df.height
        print(f"{f.name}  rows {df.height:,}")
    return total


if __name__ == "__main__":
    main()
