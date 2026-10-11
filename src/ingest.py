#!/usr/bin/env python3
"""swiss-stats F1: Ist-Daten v2 daily CSV -> Parquet (one file per day)."""
import json, sys, time
from pathlib import Path

import polars as pl
import requests
import yaml

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config" / "config.yaml"))
ROOT = Path(CFG["data_root"])
SCRATCH = ROOT / "scratch"
PQ_DIR = ROOT / "parquet" / "istdaten"
MANIFEST = ROOT / "parquet" / "manifest.jsonl"
CKAN = "https://ckan.opendata.swiss/api/3/action/package_show?id=ist-daten-v2"
UA = {"User-Agent": "Mozilla/5.0 (swiss-stats ingest)"}

TS_COLS = ["ANKUNFTSZEIT", "AN_PROGNOSE", "ABFAHRTSZEIT", "AB_PROGNOSE"]
TS_FMTS = ["%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M"]
BOOL_COLS = ["ZUSATZFAHRT_TF", "FAELLT_AUS_TF", "DURCHFAHRT_TF"]
MAX_PARSE_FAIL = 0.001  # >0.1% unparsable timestamps = reject the day


MAX_ABS_DELAY_S = 86_400  # |delay| > 1 day is a broken timestamp (seen: prognosis ~1900), not a delay


def delay_expr(prog, sched, name):
    d = (pl.col(prog) - pl.col(sched)).dt.total_seconds()
    return pl.when(d.abs() <= MAX_ABS_DELAY_S).then(d).otherwise(None).cast(pl.Int32).alias(name)


def parse_ts(col):
    return pl.coalesce([pl.col(col).str.strptime(pl.Datetime("ms"), f, strict=False) for f in TS_FMTS]).alias(col)


def convert(csv_path: Path) -> dict:
    t0 = time.time()
    raw = pl.read_csv(csv_path, separator=";", infer_schema=False)
    if "SLOID" not in raw.columns:              # early v2 (until ~2025-10) had no SLOID
        raw = raw.with_columns(pl.lit(None, dtype=pl.Utf8).alias("SLOID"))
    raw_nonnull = {c: raw[c].is_not_null().sum() for c in TS_COLS}

    df = raw.with_columns(
        pl.col("BETRIEBSTAG").str.strptime(pl.Date, "%d.%m.%Y"),
        pl.col("BPUIC").cast(pl.Int32, strict=False),
        *[(pl.col(c) == "true").alias(c) for c in BOOL_COLS],
        *[parse_ts(c) for c in TS_COLS],
    ).with_columns(
        delay_expr("AN_PROGNOSE", "ANKUNFTSZEIT", "arr_delay_s"),
        delay_expr("AB_PROGNOSE", "ABFAHRTSZEIT", "dep_delay_s"),
        pl.col("BPUIC").cast(pl.Utf8).str.starts_with("85").alias("is_ch"),
        pl.when(pl.col("BPUIC") >= 10_000_000).then(pl.col("BPUIC") // 100).otherwise(pl.col("BPUIC")).cast(pl.Int32).alias("station_uic"),
    )

    fails = {c: 1 - df[c].is_not_null().sum() / max(raw_nonnull[c], 1) for c in TS_COLS}
    worst = max(fails.values())
    if worst > MAX_PARSE_FAIL:
        raise ValueError(f"timestamp parse failure {worst:.4%}: {fails}")

    days = df["BETRIEBSTAG"].unique().to_list()
    day = max(days, key=lambda d: (df["BETRIEBSTAG"] == d).sum())
    PQ_DIR.mkdir(parents=True, exist_ok=True)
    out = PQ_DIR / f"{day}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd", compression_level=6, statistics=True, row_group_size=500_000)
    tmp.rename(out)

    real_arr = df.filter(pl.col("AN_PROGNOSE_STATUS") == "REAL")
    return {
        "day": str(day),
        "rows": df.height,
        "csv_mb": round(csv_path.stat().st_size / 2**20, 1),
        "pq_mb": round(out.stat().st_size / 2**20, 1),
        "sec": round(time.time() - t0, 1),
        "pct_ch": round(100 * df["is_ch"].mean(), 1),
        "pct_real_arr": round(100 * real_arr.height / df.height, 1),
        "pct_cancel": round(100 * df["FAELLT_AUS_TF"].mean(), 2),
        "median_arr_delay_s": real_arr["arr_delay_s"].median(),
        "parse_fail_max": round(worst, 6),
        "absurd_delay": int((df["AN_PROGNOSE"].is_not_null() & df["ANKUNFTSZEIT"].is_not_null() & df["arr_delay_s"].is_null()).sum()
                            + (df["AB_PROGNOSE"].is_not_null() & df["ABFAHRTSZEIT"].is_not_null() & df["dep_delay_s"].is_null()).sum()),
        "other_days_in_file": len(days) - 1,
    }


def list_remote():
    r = requests.get(CKAN, headers=UA, timeout=60)
    r.raise_for_status()
    res = r.json()["result"]["resources"]
    urls = sorted((x["url"] for x in res if x.get("url", "").endswith("_istdaten.csv")), key=lambda u: Path(u).name)
    return [(Path(u).name[:10], u) for u in urls]


def download(url: str, dst: Path):
    part = dst.with_suffix(".part")
    with requests.get(url, headers=UA, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(part, "wb") as f:
            for chunk in r.iter_content(8 << 20):
                f.write(chunk)
    part.rename(dst)


def log(stats: dict):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(stats) + "\n")
    print(json.dumps(stats))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "sample":
        log(convert(SCRATCH / "sample_istdaten.csv"))
        return

    remote = list_remote()
    print(f"remote files: {len(remote)}  {remote[0][0]} .. {remote[-1][0]}")
    for day, url in remote:
        if (PQ_DIR / f"{day}.parquet").exists():
            continue
        csv = SCRATCH / f"{day}_istdaten.csv"
        try:
            download(url, csv)
            log(convert(csv))
        except Exception as e:
            print(f"FAIL {day}: {e}", file=sys.stderr)
        finally:
            csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
