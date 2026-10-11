#!/usr/bin/env python3
"""swiss-stats backfill: monthly v2 archive zips -> day Parquet via ingest.convert().
Days already ingested by the daily collector are NOT overwritten: the archive version goes
to overlap_check/ and is compared with the daily one.
"""
import calendar, shutil, sys, time, zipfile
from datetime import date
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ingest

BASE = "https://archive.opentransportdata.swiss/istdaten"
MONTHS = [(2025, m) for m in range(7, 13)] + [(2026, m) for m in range(1, 10)]   # v2 archive: 2025-07 .. 2026-09
WORK = ingest.SCRATCH / "backfill"
CHECK = ingest.ROOT / "parquet" / "overlap_check"
PQ_DIR = ingest.PQ_DIR


def summary(path):
    df = pl.read_parquet(path, columns=["AN_PROGNOSE_STATUS", "arr_delay_s", "FAELLT_AUS_TF"])
    real = df.filter(pl.col("AN_PROGNOSE_STATUS") == "REAL")
    return {"rows": df.height, "real": real.height,
            "sum_delay": int(real["arr_delay_s"].sum() or 0), "cancel": int(df["FAELLT_AUS_TF"].sum())}


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    CHECK.mkdir(parents=True, exist_ok=True)
    overlap, t_all = [], time.time()
    for y, m in MONTHS:
        days = [date(y, m, d) for d in range(1, calendar.monthrange(y, m)[1] + 1)]
        missing = [d for d in days if not (PQ_DIR / f"{d}.parquet").exists()]
        to_check = [d for d in days if d not in missing and not (CHECK / f"{d}.parquet").exists()]
        if not missing and not to_check:
            print(f"{y}-{m:02d}: complete, skip")
            continue
        name = f"ist-daten-v2-{y}-{m:02d}.zip"
        zpath = WORK / name
        t = time.time()
        if not zpath.exists():
            ingest.download(f"{BASE}/{y}/{name}", zpath)
        t_dl = time.time() - t
        ok = fail = 0
        with zipfile.ZipFile(zpath) as z:
            entries = {Path(n).name[:10]: n for n in z.namelist() if n.lower().endswith(".csv")}
            for d in missing + to_check:
                key = str(d)
                if key not in entries:
                    print(f"  {key}: not in zip", file=sys.stderr)
                    fail += 1
                    continue
                csv = WORK / f"{key}.csv"
                try:
                    with z.open(entries[key]) as src, open(csv, "wb") as dst:
                        shutil.copyfileobj(src, dst, 16 << 20)
                    if d in missing:
                        ingest.PQ_DIR = PQ_DIR
                        stats = ingest.convert(csv)
                        stats["source"] = "archive"
                        ingest.log(stats)
                    else:
                        ingest.PQ_DIR = CHECK
                        ingest.convert(csv)
                        a, b = summary(CHECK / f"{key}.parquet"), summary(PQ_DIR / f"{key}.parquet")
                        overlap.append({"day": key, **{f"arch_{k}": v for k, v in a.items()},
                                        **{f"daily_{k}": v for k, v in b.items()}})
                    ok += 1
                except Exception as e:
                    print(f"  FAIL {key}: {e}", file=sys.stderr)
                    fail += 1
                finally:
                    csv.unlink(missing_ok=True)
                    ingest.PQ_DIR = PQ_DIR
        zpath.unlink()
        print(f"{y}-{m:02d}: ok {ok}, fail {fail} | download {t_dl:.0f}s, total {time.time() - t:.0f}s", flush=True)

    if overlap:
        ov = pl.DataFrame(overlap).with_columns(
            (pl.col("arch_rows") - pl.col("daily_rows")).alias("d_rows"),
            (pl.col("arch_real") - pl.col("daily_real")).alias("d_real"),
            (pl.col("arch_sum_delay") / pl.col("arch_real") - pl.col("daily_sum_delay") / pl.col("daily_real"))
              .round(2).alias("d_mean_delay_s"),
            (pl.col("arch_cancel") - pl.col("daily_cancel")).alias("d_cancel"))
        ov.write_parquet(CHECK / "overlap_report.parquet")
        with pl.Config(tbl_rows=60, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
            print("\n=== Overlap: archive vs daily ===")
            print(ov.select("day", "daily_rows", "d_rows", "d_real", "d_mean_delay_s", "d_cancel"))
    print(f"\nbackfill done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
