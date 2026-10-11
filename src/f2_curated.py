#!/usr/bin/env python3
"""swiss-stats F2.2: F2 GPU engines on the curated layer + window KPI + rail hubs by volume."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f2_aggregates as eng            # CUDA env, P100 guard, engines, constants
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

CUR = eng.ROOT / "curated"
D_MIN, D_MAX = eng.D_MIN, eng.D_MAX
RAIL_HUB_N, BIG_LINE_N = 5_000, 30_000


def main():
    t0 = time.time()
    df = pl.read_parquet(str(CUR / "arrivals" / "*.parquet"))
    n_days = df["date"].n_unique()
    dprod = pl.read_parquet(CUR / "dim_product.parquet")
    dline = pl.read_parquet(CUR / "dim_line.parquet")
    names = pl.read_parquet(CUR / "dim_station.parquet")
    n_all = df.height
    n_clip = df.filter((pl.col("delay") < D_MIN) | (pl.col("delay") > D_MAX)).height
    df = df.with_columns(pl.col("delay").clip(D_MIN, D_MAX).cast(pl.Int16))
    a = {"st": df["station_uic"].to_numpy(), "pc": df["pcode"].to_numpy(), "hr": df["hour"].to_numpy(),
         "wd": df["wd"].to_numpy(), "ln": df["lcode"].to_numpy(), "dl": df["delay"].to_numpy()}
    t_load = time.time() - t0

    budget, free = eng.acquire_budget()
    print(f"\nGPU: {eng.GPU} | free {free / 2**30:.2f} GB | budget {budget / 2**30:.2f} GB")
    print(f"curated: {n_all:,} arrivals | {names.height:,} stations | {dline.height:,} lines | "
          f"{dprod.height} products | clipped {n_clip:,} ({100 * n_clip / n_all:.3f}%) | load {t_load:.2f}s")

    t = time.time(); eng.warmup(); t_jit = time.time() - t
    eng.sync(); t = time.time()
    sph, s, sp, pwh, prod, ln, peak, parts = eng.run_gpu(a, dline.height, budget)
    eng.sync(); t_gpu = time.time() - t

    ref_mod = 1 if n_all < 200_000_000 else 10
    ref_src = df if ref_mod == 1 else df.filter(pl.col("station_uic") % ref_mod == 0)
    t = time.time()
    ref = (ref_src.select((pl.col("station_uic").cast(pl.Int64) * 512 + pl.col("pcode").cast(pl.Int64) * 32
                      + pl.col("hour").cast(pl.Int64)).alias("key"), pl.col("delay"))
           .group_by("key")
           .agg(pl.len().alias("n_cpu"),
                pl.col("delay").mean().alias("mean_cpu"),
                (pl.col("delay") <= eng.ONTIME_S).mean().alias("ontime_cpu"),
                (pl.col("delay") < eng.EARLY_S).mean().alias("early_cpu"),
                *[pl.col("delay").quantile(q, "lower").alias(f"p{int(q * 100)}_cpu") for q in eng.QS]))
    t_cpu = time.time() - t
    mism = (sph.join(ref, on="key")
            .filter((pl.col("n") != pl.col("n_cpu")) | (pl.col("p50_s") != pl.col("p50_cpu"))
                    | (pl.col("p95_s") != pl.col("p95_cpu"))).height)

    pmap = dprod.select(pl.col("pcode").cast(pl.Int64), "product")
    lmap = dline.select(pl.col("lcode").cast(pl.Int64).alias("key"),
                        pl.concat_str(["operator", "line_text"], separator=" | ").alias("line"))
    win = (pl.col("ontime_pct") - pl.col("early_pct")).alias("window_pct")
    sph = (sph.with_columns((pl.col("key") // 512).cast(pl.Int32).alias("station_uic"),
                            ((pl.col("key") // 32) % 16).alias("pcode"), (pl.col("key") % 32).alias("hour"), win)
              .join(pmap, on="pcode").join(names, on="station_uic", how="left"))
    sp = (sp.with_columns((pl.col("key") // 16).cast(pl.Int32).alias("station_uic"),
                          (pl.col("key") % 16).alias("pcode"), win)
            .join(pmap, on="pcode").join(names, on="station_uic", how="left"))
    s = s.with_columns(pl.col("key").cast(pl.Int32).alias("station_uic"), win).join(names, on="station_uic", how="left")
    pwh = (pwh.with_columns((pl.col("key") // 256).alias("pcode"), ((pl.col("key") // 32) % 8).alias("wd"),
                            (pl.col("key") % 32).alias("hour"), win).join(pmap, on="pcode"))
    prod = prod.with_columns(pl.col("key").alias("pcode"), win).join(pmap, on="pcode")
    ln = ln.with_columns(win).join(lmap, on="key")
    for d, f in ((sph, "station_product_hour"), (sp, "station_product"), (s, "station"),
                 (pwh, "product_weekday_hour"), (prod, "product"), (ln, "line")):
        d.write_parquet(eng.AGG / f"{f}.parquet")

    cols = ["n", "window_pct", "ontime_pct", "early_pct", "mean_s", "p50_s", "p95_s"]
    r1 = lambda d: d.with_columns(pl.col("window_pct", "ontime_pct", "early_pct", "mean_s").round(1))
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=36, thousands_separator=True,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print("\n=== Timing ===")
        print(pl.DataFrame([("load curated (CPU)", t_load), ("CuPy JIT warm-up (one-off)", t_jit),
                            ("GPU: 6 groupings, all metrics", t_gpu), (f"CPU Polars: 1 grouping, same metrics{'' if ref_mod == 1 else ' (10% stations, scaled)'}",
                             t_cpu * n_all / ref_src.height)],
                           schema=["step", "sec"], orient="row").with_columns(pl.col("sec").round(2)))
        print(f"partitions {parts} | VRAM pool peak {peak / 2**30:.2f} GB | groups sph {sph.height:,} | "
              f"GPU vs CPU mismatches (n, p50, p95): {mism}\n")
        print("=== By product ===")
        print(r1(prod.select(["product"] + cols).sort("n", descending=True)))
        hubs = sp.filter(pl.col("product") == "Zug")
        print("\n=== Rail hubs: top 15 by train arrivals ===")
        print(r1(hubs.select(["name"] + cols).sort("n", descending=True).head(15)))
        stops = hubs.filter(pl.col("n") >= RAIL_HUB_N)
        print(f"\n=== Rail stops: worst 10 by window [-1,+3] (n >= {RAIL_HUB_N:,}, {stops.height} stops) ===")
        print(r1(stops.select(["name"] + cols).sort("window_pct").head(10)))
        big = ln.filter(pl.col("n") >= BIG_LINE_N)
        print(f"\n=== Lines: worst 15 by p95 (n >= {BIG_LINE_N:,}, {big.height} lines) ===")
        print(r1(big.select(["line"] + cols).sort("p95_s", descending=True).head(15)))
        print("\n=== Lines: best 10 by window [-1,+3] ===")
        print(r1(big.select(["line"] + cols).sort("window_pct", descending=True).head(10)))

    z = pwh.filter((pl.col("product") == "Zug") & (pl.col("n") >= 500) & (pl.col("hour") < 24))
    arr = np.full((7, 24), np.nan)
    for row in z.iter_rows(named=True):
        arr[row["wd"], row["hour"]] = row["p95_s"] / 60
    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap="magma_r")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(24))
    ax.set_xlabel("scheduled arrival hour")
    ax.set_title(f"Swiss trains: p95 arrival delay [min], {z['n'].sum() / 1e6:.2f}M train arrivals, {n_days} days (Source: opentransportdata.swiss)")
    fig.colorbar(im, ax=ax, label="p95 delay, min")
    fig.tight_layout()
    fig.savefig(eng.OUT / "f2_zug_p95_heatmap.png", dpi=130)
    print(f"\nheatmap -> {eng.OUT / 'f2_zug_p95_heatmap.png'}")


if __name__ == "__main__":
    main()
