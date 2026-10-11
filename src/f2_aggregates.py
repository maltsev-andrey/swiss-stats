#!/usr/bin/env python3
"""swiss-stats F2.1: delay statistics on a SHARED GPU.
Sort engine: station-local groups, chunked by station (exact).
Histogram engine: global groups (product/weekday/hour, lines), int delays -> mergeable histograms (exact).
"""
import math, os, sys, time
from pathlib import Path
import yaml

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config" / "config.yaml"))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = CFG["gpu_uuid"]

import cupy as cp
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GPU = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
assert CFG["expected_gpu"] in GPU, f"WRONG GPU: {GPU}"

ROOT = Path(CFG["data_root"])
PQ = str(ROOT / "parquet" / "istdaten" / "*.parquet")
AGG = ROOT / "agg"; AGG.mkdir(exist_ok=True)
OUT = ROOT / "out"; OUT.mkdir(exist_ok=True)

D_MIN, D_MAX, D_BITS = -1800, 7200, 14     # clip window [s]; 2^14 > D_MAX - D_MIN
NB = D_MAX - D_MIN + 1                     # histogram bins, 1 s each
QS = (0.5, 0.9, 0.95)
ONTIME_S = 180                             # late <= 3 min
EARLY_S = -60                              # early by more than 1 min
SLOTS = 16 * 256                           # pcode*256 + wd*32 + hour
BYTES_PER_ROW = 112                        # conservative sort-engine peak per row
RESERVE = 1 << 30                          # always leave 1 GB to neighbours (Ollama)
MIN_BUDGET = 1 << 30
WAIT_MAX_S = 1800
RAIL_HUB_N = 5_000                         # >= 100 train arrivals/day
BIG_LINE_N = 30_000                        # >= 600 arrivals/day
pool = cp.get_default_memory_pool()


def sync():
    cp.cuda.Stream.null.synchronize()


def acquire_budget():
    deadline = time.time() + WAIT_MAX_S
    while True:
        free, _ = cp.cuda.runtime.memGetInfo()
        budget = int(min(CFG["vram_budget_gb"] * 2**30, free - RESERVE))
        if budget >= MIN_BUDGET:
            pool.set_limit(size=budget)
            return budget, free
        if time.time() > deadline:
            print(f"VRAM not available (free {free / 2**30:.2f} GB), exit 75", file=sys.stderr)
            sys.exit(75)
        print(f"waiting for VRAM: free {free / 2**30:.2f} GB, need {(MIN_BUDGET + RESERVE) / 2**30:.1f} GB")
        time.sleep(30)


def sort_stats(keys, delay):
    """keys int64, delay int32 (GPU) -> exact per-group stats via one packed sort."""
    uniq, gid = cp.unique(keys, return_inverse=True)
    packed = (gid.astype(cp.int64) << D_BITS) | (delay - D_MIN).astype(cp.int64)
    del gid
    packed.sort()
    g = packed >> D_BITS
    v = (packed & ((1 << D_BITS) - 1)).astype(cp.int32) + D_MIN
    del packed
    G = int(uniq.size)
    n = cp.bincount(g, minlength=G).astype(cp.int64)
    end = cp.cumsum(n)
    start = end - n
    zero = cp.zeros(1, cp.int64)
    seg = lambda x: (lambda c: c[end] - c[start])(cp.concatenate([zero, cp.cumsum(x.astype(cp.int64))]))
    res = {"key": uniq, "n": n,
           "mean_s": seg(v) / n,
           "ontime_pct": 100.0 * seg(v <= ONTIME_S) / n,
           "early_pct": 100.0 * seg(v < EARLY_S) / n}
    for q in QS:
        res[f"p{int(q * 100)}_s"] = v[start + cp.floor(q * (n - 1)).astype(cp.int64)]
    return pl.DataFrame({k: cp.asnumpy(a) for k, a in res.items()})


def hist_stats(hist):
    """hist (S, NB) int64 on GPU -> exact stats for non-empty rows."""
    n_all = hist.sum(axis=1)
    keep = cp.nonzero(n_all)[0]
    h, n = hist[keep], n_all[keep]
    vals = cp.arange(D_MIN, D_MAX + 1, dtype=cp.int64)
    c = cp.cumsum(h, axis=1)
    res = {"key": keep, "n": n,
           "mean_s": (h * vals).sum(axis=1) / n,
           "ontime_pct": 100.0 * h[:, : ONTIME_S - D_MIN + 1].sum(axis=1) / n,
           "early_pct": 100.0 * h[:, : EARLY_S - D_MIN].sum(axis=1) / n}
    for q in QS:
        idx = cp.floor(q * (n - 1)).astype(cp.int64)
        res[f"p{int(q * 100)}_s"] = (c <= idx[:, None]).sum(axis=1) + D_MIN   # first bin with cum > idx
    return pl.DataFrame({k: cp.asnumpy(a) for k, a in res.items()})


def warmup():
    k = cp.arange(4096, dtype=cp.int64) % 13
    d = (cp.arange(4096, dtype=cp.int32) % 600) - 100
    sort_stats(k, d)
    h = cp.zeros((4, NB), cp.int64); h[:, 2000] = 3
    hist_stats(h)
    cp.bincount(k * NB + (d - D_MIN).astype(cp.int64), minlength=13 * NB)
    sync(); pool.free_all_blocks()


def order_by_partition(a, parts):
    """Reorder host arrays ONCE so every partition is a contiguous slice (radix argsort on uint8/16)."""
    pid = (a["st"] % parts).astype(np.uint8 if parts < 256 else np.uint16)
    order = np.argsort(pid, kind="stable")
    counts = np.bincount(pid, minlength=parts)
    del pid
    for k in list(a):
        a[k] = a[k][order]
    return np.concatenate([[0], np.cumsum(counts)])


def run_partitions(a, n_lines, parts):
    hist_pwh = cp.zeros(SLOTS * NB, dtype=cp.int64)
    hist_ln = cp.zeros(n_lines * NB, dtype=cp.int64)
    sph, s, sp, peak = [], [], [], 0
    bounds = order_by_partition(a, parts)
    for p in range(parts):
        lo, hi = int(bounds[p]), int(bounds[p + 1])
        if lo == hi:
            continue
        g = {k: cp.asarray(v[lo:hi]).astype(cp.int64) for k, v in a.items() if k != "dl"}
        dl = cp.asarray(a["dl"][lo:hi]).astype(cp.int32)
        off = (dl - D_MIN).astype(cp.int64)
        hist_pwh += cp.bincount((g["pc"] * 256 + g["wd"] * 32 + g["hr"]) * NB + off, minlength=SLOTS * NB)
        hist_ln += cp.bincount(g["ln"] * NB + off, minlength=n_lines * NB)
        del off
        sph.append(sort_stats(g["st"] * 512 + g["pc"] * 32 + g["hr"], dl))
        sp.append(sort_stats(g["st"] * 16 + g["pc"], dl))
        s.append(sort_stats(g["st"], dl))
        del g, dl
        peak = max(peak, pool.total_bytes())
        pool.free_all_blocks()
    pwh = hist_stats(hist_pwh.reshape(SLOTS, NB))
    prod = hist_stats(hist_pwh.reshape(16, 256, NB).sum(axis=1))
    lines = hist_stats(hist_ln.reshape(n_lines, NB))
    del hist_pwh, hist_ln
    pool.free_all_blocks()
    return pl.concat(sph), pl.concat(s), pl.concat(sp), pwh, prod, lines, peak


def run_gpu(a, n_lines, budget):
    hist_bytes = 2 * (SLOTS + n_lines) * NB * 8
    if hist_bytes > budget * 0.7:
        sys.exit(f"histograms need {hist_bytes / 2**30:.2f} GB, budget {budget / 2**30:.2f} GB: too small")
    rows_per_chunk = max(1_000_000, (budget - hist_bytes) // BYTES_PER_ROW)
    parts = max(1, math.ceil(1.2 * len(a["dl"]) / rows_per_chunk))
    while True:
        try:
            print(f"GPU run: {parts} partition(s), ~{len(a['dl']) // parts:,} rows each")
            return (*run_partitions(a, n_lines, parts), parts)
        except cp.cuda.memory.OutOfMemoryError:
            pool.free_all_blocks()
            parts *= 2
            print(f"OOM -> retry with {parts} partitions")


def main():
    t0 = time.time()
    scan = pl.scan_parquet(PQ)
    prod_expr = pl.col("PRODUKT_ID").fill_null("unknown").str.to_lowercase().str.to_titlecase()
    line_expr = pl.concat_str([pl.col("BETREIBER_ABK").fill_null("?"), pl.col("LINIEN_TEXT").fill_null("?")],
                              separator=" | ")
    base = scan.filter(
        pl.col("is_ch")
        & (pl.col("AN_PROGNOSE_STATUS") == "REAL")
        & ~pl.col("FAELLT_AUS_TF")
        & ~pl.col("DURCHFAHRT_TF")
        & pl.col("arr_delay_s").is_not_null()
    )
    products = sorted(base.select(prod_expr.unique()).collect().to_series().to_list())
    lines = sorted(base.select(line_expr.unique()).collect().to_series().to_list())
    assert len(products) <= 16, products
    assert len(lines) <= 30_000, len(lines)
    n_clip = base.filter((pl.col("arr_delay_s") < D_MIN) | (pl.col("arr_delay_s") > D_MAX)).select(pl.len()).collect().item()
    df = base.select(
        pl.col("station_uic").cast(pl.Int32),
        prod_expr.replace_strict(products, list(range(len(products))), return_dtype=pl.Int8).alias("pcode"),
        line_expr.replace_strict(lines, list(range(len(lines))), return_dtype=pl.Int32).alias("lcode"),
        pl.col("ANKUNFTSZEIT").dt.hour().cast(pl.Int8).alias("hour"),
        (pl.col("BETRIEBSTAG").dt.weekday() - 1).cast(pl.Int8).alias("wd"),      # 0 = Monday
        pl.col("arr_delay_s").clip(D_MIN, D_MAX).cast(pl.Int16).alias("delay"),
    ).collect(engine="streaming")
    names = (scan.filter(pl.col("is_ch"))
             .group_by("station_uic", "HALTESTELLEN_NAME").len()
             .sort("len", descending=True).unique("station_uic", keep="first")
             .select("station_uic", pl.col("HALTESTELLEN_NAME").alias("name")).collect())
    n_all = df.height
    t_load = time.time() - t0

    budget, free = acquire_budget()
    print(f"\nGPU: {GPU} | free {free / 2**30:.2f} GB | budget {budget / 2**30:.2f} GB")
    print(f"loaded {n_all:,} REAL CH arrivals | {df['station_uic'].n_unique():,} stations | "
          f"{len(lines):,} lines | products {products} | clipped {n_clip:,} ({100 * n_clip / n_all:.3f}%) | {t_load:.1f}s")

    a = {"st": df["station_uic"].to_numpy(), "pc": df["pcode"].to_numpy(), "hr": df["hour"].to_numpy(),
         "wd": df["wd"].to_numpy(), "ln": df["lcode"].to_numpy(), "dl": df["delay"].to_numpy()}
    t = time.time(); warmup(); t_jit = time.time() - t
    sync(); t = time.time()
    sph, s, sp, pwh, prod, ln, peak, parts = run_gpu(a, len(lines), budget)
    sync(); t_gpu = time.time() - t

    # CPU reference: same metrics, biggest grouping
    t = time.time()
    ref = (df.select((pl.col("station_uic").cast(pl.Int64) * 512 + pl.col("pcode").cast(pl.Int64) * 32
                      + pl.col("hour").cast(pl.Int64)).alias("key"), pl.col("delay"))
           .group_by("key")
           .agg(pl.len().alias("n_cpu"),
                pl.col("delay").mean().alias("mean_cpu"),
                (pl.col("delay") <= ONTIME_S).mean().alias("ontime_cpu"),
                (pl.col("delay") < EARLY_S).mean().alias("early_cpu"),
                *[pl.col("delay").quantile(q, "lower").alias(f"p{int(q * 100)}_cpu") for q in QS]))
    t_cpu = time.time() - t
    mism = (sph.join(ref, on="key")
            .filter((pl.col("n") != pl.col("n_cpu")) | (pl.col("p50_s") != pl.col("p50_cpu"))
                    | (pl.col("p95_s") != pl.col("p95_cpu"))).height)

    # decode + save
    pmap = pl.DataFrame({"pcode": list(range(len(products))), "product": products},
                        schema={"pcode": pl.Int64, "product": pl.Utf8})
    lmap = pl.DataFrame({"key": list(range(len(lines))), "line": lines}, schema={"key": pl.Int64, "line": pl.Utf8})
    sph = (sph.with_columns((pl.col("key") // 512).cast(pl.Int32).alias("station_uic"),
                            ((pl.col("key") // 32) % 16).alias("pcode"), (pl.col("key") % 32).alias("hour"))
              .join(pmap, on="pcode").join(names, on="station_uic", how="left"))
    sp = (sp.with_columns((pl.col("key") // 16).cast(pl.Int32).alias("station_uic"), (pl.col("key") % 16).alias("pcode"))
            .join(pmap, on="pcode").join(names, on="station_uic", how="left"))
    s = s.with_columns(pl.col("key").cast(pl.Int32).alias("station_uic")).join(names, on="station_uic", how="left")
    pwh = (pwh.with_columns((pl.col("key") // 256).alias("pcode"), ((pl.col("key") // 32) % 8).alias("wd"),
                            (pl.col("key") % 32).alias("hour")).join(pmap, on="pcode"))
    prod = prod.with_columns(pl.col("key").alias("pcode")).join(pmap, on="pcode")
    ln = ln.join(lmap, on="key")
    for d, f in ((sph, "station_product_hour"), (sp, "station_product"), (s, "station"),
                 (pwh, "product_weekday_hour"), (prod, "product"), (ln, "line")):
        d.write_parquet(AGG / f"{f}.parquet")

    cols = ["n", "ontime_pct", "early_pct", "mean_s", "p50_s", "p90_s", "p95_s"]
    r1 = lambda d: d.with_columns(pl.col("ontime_pct", "early_pct", "mean_s").round(1))
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=36, thousands_separator=True, tbl_hide_dataframe_shape=True):
        print("\n=== Timing ===")
        print(pl.DataFrame([("load (Polars streaming, CPU)", t_load),
                            ("CuPy JIT warm-up (one-off)", t_jit),
                            ("GPU: 6 groupings, all metrics, incl. H2D", t_gpu),
                            ("CPU Polars: 1 grouping, same metrics", t_cpu)],
                           schema=["step", "sec"], orient="row").with_columns(pl.col("sec").round(2)))
        print(f"partitions {parts} | VRAM pool peak {peak / 2**30:.2f} GB | groups sph {sph.height:,} | "
              f"GPU vs CPU mismatches (n, p50, p95): {mism}\n")
        print("=== By product ===")
        print(r1(prod.select(["product"] + cols).sort("n", descending=True)))
        hubs = sp.filter((pl.col("product") == "Zug") & (pl.col("n") >= RAIL_HUB_N))
        print(f"\n=== Rail hubs, trains only: worst 10 by on-time (n >= {RAIL_HUB_N:,}, {hubs.height} hubs) ===")
        print(r1(hubs.select(["name"] + cols).sort("ontime_pct").head(10)))
        print("\n=== Rail hubs, trains only: best 10 by on-time ===")
        print(r1(hubs.select(["name"] + cols).sort("ontime_pct", descending=True).head(10)))
        big = ln.filter(pl.col("n") >= BIG_LINE_N)
        print(f"\n=== Lines: worst 15 by p95 (n >= {BIG_LINE_N:,}, {big.height} lines) ===")
        print(r1(big.select(["line"] + cols).sort("p95_s", descending=True).head(15)))
        print("\n=== Lines: best 10 by on-time ===")
        print(r1(big.select(["line"] + cols).sort("ontime_pct", descending=True).head(10)))

    z = pwh.filter((pl.col("product") == "Zug") & (pl.col("n") >= 500) & (pl.col("hour") < 24))
    arr = np.full((7, 24), np.nan)
    for row in z.iter_rows(named=True):
        arr[row["wd"], row["hour"]] = row["p95_s"] / 60
    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap="magma_r")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(24))
    ax.set_xlabel("scheduled arrival hour")
    ax.set_title(f"Swiss trains: p95 arrival delay [min], {n_all / 1e6:.0f}M arrivals, 50 days (Source: opentransportdata.swiss)")
    fig.colorbar(im, ax=ax, label="p95 delay, min")
    fig.tight_layout()
    fig.savefig(OUT / "f2_zug_p95_heatmap.png", dpi=130)
    print(f"\nheatmap -> {OUT / 'f2_zug_p95_heatmap.png'}")


if __name__ == "__main__":
    main()
