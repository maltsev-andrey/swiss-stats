#!/usr/bin/env python3
"""swiss-stats F4: Aug-Sep 2025 vs 2026, DAY-BLOCK bootstrap CIs on GPU.
Arrivals on the same day are correlated (weather, incidents, knock-on delays), so an iid bootstrap
over arrivals is overconfident. We resample whole DAYS: per period, per-day histograms H[slot, day, bin];
replicate weights W ~ Multinomial(D) per row, replicate histograms = W @ H (cuBLAS SGEMM, all B at once),
quantiles from cumsum. Periods aligned by 364 days (same weekdays), Aug 1 (national holiday) excluded.
"""
import sys, time
from datetime import date, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f2_aggregates as eng            # CUDA env, P100 guard, budget, constants
from f3_bootstrap import KERNEL, hist_boot_ci
import cupy as cp
import cupyx
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

CUR = eng.ROOT / "curated"
D_MIN, D_MAX, NB = eng.D_MIN, eng.D_MAX, eng.NB
ONT_BIN = eng.ONTIME_S - D_MIN
Q, B, SEED = 0.95, 1000, 20261010
P2_START, P2_END = date(2026, 8, 2), date(2026, 9, 30)
SHIFT = timedelta(days=364)
N_HUBS, N_LINES = 25, 150
MIN_EFFECT_S = 10
N_FLOOR = 200
I_LO, I_HI = int(0.025 * B), int(0.975 * B) - 1
pool = eng.pool
sync = eng.sync


def day_hist(days, ctx):
    S, D = ctx["S"], len(days)
    H = cp.zeros(S * D * NB, dtype=cp.int32)
    for di, d in enumerate(days):
        df = pl.read_parquet(CUR / "arrivals" / f"{d}.parquet", columns=["station_uic", "pcode", "lcode", "hour", "delay"])
        st = df["station_uic"].to_numpy()
        pc = df["pcode"].to_numpy().astype(np.int32)
        hr = df["hour"].to_numpy().astype(np.int32)
        b = (np.clip(df["delay"].to_numpy(), D_MIN, D_MAX) - D_MIN).astype(np.int64)
        hs, hslot = ctx["hub_sorted"], ctx["hub_slot_sorted"]
        pos = np.minimum(np.searchsorted(hs, st), len(hs) - 1)
        is_zug = pc == ctx["zug"]
        hub = np.where((hs[pos] == st) & is_zug, hslot[pos], -1)
        hrs = np.where(is_zug, ctx["off_hour"] + hr, -1)
        lns = ctx["line_slot"][df["lcode"].to_numpy()]
        slots = np.concatenate([pc, hrs, hub, lns])
        bins = np.concatenate([b, b, b, b])
        keep = slots >= 0
        idx = (slots[keep].astype(np.int64) * D + di) * NB + bins[keep]
        cupyx.scatter_add(H, cp.asarray(idx), 1)
    return H.reshape(S, D, NB)


def day_weights(R, D, seed):
    rs = cp.random.RandomState(seed)
    idx = rs.randint(0, D, size=(R, D))
    W = cp.zeros((R, D), cp.float32)
    cupyx.scatter_add(W, (cp.repeat(cp.arange(R), D), idx.ravel()), 1.0)
    return W


def block_boot(H, W, s_chunk):
    """H (S, D, NB) int32, W (R, D) float32 -> p95 (R, S) int16 [s], on-time share (R, S) float32."""
    S, D, _ = H.shape
    R = W.shape[0]
    p95, ont = cp.empty((R, S), cp.int16), cp.empty((R, S), cp.float32)
    for s0 in range(0, S, s_chunk):
        s1 = min(S, s0 + s_chunk)
        Sc = s1 - s0
        M = H[s0:s1].transpose(1, 0, 2).reshape(D, Sc * NB).astype(cp.float32)
        T = (W @ M).reshape(R, Sc, NB)                      # cuBLAS SGEMM: all replicates at once
        del M
        c = cp.cumsum(T, axis=2, dtype=cp.float64)
        del T
        n = c[:, :, -1]
        k = cp.floor(Q * (n - 1))
        p95[:, s0:s1] = ((c <= k[:, :, None]).sum(axis=2) + D_MIN).astype(cp.int16)
        ont[:, s0:s1] = (c[:, :, ONT_BIN] / cp.maximum(n, 1)).astype(cp.float32)
        del c
        pool.free_all_blocks()
    return p95, ont


def ci(rep):
    s = cp.sort(rep, axis=0)
    return cp.asnumpy(s[I_LO]), cp.asnumpy(s[I_HI])


def main():
    t0 = time.time()
    dprod = pl.read_parquet(CUR / "dim_product.parquet").sort("pcode")
    dline = pl.read_parquet(CUR / "dim_line.parquet")
    P = int(dprod["pcode"].max()) + 1
    zug = int(dprod.filter(pl.col("product") == "Zug")["pcode"][0])
    hubs = (pl.read_parquet(eng.AGG / "ci_station_product.parquet")
            .filter(pl.col("product") == "Zug").sort("n", descending=True).head(N_HUBS))
    lines = pl.read_parquet(eng.AGG / "line.parquet").sort("n", descending=True).head(N_LINES)
    NH, NL = hubs.height, lines.height
    off_hour, off_hub, off_line = P, P + 24, P + 24 + NH
    S = off_line + NL
    pname = dict(zip(dprod["pcode"].to_list(), dprod["product"].to_list()))
    kinds = ["product"] * P + ["hour"] * 24 + ["hub"] * NH + ["line"] * NL
    labels = ([pname.get(i, f"p{i}") for i in range(P)] + [f"{h:02d}:00" for h in range(24)]
              + hubs["name"].to_list() + lines["line"].to_list())
    hub_uic = hubs["station_uic"].to_numpy().astype(np.int32)
    order = np.argsort(hub_uic)
    line_slot = np.full(dline.height, -1, np.int32)
    line_slot[lines["key"].to_numpy()] = off_line + np.arange(NL, dtype=np.int32)
    ctx = {"S": S, "zug": zug, "off_hour": off_hour, "line_slot": line_slot,
           "hub_sorted": hub_uic[order], "hub_slot_sorted": (off_hub + np.arange(NH, dtype=np.int32))[order]}

    P2 = [P2_START + timedelta(days=i) for i in range((P2_END - P2_START).days + 1)]
    P1 = [d - SHIFT for d in P2]
    D = len(P2)
    for d in P1 + P2:
        assert (CUR / "arrivals" / f"{d}.parquet").exists(), f"missing day {d}"

    budget, free = eng.acquire_budget()
    print(f"\nGPU: {eng.GPU} | free {free / 2**30:.2f} GB | budget {budget / 2**30:.2f} GB")
    print(f"P1 {P1[0]} .. {P1[-1]} | P2 {P2[0]} .. {P2[-1]} | {D} days each, weekday-aligned | "
          f"slots {S} (products {P}, train hours 24, hubs {NH}, lines {NL}) | B={B}")

    t = time.time()
    H1, H2 = day_hist(P1, ctx), day_hist(P2, ctx)
    sync(); t_build = time.time() - t
    n1, n2 = cp.asnumpy(H1.sum(axis=(1, 2))), cp.asnumpy(H2.sum(axis=(1, 2)))

    used = 2 * S * D * NB * 4
    s_chunk = max(1, int((budget - used) * 0.5 // (B * NB * 17)))
    t = time.time()
    W1, W2 = day_weights(B, D, SEED), day_weights(B, D, SEED + 1)
    r1_p95, r1_ont = block_boot(H1, W1, s_chunk)
    r2_p95, r2_ont = block_boot(H2, W2, s_chunk)
    sync(); t_boot = time.time() - t
    ones = cp.ones((1, D), cp.float32)
    pt1_p95, pt1_ont = (cp.asnumpy(x)[0] for x in block_boot(H1, ones, s_chunk))
    pt2_p95, pt2_ont = (cp.asnumpy(x)[0] for x in block_boot(H2, ones, s_chunk))

    # CPU reference for the same block bootstrap: one slot chunk with NumPy (multithreaded BLAS), extrapolated
    Sc = min(s_chunk, S)
    M = cp.asnumpy(H2[:Sc].transpose(1, 0, 2).reshape(D, Sc * NB)).astype(np.float32)
    Wh = cp.asnumpy(W2)
    t = time.time()
    c = np.cumsum((Wh @ M).reshape(B, Sc, NB), axis=2, dtype=np.float64)
    k = np.floor(Q * (c[:, :, -1] - 1))
    _ = (c <= k[:, :, None]).sum(axis=2)
    t_cpu = (time.time() - t) * (S / Sc) * 2
    del M, c

    # naive iid bootstrap (F3 kernel) on the same slots, for the design effect
    kern = cp.RawModule(code=KERNEL).get_function("boot_index")
    t = time.time()
    iid_w = np.full(S, np.nan)
    keep, _, _, lo, hi, _ = hist_boot_ci(kern, H2.sum(axis=1).astype(cp.int64), base=1 << 52)
    iid_w[cp.asnumpy(keep)] = cp.asnumpy(hi).astype(float) - cp.asnumpy(lo)
    sync(); t_iid = time.time() - t

    b_lo1, b_hi1 = ci(r1_p95)
    b_lo2, b_hi2 = ci(r2_p95)
    d_lo, d_hi = ci(r2_p95.astype(cp.int32) - r1_p95.astype(cp.int32))
    o_lo, o_hi = ci((r2_ont - r1_ont) * 100)

    res = pl.DataFrame({
        "kind": kinds, "label": labels,
        "n_2025": n1.astype(np.int64), "n_2026": n2.astype(np.int64),
        "p95_2025": pt1_p95.astype(np.int32), "p95_2026": pt2_p95.astype(np.int32),
        "lo_2025": b_lo1.astype(np.int32), "hi_2025": b_hi1.astype(np.int32),
        "lo_2026": b_lo2.astype(np.int32), "hi_2026": b_hi2.astype(np.int32),
        "d_lo": d_lo.astype(np.int32), "d_hi": d_hi.astype(np.int32),
        "ont_2025": (pt1_ont.astype(np.float64) * 100).round(1), "ont_2026": (pt2_ont.astype(np.float64) * 100).round(1),
        "d_ont_lo": o_lo.astype(np.float64).round(1), "d_ont_hi": o_hi.astype(np.float64).round(1),
        "ci_block_s": (b_hi2 - b_lo2).astype(float), "ci_iid_s": iid_w,
    }).with_columns(
        *[pl.when(pl.col(f"n_{y}") == 0).then(None).otherwise(pl.col(f"{c}_{y}")).alias(f"{c}_{y}")
          for y in ("2025", "2026") for c in ("p95", "lo", "hi", "ont")]
    ).with_columns(
        (pl.col("p95_2026") - pl.col("p95_2025")).alias("d_p95_s"),
        (pl.col("ont_2026") - pl.col("ont_2025")).round(1).alias("d_ont_pp"),
    ).with_columns(
        pl.when((pl.col("n_2025") < N_FLOOR) | (pl.col("n_2026") < N_FLOOR)).then(pl.lit("n/a"))
          .when((pl.col("d_lo") > 0) & (pl.col("d_p95_s") >= MIN_EFFECT_S)).then(pl.lit("WORSE"))
          .when((pl.col("d_hi") < 0) & (pl.col("d_p95_s") <= -MIN_EFFECT_S)).then(pl.lit("better"))
          .when((pl.col("d_lo") > 0) | (pl.col("d_hi") < 0)).then(pl.lit("small change"))
          .otherwise(pl.lit("no change")).alias("verdict"),
        pl.format("{} [{}, {}]", pl.col("p95_2026") - pl.col("p95_2025"), "d_lo", "d_hi").alias("d_p95 [CI]"),
        pl.format("{} [{}, {}]", (pl.col("ont_2026") - pl.col("ont_2025")).round(1), "d_ont_lo", "d_ont_hi").alias("d_ontime_pp [CI]"),
    )
    res.write_parquet(eng.AGG / "yoy_aug_sep.parquet")

    cols = ["label", "n_2026", "p95_2025", "p95_2026", "d_p95 [CI]", "ont_2025", "ont_2026", "d_ontime_pp [CI]", "verdict"]
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=34, thousands_separator=True,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print("\n=== Timing ===")
        print(pl.DataFrame([("day histograms (read + GPU scatter_add), 2 periods", t_build),
                            (f"GPU day-block bootstrap, 2 x {B} replicates (SGEMM)", t_boot),
                            ("CPU NumPy/BLAS same bootstrap (extrapolated)", t_cpu),
                            ("GPU iid bootstrap (F3 kernel), for comparison", t_iid)],
                           schema=["step", "sec"], orient="row").with_columns(pl.col("sec").round(2)))
        print(f"block bootstrap: GPU {t_cpu / t_boot:,.1f}x faster than CPU BLAS | slot chunk {s_chunk}\n")
        de = (res.filter((pl.col("n_2026") >= N_FLOOR) & (pl.col("ci_iid_s") > 0))
              .with_columns((pl.col("ci_block_s") / pl.col("ci_iid_s")).alias("de"))
              .group_by("kind").agg(pl.len().alias("slots"), pl.col("de").median().round(1).alias("median_design_effect"),
                                    pl.col("ci_iid_s").median().alias("median_iid_ci_s"),
                                    pl.col("ci_block_s").median().alias("median_block_ci_s")).sort("kind"))
        print("=== Design effect: how much wider the honest (day-block) CI is than the naive iid CI ===")
        print(de)
        print("\n=== By product: Aug-Sep 2026 vs 2025 ===")
        print(res.filter(pl.col("kind") == "product").sort("n_2026", descending=True).select(cols))
        print("\n=== Rail hubs (trains) ===")
        print(res.filter(pl.col("kind") == "hub").select(cols))
        print("\n=== Trains by hour ===")
        print(res.filter(pl.col("kind") == "hour").select(cols))
        ln = res.filter(pl.col("kind") == "line")
        print("\n=== Lines: biggest material deteriorations ===")
        print(ln.filter(pl.col("verdict") == "WORSE").sort("d_p95_s", descending=True).head(10).select(cols))
        print("\n=== Lines: biggest material improvements ===")
        print(ln.filter(pl.col("verdict") == "better").sort("d_p95_s").head(10).select(cols))
        print("\n=== Verdict counts ===")
        print(res.group_by("kind", "verdict").len().sort("kind", "verdict"))

    # forest plot: rail hubs
    h = res.filter(pl.col("kind") == "hub").sort("d_p95_s")
    y = np.arange(h.height)
    col = {"WORSE": "tab:red", "better": "tab:green", "small change": "tab:orange", "no change": "grey", "n/a": "lightgrey"}
    fig, ax = plt.subplots(figsize=(10, 9))
    for i, r in enumerate(h.iter_rows(named=True)):
        ax.errorbar(r["d_p95_s"], i, xerr=[[r["d_p95_s"] - r["d_lo"]], [r["d_hi"] - r["d_p95_s"]]],
                    fmt="o", color=col[r["verdict"]], capsize=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, h["label"].to_list())
    ax.set_xlabel("change in p95 arrival delay [s], Aug-Sep 2026 minus 2025 (95% day-block bootstrap CI)")
    ax.set_title("Swiss rail hubs, year over year: red = materially worse, green = better, grey = no change")
    fig.tight_layout(); fig.savefig(eng.OUT / "f4_yoy_hubs.png", dpi=130)

    # hour profile for trains, both years with CI bands
    hh = res.filter(pl.col("kind") == "hour")
    x = np.arange(24)
    fig, ax = plt.subplots(figsize=(13, 5))
    for yr, c in (("2025", "tab:blue"), ("2026", "tab:red")):
        ax.plot(x, hh[f"p95_{yr}"] / 60, "-o", color=c, label=f"Aug-Sep {yr}")
        ax.fill_between(x, hh[f"lo_{yr}"] / 60, hh[f"hi_{yr}"] / 60, color=c, alpha=0.2)
    ax.set_xticks(x)
    ax.set_xlabel("scheduled arrival hour"); ax.set_ylabel("p95 arrival delay [min]")
    ax.set_title(f"Swiss trains, p95 by hour, {D} weekday-aligned days per year, bands = 95% day-block CI "
                 "(Source: opentransportdata.swiss)")
    ax.legend()
    fig.tight_layout(); fig.savefig(eng.OUT / "f4_yoy_train_hours.png", dpi=130)
    print(f"\nplots -> {eng.OUT / 'f4_yoy_hubs.png'} | {eng.OUT / 'f4_yoy_train_hours.png'} | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
