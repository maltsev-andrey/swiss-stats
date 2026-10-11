#!/usr/bin/env python3
"""swiss-stats F4.1: like-for-like YoY, Aug-Sep 2025 vs 2026.
1) Line panel: present in both periods, volume ratio within [0.8, 1.25], stop-set Jaccard >= 0.8.
2) Product level on panel lines only, each line assigned its dominant product (removes coverage and label drift).
3) Day-block bootstrap from F4; p-values from bootstrap SE (z), Benjamini-Hochberg FDR per family.
"""
import math, sys, time
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f4_yoy as f4
eng = f4.eng
import cupy as cp
import cupyx
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

CUR = f4.CUR
D_MIN, D_MAX, NB, B = f4.D_MIN, f4.D_MAX, f4.NB, f4.B
VOL_LO, VOL_HI, JACCARD_MIN = 0.8, 1.25, 0.8
FDR_Q, MIN_EFFECT_S, N_FLOOR = 0.05, f4.MIN_EFFECT_S, f4.N_FLOOR


def bh(p, q):
    m = len(p)
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = (np.max(np.nonzero(passed)[0]) + 1) if passed.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


def day_hist(days, ctx):
    S, D = ctx["S"], len(days)
    H = cp.zeros(S * D * NB, dtype=cp.int32)
    for di, d in enumerate(days):
        df = pl.read_parquet(CUR / "arrivals" / f"{d}.parquet", columns=["station_uic", "pcode", "lcode", "hour", "delay"])
        st, ln = df["station_uic"].to_numpy(), df["lcode"].to_numpy()
        pc, hr = df["pcode"].to_numpy().astype(np.int32), df["hour"].to_numpy().astype(np.int32)
        b = (np.clip(df["delay"].to_numpy(), D_MIN, D_MAX) - D_MIN).astype(np.int64)
        hs, hslot = ctx["hub_sorted"], ctx["hub_slot_sorted"]
        pos = np.minimum(np.searchsorted(hs, st), len(hs) - 1)
        is_zug = pc == ctx["zug"]
        hub = np.where((hs[pos] == st) & is_zug, hslot[pos], -1)
        hrs = np.where(is_zug, ctx["off_hour"] + hr, -1)
        slots = np.concatenate([ctx["prod_slot"][ln], hrs, hub, ctx["line_slot"][ln]])
        bins = np.concatenate([b, b, b, b])
        keep = slots >= 0
        cupyx.scatter_add(H, cp.asarray((slots[keep].astype(np.int64) * D + di) * NB + bins[keep]), 1)
    return H.reshape(S, D, NB)


def main():
    t0 = time.time()
    P2 = [f4.P2_START + timedelta(days=i) for i in range((f4.P2_END - f4.P2_START).days + 1)]
    P1 = [d - f4.SHIFT for d in P2]
    D = len(P2)
    dprod = pl.read_parquet(CUR / "dim_product.parquet").sort("pcode")
    dline = pl.read_parquet(CUR / "dim_line.parquet")
    pname = {int(k): v for k, v in zip(dprod["pcode"].to_list(), dprod["product"].to_list())}
    P = max(pname) + 1
    zug = [k for k, v in pname.items() if v == "Zug"][0]

    # ---- 1) line panel ----
    t = time.time()
    lf = (pl.scan_parquet(str(CUR / "arrivals" / "*.parquet")).filter(pl.col("date").is_in(P1 + P2))
          .with_columns(pl.when(pl.col("date").is_in(P1)).then(pl.lit("2025")).otherwise(pl.lit("2026")).alias("per")))
    agg = (lf.group_by("per", "lcode").agg(pl.len().alias("n"), pl.col("station_uic").unique().alias("stops"))
           .collect(engine="streaming"))
    dom = (lf.group_by("lcode", "pcode").len().collect(engine="streaming")
           .sort("len", descending=True).unique("lcode", keep="first")
           .select("lcode", pl.col("pcode").cast(pl.Int64).alias("dom_pc")))
    a = agg.filter(pl.col("per") == "2025").drop("per").rename({"n": "n_2025", "stops": "stops_2025"})
    b = agg.filter(pl.col("per") == "2026").drop("per").rename({"n": "n_2026", "stops": "stops_2026"})
    pan = (a.join(b, on="lcode", how="full", coalesce=True).join(dom, on="lcode", how="left")
           .with_columns((pl.col("n_2026") / pl.col("n_2025")).alias("vol_ratio"),
                         (pl.col("stops_2025").list.set_intersection("stops_2026").list.len()
                          / pl.col("stops_2025").list.set_union("stops_2026").list.len()).alias("jaccard"))
           .with_columns(pl.when(pl.col("n_2025").is_null()).then(pl.lit("new"))
                           .when(pl.col("n_2026").is_null()).then(pl.lit("dropped"))
                           .when(pl.col("jaccard") < JACCARD_MIN).then(pl.lit("route changed"))
                           .when(~pl.col("vol_ratio").is_between(VOL_LO, VOL_HI)).then(pl.lit("volume changed"))
                           .otherwise(pl.lit("panel")).alias("status"),
                         pl.col("dom_pc").replace_strict(pname, default="?", return_dtype=pl.Utf8).alias("dom_product"))
           .join(dline.select("lcode", "operator", "line_text"), on="lcode", how="left")
           .drop("stops_2025", "stops_2026"))
    t_panel = time.time() - t

    # ---- slots ----
    hubs = (pl.read_parquet(eng.AGG / "ci_station_product.parquet")
            .filter(pl.col("product") == "Zug").sort("n", descending=True).head(f4.N_HUBS))
    lines = pl.read_parquet(eng.AGG / "line.parquet").sort("n", descending=True).head(f4.N_LINES)
    NH, NL = hubs.height, lines.height
    off_hour, off_hub, off_line = P, P + 24, P + 24 + NH
    S = off_line + NL
    kinds = np.array(["product"] * P + ["hour"] * 24 + ["hub"] * NH + ["line"] * NL)
    labels = ([pname.get(i, f"p{i}") for i in range(P)] + [f"{h:02d}:00" for h in range(24)]
              + hubs["name"].to_list() + lines["line"].to_list())
    L = dline.height
    prod_slot = np.full(L, -1, np.int32)
    pp = pan.filter(pl.col("status") == "panel")
    prod_slot[pp["lcode"].to_numpy()] = pp["dom_pc"].to_numpy().astype(np.int32)
    line_slot = np.full(L, -1, np.int32)
    line_slot[lines["key"].to_numpy()] = off_line + np.arange(NL, dtype=np.int32)
    hub_uic = hubs["station_uic"].to_numpy().astype(np.int32)
    order = np.argsort(hub_uic)
    ctx = {"S": S, "zug": zug, "off_hour": off_hour, "prod_slot": prod_slot, "line_slot": line_slot,
           "hub_sorted": hub_uic[order], "hub_slot_sorted": (off_hub + np.arange(NH, dtype=np.int32))[order]}
    line_status = dict(zip(pan["lcode"].to_list(), pan["status"].to_list()))
    status = np.array(["-"] * (S - NL) + [line_status.get(int(k), "?") for k in lines["key"].to_list()], dtype=object)

    # ---- 2) histograms + day-block bootstrap (F4 engine) ----
    budget, free = eng.acquire_budget()
    print(f"\nGPU: {eng.GPU} | free {free / 2**30:.2f} GB | budget {budget / 2**30:.2f} GB | "
          f"P1 {P1[0]}..{P1[-1]} | P2 {P2[0]}..{P2[-1]} | {D} days | slots {S} | B={B}")
    t = time.time()
    H1, H2 = day_hist(P1, ctx), day_hist(P2, ctx)
    eng.sync(); t_build = time.time() - t
    n1, n2 = cp.asnumpy(H1.sum(axis=(1, 2))), cp.asnumpy(H2.sum(axis=(1, 2)))
    s_chunk = max(1, int((budget - 2 * S * D * NB * 4) * 0.5 // (B * NB * 17)))
    t = time.time()
    r1_p95, r1_ont = f4.block_boot(H1, f4.day_weights(B, D, f4.SEED), s_chunk)
    r2_p95, r2_ont = f4.block_boot(H2, f4.day_weights(B, D, f4.SEED + 1), s_chunk)
    eng.sync(); t_boot = time.time() - t
    ones = cp.ones((1, D), cp.float32)
    pt1, po1 = (cp.asnumpy(x)[0] for x in f4.block_boot(H1, ones, s_chunk))
    pt2, po2 = (cp.asnumpy(x)[0] for x in f4.block_boot(H2, ones, s_chunk))

    # ---- 3) inference: percentile CI for display, z-test with bootstrap SE, BH-FDR per family ----
    d_rep = r2_p95.astype(cp.float32) - r1_p95.astype(cp.float32)
    se = cp.asnumpy(d_rep.std(axis=0)).astype(float)
    d_lo, d_hi = f4.ci(d_rep)
    o_lo, o_hi = f4.ci((r2_ont - r1_ont) * 100)
    d_pt = pt2.astype(float) - pt1.astype(float)
    z = np.where(se > 0, np.abs(d_pt) / np.maximum(se, 1e-9), np.where(d_pt != 0, 50.0, 0.0))
    p = np.array([math.erfc(x / math.sqrt(2)) for x in z])
    elig = (n1 >= N_FLOOR) & (n2 >= N_FLOOR) & ((kinds != "line") | (status == "panel"))
    rej = np.zeros(S, bool)
    for k in np.unique(kinds):
        idx = np.nonzero(elig & (kinds == k))[0]
        if len(idx):
            rej[idx] = bh(p[idx], FDR_Q)
    verdict = np.where(~elig, np.where((kinds == "line") & (status != "panel"), status, "n/a"),
              np.where(rej & (d_pt >= MIN_EFFECT_S), "WORSE",
              np.where(rej & (d_pt <= -MIN_EFFECT_S), "better",
              np.where(rej, "small change", "no change"))))

    res = pl.DataFrame({
        "kind": kinds, "label": labels, "status": status.astype(str),
        "n_2025": n1.astype(np.int64), "n_2026": n2.astype(np.int64),
        "p95_2025": pt1.astype(np.int32), "p95_2026": pt2.astype(np.int32),
        "d_p95_s": d_pt.astype(np.int32), "d_lo": d_lo.astype(np.int32), "d_hi": d_hi.astype(np.int32),
        "ont_2025": (po1.astype(np.float64) * 100).round(1), "ont_2026": (po2.astype(np.float64) * 100).round(1),
        "d_ont_lo": o_lo.astype(np.float64).round(1), "d_ont_hi": o_hi.astype(np.float64).round(1),
        "p_value": p, "fdr_reject": rej, "verdict": verdict.astype(str),
    }).with_columns(
        *[pl.when(pl.col(f"n_{y}") == 0).then(None).otherwise(pl.col(c)).alias(c)
          for y in ("2025", "2026") for c in (f"p95_{y}", f"ont_{y}")],
    ).with_columns(
        pl.format("{} [{}, {}]", "d_p95_s", "d_lo", "d_hi").alias("d_p95 [CI]"),
        pl.format("{} [{}, {}]", (pl.col("ont_2026") - pl.col("ont_2025")).round(1), "d_ont_lo", "d_ont_hi").alias("d_ontime_pp [CI]"),
    )
    res.write_parquet(eng.AGG / "yoy_aug_sep_panel.parquet")

    cols = ["label", "n_2025", "n_2026", "p95_2025", "p95_2026", "d_p95 [CI]", "d_ontime_pp [CI]", "p_value", "verdict"]
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=30, thousands_separator=True, float_precision=4,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print("\n=== Timing ===")
        print(pl.DataFrame([("line panel (Polars, 120 days)", t_panel), ("day histograms, 2 periods", t_build),
                            (f"GPU day-block bootstrap, 2 x {B}", t_boot)],
                           schema=["step", "sec"], orient="row").with_columns(pl.col("sec").round(2)))
        print("\n=== Coverage: lines and arrivals by status and dominant product ===")
        print(pan.group_by("dom_product", "status").agg(pl.len().alias("lines"), pl.col("n_2025").sum(), pl.col("n_2026").sum())
                 .sort("dom_product", "status"))
        print("\n=== Where did Metro come from? ===")
        print(pan.filter(pl.col("dom_product") == "Metro").select("operator", "line_text", "status", "n_2025", "n_2026", "jaccard"))
        print("\n=== Products, like-for-like (panel lines, dominant product) ===")
        print(res.filter(pl.col("kind") == "product").sort("n_2026", descending=True, nulls_last=True).select(cols))
        print("\n=== Rail hubs (trains), BH-FDR 5% ===")
        print(res.filter(pl.col("kind") == "hub").sort("p_value").select(cols))
        print("\n=== Trains by hour, BH-FDR 5%: only non-'no change' ===")
        print(res.filter((pl.col("kind") == "hour") & (pl.col("verdict") != "no change")).select(cols))
        ln = res.filter(pl.col("kind") == "line")
        print("\n=== Top lines: status ===")
        print(ln.group_by("status").len().sort("len", descending=True))
        print("\n=== Panel lines: material deteriorations after FDR ===")
        print(ln.filter(pl.col("verdict") == "WORSE").sort("d_p95_s", descending=True).head(10).select(cols))
        print("\n=== Panel lines: material improvements after FDR ===")
        print(ln.filter(pl.col("verdict") == "better").sort("d_p95_s").head(10).select(cols))
        old = eng.AGG / "yoy_aug_sep.parquet"
        if old.exists():
            cmp = (pl.read_parquet(old).select("kind", "label", pl.col("verdict").alias("F4"))
                   .join(res.select("kind", "label", pl.col("verdict").alias("F4.1")), on=["kind", "label"]))
            print("\n=== How many F4 claims survive like-for-like + FDR (lines and hubs) ===")
            print(cmp.filter(pl.col("kind").is_in(["line", "hub"]))
                     .group_by("kind", "F4", "F4.1").len().sort("kind", "F4", "len", descending=[False, False, True]))

    sel = res.filter((pl.col("kind") == "line") & pl.col("verdict").is_in(["WORSE", "better"])).sort("d_p95_s")
    if sel.height:
        fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * sel.height)))
        for i, r in enumerate(sel.iter_rows(named=True)):
            ax.errorbar(r["d_p95_s"], i, xerr=[[r["d_p95_s"] - r["d_lo"]], [r["d_hi"] - r["d_p95_s"]]], fmt="o",
                        color="tab:red" if r["verdict"] == "WORSE" else "tab:green", capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(sel.height), sel["label"].to_list())
        ax.set_xlabel("change in p95 arrival delay [s], Aug-Sep 2026 minus 2025 (95% day-block CI)")
        ax.set_title("Lines with stable route and volume, significant after BH-FDR 5% and |change| >= 10 s")
        fig.tight_layout(); fig.savefig(eng.OUT / "f41_yoy_lines.png", dpi=130)
    print(f"\nplot -> {eng.OUT / 'f41_yoy_lines.png'} | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
