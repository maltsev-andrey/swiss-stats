#!/usr/bin/env python3
"""swiss-stats F3.1: reporting policy for p95.
Tier 1: n >= N_FLOOR (theory: >= 10 observations above p95, else percentile bootstrap is unreliable).
Tier 2: per-cell measured CI decides precision: half-width <= max(60 s, 20% of |p95|) and no clip hit.
"""
import re
from pathlib import Path
import numpy as np
import polars as pl
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
CFG = yaml.safe_load(open(CFG_PATH))
ROOT = Path(CFG["data_root"])
AGG, OUT = ROOT / "agg", ROOT / "out"

D_MAX = 7200
Q = 0.95
N_FLOOR = int(np.ceil(10 / (1 - Q)))      # 200
ABS_HALF_S, REL_HALF = 60.0, 0.20
BINS = [30, 40, 50, 75, 100, 150, 200, 300, 500, 1_000, 2_000, 5_000, 10_000]

HALF = ((pl.col("hi_s") - pl.col("lo_s")) / 2)
TOL = pl.max_horizontal(pl.lit(ABS_HALF_S), REL_HALF * pl.col("p95_s").abs())
PRECISE_CI = (HALF <= TOL) & (pl.col("hi_s") < D_MAX)


def classify(df):
    hi_txt = pl.when(pl.col("hi_s") >= D_MAX).then(pl.lit("≥2h")).otherwise(pl.col("hi_s").cast(pl.Utf8))
    tier = (pl.when(pl.col("n") < N_FLOOR).then(pl.lit("n/a"))
            .when(PRECISE_CI).then(pl.lit("precise"))
            .otherwise(pl.lit("low precision")))
    return df.with_columns(
        (pl.col("hi_s") >= D_MAX).alias("clip_hit"),
        (pl.col("n") >= N_FLOOR).alias("reportable"),
        tier.alias("tier"),
    ).with_columns(
        (pl.col("tier") == "precise").alias("precise"),
        pl.when(pl.col("tier") == "n/a").then(pl.format("n/a (n={})", "n"))
          .when(pl.col("tier") == "precise").then(pl.format("{} [{}, {}]", "p95_s", "lo_s", hi_txt))
          .otherwise(pl.format("~{} [{}, {}]", "p95_s", "lo_s", hi_txt)).alias("display"))


def bin_table(d):
    return (d.group_by("bin_lo")
            .agg(pl.len().alias("groups"), pl.col("n").sum().alias("arrivals"),
                 pl.col("ci_width_s").median().round(0).alias("median_w_s"),
                 pl.col("ci_width_s").quantile(0.9).round(0).alias("p90_w_s"),
                 (100 * PRECISE_CI.mean()).round(1).alias("precise_by_ci_%"))
            .sort("bin_lo"))


def coverage(df, label):
    tot_g, tot_a = df.height, df["n"].sum()
    return (df.group_by("tier").agg(pl.len().alias("groups"), pl.col("n").sum().alias("arrivals"))
            .with_columns(pl.lit(label).alias("scope"),
                          (100 * pl.col("groups") / tot_g).round(1).alias("groups_%"),
                          (100 * pl.col("arrivals") / tot_a).round(1).alias("arrivals_%"))
            .select("scope", "tier", "groups", "groups_%", "arrivals", "arrivals_%"))


def main():
    n_days = pl.scan_parquet(str(ROOT / "curated" / "arrivals" / "*.parquet")).select(pl.col("date").n_unique()).collect().item()
    sph = classify(pl.read_parquet(AGG / "ci_station_product_hour.parquet"))
    sp = classify(pl.read_parquet(AGG / "ci_station_product.parquet"))
    pwh = classify(pl.read_parquet(AGG / "ci_product_weekday_hour.parquet"))

    edges = np.array(BINS)
    d = sph.filter(pl.col("n") >= BINS[0])
    d = d.with_columns(pl.Series("bin_lo", edges[np.searchsorted(edges, d["n"].to_numpy(), side="right") - 1]))

    order = {"precise": 0, "low precision": 1, "n/a": 2}
    with pl.Config(tbl_rows=40, tbl_cols=12, thousands_separator=True,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        for p in ["Zug", "Bus"]:
            print(f"\n=== {p}: CI width vs n, share precise by measured CI ===")
            print(bin_table(d.filter(pl.col("product") == p)))
        print(f"\npolicy: n >= {N_FLOOR} (theory) AND CI half-width <= max({ABS_HALF_S:.0f}s, {REL_HALF:.0%} of p95), no clip hit")
        cov = pl.concat([coverage(sph, "all"),
                         *[coverage(sph.filter(pl.col("product") == p), p) for p in ("Zug", "Tram", "Bus")]])
        print("\n=== Coverage of station x product x hour cells ===")
        print(cov.with_columns(pl.col("tier").replace_strict(order, return_dtype=pl.Int8).alias("o"))
                 .sort("scope", "o").drop("o"))
        hubs = sp.filter(pl.col("product") == "Zug").sort("n", descending=True).head(8)
        print("\n=== Rail hubs as reported ===")
        print(hubs.select("name", "n", "display", "tier"))
        lowp = (sph.filter((pl.col("product") == "Zug") & (pl.col("tier") == "low precision"))
                .sort("n", descending=True).head(8))
        print("\n=== Biggest train cells that are still low precision ===")
        print(lowp.select("name", "hour", "n", "display"))

    txt = CFG_PATH.read_text()
    lines = {"report_min_n": f"report_min_n: {N_FLOOR}                                    # F3.1: p95 shown only if n >= this",
             "precise_rule": f'precise_rule: "half_width <= max({ABS_HALF_S:.0f}s, {REL_HALF:.0%} of p95), no clip"'}
    for key, line in lines.items():
        txt = re.sub(rf"^{key}:.*$", line, txt, flags=re.M) if re.search(rf"^{key}:", txt, re.M) else txt.rstrip() + "\n" + line + "\n"
    CFG_PATH.write_text(txt)
    for df, f in ((sph, "ci_station_product_hour"), (sp, "ci_station_product"), (pwh, "ci_product_weekday_hour")):
        df.write_parquet(AGG / f"{f}.parquet")

    # plot 1: CI width vs n
    sample = d.sample(n=min(60_000, d.height), seed=1)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(sample["n"], sample["ci_width_s"] + 1, s=2, alpha=0.15, color="grey", label="cells (sample)")
    for p, c in (("Zug", "tab:blue"), ("Bus", "tab:orange")):
        t = bin_table(d.filter(pl.col("product") == p))
        x = t["bin_lo"].to_numpy() * 1.2
        ax.plot(x, t["median_w_s"].to_numpy() + 1, "-o", color=c, label=f"{p}: median CI width")
        ax.plot(x, t["p90_w_s"].to_numpy() + 1, "--", color=c, label=f"{p}: p90 CI width")
    ax.axhline(2 * ABS_HALF_S, color="green", lw=1, label="±1 min")
    ax.axvline(N_FLOOR, color="red", lw=1.5, label=f"n floor = {N_FLOOR} (≥10 obs above p95)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arrivals in cell (n)"); ax.set_ylabel("width of 95% CI for p95 [s] (+1)")
    ax.set_title(f"How many arrivals do we need to trust a p95? Swiss ÖV, {n_days} days, GPU bootstrap B=1000")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "f31_ci_vs_n.png", dpi=130)

    # plot 2: trains heatmap with policy applied
    z = pwh.filter((pl.col("product") == "Zug") & (pl.col("hour") < 24))
    p95, half, nn = np.full((7, 24), np.nan), np.full((7, 24), np.nan), np.zeros((7, 24))
    tier = np.full((7, 24), "", dtype=object)
    for r in z.iter_rows(named=True):
        w, h = r["wd"], r["hour"]
        p95[w, h], half[w, h], nn[w, h], tier[w, h] = r["p95_s"] / 60, (r["hi_s"] - r["lo_s"]) / 120, r["n"], r["tier"]
    masked = np.ma.masked_where(np.isnan(p95) | (nn < N_FLOOR), p95)
    cmap = plt.get_cmap("magma_r").copy(); cmap.set_bad("#dcdcdc")
    fig, ax = plt.subplots(figsize=(17, 5.2))
    im = ax.imshow(masked, aspect="auto", cmap=cmap)
    for w in range(7):
        for h in range(24):
            if nn[w, h] == 0:
                continue
            if tier[w, h] == "n/a":
                ax.text(h, w, f"n={int(nn[w, h])}", ha="center", va="center", fontsize=6, color="#555")
                continue
            pre = "" if tier[w, h] == "precise" else "~"
            col = "white" if im.norm(p95[w, h]) > 0.6 else "black"
            ax.text(h, w, f"{pre}{p95[w, h]:.1f}\n±{half[w, h]:.1f}", ha="center", va="center", fontsize=6.5, color=col)
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(24))
    ax.set_xlabel("scheduled arrival hour (Source: opentransportdata.swiss)")
    ax.set_title(f"Swiss trains: p95 arrival delay [min] ± 95% CI half-width | grey: n < {N_FLOOR} | "
                 f"~: low precision | {z['n'].sum() / 1e6:.2f}M arrivals, {n_days} days")
    fig.colorbar(im, ax=ax, label="p95, min")
    fig.tight_layout(); fig.savefig(OUT / "f31_zug_p95_masked.png", dpi=130)
    print(f"\nplots -> {OUT / 'f31_ci_vs_n.png'} | {OUT / 'f31_zug_p95_masked.png'}")


if __name__ == "__main__":
    main()
