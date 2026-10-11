#!/usr/bin/env python3
"""DQ rule 6: line identity by route fingerprint, not by vendor name.
A line that disappears is linked to a line that appears when their stop sets match:
Jaccard >= 0.8, same product, >= 4 stops and >= 200 arrivals on each side, strict 1:1 (greedy by Jaccard).
Output: curated/line_renames.parquet (lcode_old -> lcode_new)."""
from datetime import date, timedelta
import polars as pl

CUR = "/data/swiss-stats/curated"
J_MIN, MIN_STOPS, MIN_N = 0.8, 4, 200
P2 = [date(2026, 8, 2) + timedelta(days=i) for i in range(60)]
P1 = [d - timedelta(days=364) for d in P2]

dl = pl.read_parquet(f"{CUR}/dim_line.parquet")
dp = pl.read_parquet(f"{CUR}/dim_product.parquet")
lf = (pl.scan_parquet(f"{CUR}/arrivals/*.parquet").filter(pl.col("date").is_in(P1 + P2))
      .with_columns(pl.when(pl.col("date").is_in(P1)).then(pl.lit("old")).otherwise(pl.lit("new")).alias("per")))
agg = (lf.group_by("per", "lcode").agg(pl.len().alias("n"), pl.col("station_uic").unique().alias("stops"),
                                       pl.col("pcode").mode().first().alias("pcode"))
       .collect(engine="streaming").join(dp, on="pcode").join(dl, on="lcode"))
a, b = agg.filter(pl.col("per") == "old"), agg.filter(pl.col("per") == "new")
cols = ["lcode", "operator", "line_text", "product", "n", "stops"]
gone = a.join(b.select("lcode"), on="lcode", how="anti").select(cols)
born = b.join(a.select("lcode"), on="lcode", how="anti").select(cols)
pairs = (gone.join(born, how="cross", suffix="_new")
         .filter((pl.col("product") == pl.col("product_new"))
                 & (pl.col("stops").list.len() >= MIN_STOPS) & (pl.col("stops_new").list.len() >= MIN_STOPS)
                 & (pl.col("n") >= MIN_N) & (pl.col("n_new") >= MIN_N))
         .with_columns((pl.col("stops").list.set_intersection("stops_new").list.len()
                        / pl.col("stops").list.set_union("stops_new").list.len()).alias("jaccard"))
         .filter(pl.col("jaccard") >= J_MIN).sort("jaccard", descending=True))
seen_old, seen_new, keep = set(), set(), []
for r in pairs.drop("stops", "stops_new").iter_rows(named=True):
    if r["lcode"] in seen_old or r["lcode_new"] in seen_new:
        continue
    seen_old.add(r["lcode"]); seen_new.add(r["lcode_new"]); keep.append(r)
ren = pl.DataFrame(keep).rename({"lcode": "lcode_old"}) if keep else pl.DataFrame()
ren.write_parquet(f"{CUR}/line_renames.parquet")
with pl.Config(tbl_rows=30, fmt_str_lengths=20, thousands_separator=True, tbl_hide_dataframe_shape=True):
    print(f"gone {gone.height} | born {born.height} | renames (guarded, 1:1): {ren.height}")
    print(ren.select("operator", "line_text", "operator_new", "line_text_new", "product", "n", "n_new", "jaccard"))

# Lausanne metro: did the stop IDs change together with operator and name?
m = (pl.scan_parquet(f"{CUR}/arrivals/*.parquet").filter(pl.col("lcode").is_in([420, 421, 2246, 2361]))
     .group_by("lcode").agg(pl.col("date").min().alias("first"), pl.col("date").max().alias("last"),
                            pl.len().alias("n"), pl.col("station_uic").unique().alias("stops"))
     .collect(engine="streaming").join(dl, on="lcode").sort("lcode"))
print("\n=== Lausanne metro across the whole history ===")
print(m.select("operator", "line_text", "first", "last", "n", pl.col("stops").list.len().alias("n_stops")))
s = {r["lcode"]: set(r["stops"]) for r in m.iter_rows(named=True)}
for old, new in ((420, 2246), (421, 2361)):
    if old in s and new in s:
        j = len(s[old] & s[new]) / len(s[old] | s[new])
        print(f"stop-set Jaccard lcode {old} vs {new}: {j:.2f} | sample old {sorted(s[old])[:3]} new {sorted(s[new])[:3]}")
