#!/usr/bin/env python3
"""swiss-stats F5: self-contained HTML report from agg/ (Plotly via CDN, all data inlined)."""
import json, re
from pathlib import Path
import polars as pl
import yaml

CFG = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config" / "config.yaml"))
ROOT = Path(CFG["data_root"]); AGG, CUR, OUT = ROOT / "agg", ROOT / "curated", ROOT / "out"
LOGS = ROOT
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def grep(path, pattern, default="n/a"):
    try:
        m = re.search(pattern, (LOGS / path).read_text())
        return m.group(1).strip() if m else default
    except FileNotFoundError:
        return default


def rows(df, cols):
    return df.select(cols).to_dicts()


def table(recs, cols, fmt=None):
    fmt = fmt or {}
    th = "".join(f"<th>{c[1]}</th>" for c in cols)
    body = ""
    for r in recs:
        tds = ""
        for key, _ in cols:
            v = r.get(key)
            v = fmt.get(key, lambda x: x)(v) if v is not None else "–"
            cls = {"WORSE": "bad", "better": "good"}.get(str(v), "")
            tds += f'<td class="{cls}">{v}</td>'
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def main():
    prod = pl.read_parquet(AGG / "product.parquet").sort("n", descending=True)
    hubs = (pl.read_parquet(AGG / "ci_station_product.parquet").filter(pl.col("product") == "Zug")
            .join(pl.read_parquet(AGG / "station_product.parquet").filter(pl.col("product") == "Zug")
                  .select("station_uic", "ontime_pct", "window_pct"), on="station_uic", how="left")
            .sort("n", descending=True).head(25))
    pwh = pl.read_parquet(AGG / "ci_product_weekday_hour.parquet").filter((pl.col("product") == "Zug") & (pl.col("hour") < 24))
    lines = (pl.read_parquet(AGG / "line.parquet").filter(pl.col("n") >= 30_000)
             .with_columns(pl.col("line").str.replace_all(r"\s+", " ")))
    yoy = pl.read_parquet(AGG / "yoy_aug_sep_panel.parquet")
    yoy0 = pl.read_parquet(AGG / "yoy_aug_sep.parquet")
    n_days = len(list((CUR / "arrivals").glob("*.parquet")))
    n_arr = int(prod["n"].sum())
    n_st = pl.read_parquet(CUR / "dim_station.parquet").height
    n_ln = pl.read_parquet(CUR / "dim_line.parquet").height
    n_ren = pl.read_parquet(CUR / "line_renames.parquet").height if (CUR / "line_renames.parquet").exists() else 0
    days = pl.scan_parquet(str(CUR / "arrivals" / "*.parquet")).select(pl.col("date").min().alias("a"), pl.col("date").max().alias("b")).collect()
    d_first, d_last = days["a"][0], days["b"][0]

    # heatmap matrices
    z = [[None] * 24 for _ in range(7)]; w = [[None] * 24 for _ in range(7)]; txt = [[""] * 24 for _ in range(7)]
    for r in pwh.iter_rows(named=True):
        i, j = r["wd"], r["hour"]
        z[i][j] = round(r["p95_s"] / 60, 2); w[i][j] = round((r["hi_s"] - r["lo_s"]) / 60, 2)
        txt[i][j] = f"n={r['n']:,}<br>p95 {r['p95_s']} s [{r['lo_s']}, {r['hi_s']}]<br>{r['tier']}"

    # YoY
    yh = yoy.filter(pl.col("kind") == "hub").sort("d_p95_s")
    yl = yoy.filter((pl.col("kind") == "line") & pl.col("verdict").is_in(["WORSE", "better"])).sort("d_p95_s")
    surv = (yoy0.select("kind", "label", pl.col("verdict").alias("F4"))
            .join(yoy.select("kind", "label", pl.col("verdict").alias("F41")), on=["kind", "label"])
            .filter(pl.col("kind") == "line").group_by("F4", "F41").len().sort("F4", "len", descending=[False, True]))
    de = (yoy0.filter((pl.col("n_2026") >= 200) & (pl.col("ci_iid_s") > 0))
          .with_columns((pl.col("ci_block_s") / pl.col("ci_iid_s")).alias("de"))
          .group_by("kind").agg(pl.len().alias("slots"), pl.col("ci_iid_s").median().alias("iid"),
                                pl.col("ci_block_s").median().alias("block"), pl.col("de").median().round(1).alias("de")).sort("kind"))

    bench = [
        ("Bootstrap CI for p95, all station×product×hour groups (F3)", "CPU naive resample", grep("f3_full.log", r"CPU naive resample.*?┆\s*([\d.,]+ [sh])")),
        ("", "CPU NumPy, Beta trick", grep("f3_full.log", r"CPU NumPy Beta trick.*?┆\s*([\d.,]+ [sh])")),
        ("", "GPU kernel, Beta trick", grep("f3_full.log", r"GPU kernel Beta trick.*?┆\s*([\d.,]+ [sh])")),
        ("", "Total speedup", grep("f3_full.log", r"total ([\d,]+)x") + "x"),
        ("Day-block bootstrap, 2×1000 replicates (F4)", "GPU vs CPU BLAS", grep("f4.log", r"GPU ([\d.]+)x faster") + "x"),
        ("Group statistics, 6 groupings, all metrics (F2)", "GPU", grep("f2_full.log", r"GPU: 6 groupings.*?┆\s*([\d.]+)") + " s"),
        ("", "CPU Polars, 1 grouping (scaled)", grep("f2_full.log", r"CPU Polars: 1 grouping.*?┆\s*([\d.]+)") + " s"),
        ("", "GPU vs CPU exact-match check", grep("f2_full.log", r"mismatches \(n, p50, p95\): (\d+)") + " mismatches"),
    ]
    try:
        g6, c1 = float(bench[5][2].split()[0]), float(bench[6][2].split()[0])
        bench.insert(7, ("", "Per grouping: GPU vs CPU", f"{g6 / 6:.1f} s vs {c1:.1f} s, {c1 / (g6 / 6):.1f}x"))
    except ValueError:
        pass
    dq = [
        ("Swiss stop = UIC prefix 85 (string), not integer division", "19% 'Swiss' stops instead of 98% (7- vs 9-digit BPUIC)"),
        ("|delay| > 1 day is a broken timestamp, not a delay", "8,821 records, down to −56 years; mean shifts, p95 does not"),
        ("One canonical raw schema across format evolution", "v2 had no SLOID before 2025-10; readers refused mixed files"),
        ("Product label normalization", "'Bua' as an 8th product on 5,250 rows"),
        ("Prefix 85 means 'in the Swiss register', not 'in Switzerland'", "Italian stations (Luino line) among the worst stops"),
        ("Line identity by route fingerprint, not vendor name", f"{n_ren} renames found (≈1% of volume drift); 99% was new coverage"),
    ]
    meth = [
        ("Quantiles, not means", "8k garbage values moved the mean, not a single p95."),
        ("Design effect", "Arrivals within a day are correlated; iid bootstrap CIs were 2.5–10× too narrow. We resample whole days (GEMM on GPU)."),
        ("Like-for-like + FDR", "45% of line-level 'changes' were route rebuilds, volume shifts or noise. Panel of stable lines, Benjamini–Hochberg 5%."),
        ("Reporting policy", "p95 shown only with n ≥ 200 (≥ 10 obs above p95); 'precise' only if CI half-width ≤ max(60 s, 20%)."),
    ]

    i1 = lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x
    f1 = lambda x: f"{x:.1f}" if isinstance(x, float) else x
    H = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Swiss ÖV punctuality · GPU statistics</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body{{font:15px/1.45 system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:24px;color:#222;background:#fafafa}}
h1{{margin:0 0 4px}} h2{{margin-top:40px;border-bottom:2px solid #ddd;padding-bottom:4px}} .sub{{color:#666;margin-bottom:20px}}
.kpi{{display:flex;gap:16px;flex-wrap:wrap}} .kpi div{{background:#fff;border:1px solid #e3e3e3;border-radius:8px;padding:12px 18px;min-width:150px}}
.kpi b{{display:block;font-size:26px}} table{{border-collapse:collapse;width:100%;background:#fff;font-size:13.5px;margin:10px 0}}
th,td{{border:1px solid #e3e3e3;padding:5px 8px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
th{{background:#f1f1f1}} td.bad{{color:#b3261e;font-weight:600}} td.good{{color:#1b7f3b;font-weight:600}}
.note{{background:#fff8e1;border-left:4px solid #f0b429;padding:8px 12px;margin:10px 0}} .plot{{background:#fff;border:1px solid #e3e3e3;border-radius:8px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Swiss public transport punctuality</h1>
<div class="sub">{n_arr:,} measured arrivals · {n_days} days ({d_first} … {d_last}) · GPU-accelerated statistics on a Tesla P100 · Source: opentransportdata.swiss (Ist-Daten v2)</div>
<div class="kpi"><div><b>{n_arr/1e6:,.0f}M</b>arrivals</div><div><b>{n_days}</b>days</div><div><b>{n_st:,}</b>stops</div><div><b>{n_ln:,}</b>lines</div>
<div><b>{bench[3][2]}</b>bootstrap speedup vs naive CPU</div><div><b>{bench[7][2].split()[0]}</b>GPU vs CPU mismatches</div></div>

<h2>1. Punctuality by mode</h2>
<p>On-time = arrival ≤ 3 min late (passenger view). Window = arrival within [−1, +3] min (timetable accuracy). Early = more than 1 min before schedule.</p>
{table(rows(prod, ["product","n","ontime_pct","window_pct","early_pct","p50_s","p95_s"]),
       [("product","Mode"),("n","Arrivals"),("ontime_pct","On-time %"),("window_pct","Window %"),("early_pct","Early %"),("p50_s","p50 [s]"),("p95_s","p95 [s]")],
       {"n":i1,"ontime_pct":f1,"window_pct":f1,"early_pct":f1})}

<h2>2. Rail hubs: p95 arrival delay with 95% bootstrap CI</h2>
{table(rows(hubs, ["name","n","ontime_pct","display","tier"]),
       [("name","Station"),("n","Train arrivals"),("ontime_pct","On-time %"),("display","p95 [s] (95% CI)"),("tier","Precision")], {"n":i1,"ontime_pct":f1})}
<div class="note">CIs here are per-arrival (iid) bootstrap: exact for the level, but optimistic for comparisons between periods — see design effect in section 5.</div>

<h2>3. Trains: when are delays worst?</h2>
<div class="grid"><div id="hm" class="plot"></div><div id="hmw" class="plot"></div></div>

<h2>4. Lines</h2>
<div class="grid"><div><h3>Worst 15 by p95 (n ≥ 30k)</h3>
{table(rows(lines.sort("p95_s", descending=True).head(15), ["line","n","ontime_pct","p95_s"]), [("line","Line"),("n","Arrivals"),("ontime_pct","On-time %"),("p95_s","p95 [s]")], {"n":i1,"ontime_pct":f1})}
</div><div><h3>Best 15 by window [−1, +3]</h3>
{table(rows(lines.sort("window_pct", descending=True).head(15), ["line","n","window_pct","p95_s"]), [("line","Line"),("n","Arrivals"),("window_pct","Window %"),("p95_s","p95 [s]")], {"n":i1,"window_pct":f1})}
</div></div>

<h2>5. Year over year: Aug–Sep 2026 vs 2025</h2>
<p>60 weekday-aligned days per year (shift 364 d). Day-block bootstrap (whole days resampled, B=1000, cuBLAS SGEMM). Lines compared only if route (stop-set Jaccard ≥ 0.8) and volume (±25%) are stable. Benjamini–Hochberg FDR 5% per family; material = |Δp95| ≥ 10 s.</p>
<div class="grid"><div id="forest" class="plot"></div><div id="flines" class="plot"></div></div>
<h3>Design effect: honest (day-block) CI width ÷ naive (iid) CI width</h3>
{table(de.to_dicts(), [("kind","Family"),("slots","Slots"),("iid","Median iid CI [s]"),("block","Median day-block CI [s]"),("de","Design effect")], {"iid":f1,"block":f1})}
<h3>How many naive YoY claims (F4) survived like-for-like + FDR (F4.1), top-150 lines</h3>
{table(surv.to_dicts(), [("F4","Naive verdict"),("F41","After panel + FDR"),("len","Lines")])}

<h2>6. GPU benchmarks (Tesla P100, 16 GB, shared with an LLM inference service)</h2>
{table([{"g":a,"m":b,"v":c} for a,b,c in bench], [("g","Task"),("m","Method"),("v","Result")])}

<h2>7. Data quality rules (each one found in the data, not assumed)</h2>
{table([{"r":a,"h":b} for a,b in dq], [("r","Rule"),("h","How it surfaced")])}

<h2>8. Methodology</h2>
{table([{"r":a,"h":b} for a,b in meth], [("r","Principle"),("h","Why")])}

<script>
const WD={json.dumps(WD)}, HRS=[...Array(24).keys()];
Plotly.newPlot('hm',[{{type:'heatmap',z:{json.dumps(z)},x:HRS,y:WD,text:{json.dumps(txt)},hovertemplate:'%{{y}} %{{x}}:00<br>%{{text}}<extra></extra>',colorscale:'Magma',reversescale:true,colorbar:{{title:'min'}}}}],
 {{title:'p95 arrival delay [min], trains',xaxis:{{title:'scheduled arrival hour',dtick:1}},yaxis:{{autorange:'reversed'}},margin:{{t:40,l:50,r:10,b:50}}}},{{responsive:true}});
Plotly.newPlot('hmw',[{{type:'heatmap',z:{json.dumps(w)},x:HRS,y:WD,colorscale:'Magma',reversescale:true,colorbar:{{title:'min'}}}}],
 {{title:'width of 95% CI for p95 [min]',xaxis:{{title:'scheduled arrival hour',dtick:1}},yaxis:{{autorange:'reversed'}},margin:{{t:40,l:50,r:10,b:50}}}},{{responsive:true}});
const col=v=>({{WORSE:'#b3261e',better:'#1b7f3b'}}[v]||'#888');
function forest(id,recs,title){{
  Plotly.newPlot(id,[{{type:'scatter',mode:'markers',x:recs.map(r=>r.d_p95_s),y:recs.map(r=>r.label),
    error_x:{{type:'data',symmetric:false,array:recs.map(r=>r.d_hi-r.d_p95_s),arrayminus:recs.map(r=>r.d_p95_s-r.d_lo),thickness:1.2}},
    marker:{{color:recs.map(r=>col(r.verdict)),size:8}},text:recs.map(r=>`${{r.verdict}} · p=${{(+r.p_value).toExponential(1)}}`),hovertemplate:'%{{y}}: %{{x}} s %{{text}}<extra></extra>'}}],
    {{title,xaxis:{{title:'Δ p95 [s], 2026 − 2025',zeroline:true,zerolinewidth:2}},yaxis:{{automargin:true}},height:Math.max(380,22*recs.length+90),margin:{{t:40,r:10,b:50}}}},{{responsive:true}});
}}
forest('forest',{json.dumps(rows(yh, ["label","d_p95_s","d_lo","d_hi","verdict","p_value"]))},'Rail hubs (red/green = significant after FDR and ≥ 10 s)');
forest('flines',{json.dumps(rows(yl, ["label","d_p95_s","d_lo","d_hi","verdict","p_value"]))},'Lines with stable route and volume, significant after FDR');
</script></body></html>"""
    out = OUT / "report.html"
    out.write_text(H, encoding="utf-8")
    print(f"report -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
