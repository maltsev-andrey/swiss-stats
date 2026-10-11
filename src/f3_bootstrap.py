#!/usr/bin/env python3
"""swiss-stats F3: exact bootstrap CIs for p95 on GPU.
The k-th order statistic of n iid uniform indices is floor(n*U), U ~ Beta(k+1, n-k).
So one bootstrap replicate of a sample quantile = one Beta draw + one lookup in the sorted group:
exact bootstrap distribution, O(1) per replicate instead of O(n log n).
"""
import math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f2_aggregates as eng            # CUDA env, P100 guard, budget, constants
import cupy as cp
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

CUR = eng.ROOT / "curated"
D_MIN, D_MAX, D_BITS, NB, SLOTS = eng.D_MIN, eng.D_MAX, eng.D_BITS, eng.NB, eng.SLOTS
Q, B, ALPHA, SEED = 0.95, 1000, 0.05, 20261009
I_LO, I_HI = int(ALPHA / 2 * B), int((1 - ALPHA / 2) * B) - 1
CHUNK_REPL = 50_000_000            # replicates per kernel launch (memory bound)
BYTES_PER_ROW = 96
CPU_ALGO_FRACTION = 0.10           # CPU same-algorithm benchmark on 10% of groups, extrapolated
NAIVE_GROUPS, NAIVE_MIN_N, NAIVE_MAX_N = 300, 50, 5000
pool = eng.pool

KERNEL = r'''
extern "C" {
__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
struct Rng { unsigned long long s; };
__device__ __forceinline__ float unif(Rng& r) {                    // splitmix64 stream -> (0,1)
    r.s += 0x9E3779B97F4A7C15ULL;
    return ((float)(mix64(r.s) >> 40) + 0.5f) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ float normal(Rng& r) {                  // Box-Muller
    const float u1 = unif(r), u2 = unif(r);
    return sqrtf(-2.0f * logf(u1)) * cospif(2.0f * u2);
}
__device__ float gamma_ge1(float a, Rng& r) {                      // Marsaglia-Tsang, a >= 1
    const float d = a - 1.0f / 3.0f, c = rsqrtf(9.0f * d);
    for (;;) {
        float x, v;
        do { x = normal(r); v = 1.0f + c * x; } while (v <= 0.0f);
        v = v * v * v;
        const float u = unif(r), x2 = x * x;
        if (u < 1.0f - 0.0331f * x2 * x2) return d * v;
        if (logf(u) < 0.5f * x2 + d * (1.0f - v + logf(v))) return d * v;
    }
}
// one thread = one bootstrap replicate: local index of order statistic k (0-based) in a group of n
__global__ void boot_index(const int* __restrict__ n, const int* __restrict__ k, int* __restrict__ out,
                           int G, int B, long long base, unsigned long long seed) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= (long long)G * B) return;
    const int g = (int)(t / B);
    const int ng = n[g], kg = k[g];
    Rng r; r.s = mix64(seed ^ mix64((unsigned long long)(base + t)));
    const float x = gamma_ge1((float)(kg + 1), r);
    const float y = gamma_ge1((float)(ng - kg), r);
    const int idx = (int)((x / (x + y)) * (float)ng);
    out[t] = idx < ng ? idx : ng - 1;
}
}
'''


def sync():
    cp.cuda.Stream.null.synchronize()


def launch(kern, n32, k32, base):
    G = int(n32.size)
    idx = cp.empty(G * B, cp.int32)
    kern((((G * B) + 255) // 256,), (256,),
         (n32, k32, idx, np.int32(G), np.int32(B), np.int64(base), np.uint64(SEED)))
    return idx.reshape(G, B)


def sorted_groups(keys, delay):
    """keys int64, delay int16 (GPU) -> uniq keys, group-contiguous sorted values, start, n."""
    uniq, gid = cp.unique(keys, return_inverse=True)
    packed = (gid.astype(cp.int64) << D_BITS) | (delay.astype(cp.int64) - D_MIN)
    del gid
    packed.sort()
    g = packed >> D_BITS
    v = ((packed & ((1 << D_BITS) - 1)) + D_MIN).astype(cp.int16)
    del packed
    n = cp.bincount(g, minlength=int(uniq.size)).astype(cp.int64)
    del g
    return uniq, v, cp.cumsum(n) - n, n


def boot_ci(kern, v, start, n, base):
    """Exact bootstrap percentile CI of the Q-quantile for every group."""
    G = int(n.size)
    k = cp.floor(Q * (n - 1)).astype(cp.int32)
    n32 = n.astype(cp.int32)
    p_hat = v[start + k]
    lo, hi = cp.empty(G, cp.int16), cp.empty(G, cp.int16)
    se = cp.empty(G, cp.float32)
    gb = max(1, CHUNK_REPL // B)
    for g0 in range(0, G, gb):
        g1 = min(G, g0 + gb)
        idx = launch(kern, n32[g0:g1], k[g0:g1], base + g0 * B)
        vals = v[start[g0:g1, None] + idx]
        del idx
        vals.sort(axis=1)
        lo[g0:g1], hi[g0:g1] = vals[:, I_LO], vals[:, I_HI]
        se[g0:g1] = vals.astype(cp.float32).std(axis=1)
        del vals
    return p_hat, lo, hi, se


def hist_boot_ci(kern, hist2d, base):
    """Same CI from histograms (global groups): order index -> value via search in global cumsum."""
    nrow = hist2d.sum(axis=1)
    keep = cp.nonzero(nrow)[0]
    h, n = hist2d[keep], nrow[keep]
    flat = cp.cumsum(h.ravel())
    off = cp.cumsum(n) - n
    k = cp.floor(Q * (n - 1)).astype(cp.int64)
    value_at = lambda j: (cp.searchsorted(flat, (off[:, None] + j).ravel(), side="right") % NB + D_MIN) \
        .reshape(int(n.size), -1).astype(cp.int16)
    p_hat = value_at(k[:, None])[:, 0]
    vals = value_at(launch(kern, n.astype(cp.int32), k.astype(cp.int32), base).astype(cp.int64))
    vals.sort(axis=1)
    return keep, n, p_hat, vals[:, I_LO], vals[:, I_HI], vals.astype(cp.float32).std(axis=1)


def cpu_same_algo(v, start, n, frac, rng):
    """NumPy version of the Beta trick on a fraction of groups (benchmark)."""
    G = len(n)
    sel = rng.choice(G, size=max(1, int(G * frac)), replace=False)
    t = time.time()
    for c0 in range(0, len(sel), 20_000):
        s = sel[c0:c0 + 20_000]
        nn, kk = n[s], np.floor(Q * (n[s] - 1)).astype(np.int64)
        u = rng.beta(kk[:, None] + 1, (nn - kk)[:, None], size=(len(s), B))
        idx = np.minimum((u * nn[:, None]).astype(np.int64), nn[:, None] - 1)
        vals = np.sort(v[start[s][:, None] + idx], axis=1)
        _ = vals[:, I_LO], vals[:, I_HI]
    return time.time() - t, len(sel)


def cpu_naive(v, start, n, rng):
    """Honest bootstrap (resample + partition) on a sample of groups: benchmark + validation."""
    cand = np.nonzero((n >= NAIVE_MIN_N) & (n <= NAIVE_MAX_N))[0]
    sel = rng.choice(cand, size=min(NAIVE_GROUPS, len(cand)), replace=False)
    lo, hi, work = np.empty(len(sel)), np.empty(len(sel)), 0
    t = time.time()
    for i, g in enumerate(sel):
        x = v[start[g]:start[g] + n[g]]
        k = int(math.floor(Q * (n[g] - 1)))
        res = rng.choice(x, size=(B, n[g]), replace=True)
        q = np.sort(np.partition(res, k, axis=1)[:, k])
        lo[i], hi[i] = q[I_LO], q[I_HI]
        work += n[g] * B
    return time.time() - t, sel, lo, hi, work


def main():
    t0 = time.time()
    df = pl.read_parquet(str(CUR / "arrivals" / "*.parquet"), columns=["date", "station_uic", "pcode", "hour", "wd", "delay"])
    dprod = pl.read_parquet(CUR / "dim_product.parquet")
    names = pl.read_parquet(CUR / "dim_station.parquet")
    st, pc = df["station_uic"].to_numpy(), df["pcode"].to_numpy()
    hr, wd = df["hour"].to_numpy(), df["wd"].to_numpy()
    dl = np.clip(df["delay"].to_numpy(), D_MIN, D_MAX).astype(np.int16)
    N = len(dl)
    n_days = df["date"].n_unique()
    del df
    t_load = time.time() - t0

    budget, free = eng.acquire_budget()
    print(f"\nGPU: {eng.GPU} | free {free / 2**30:.2f} GB | budget {budget / 2**30:.2f} GB | "
          f"{N:,} arrivals | Q={Q} B={B} | load {t_load:.2f}s")

    t = time.time()
    kern = cp.RawModule(code=KERNEL).get_function("boot_index")
    eng.warmup()
    _ = boot_ci(kern, cp.arange(64, dtype=cp.int16), cp.array([0, 32], cp.int64), cp.array([32, 32], cp.int64), 0)
    sync(); pool.free_all_blocks()
    t_jit = time.time() - t

    hist_bytes = 2 * SLOTS * NB * 8
    parts = max(1, math.ceil(1.2 * N / max(1_000_000, (budget - hist_bytes) // BYTES_PER_ROW)))
    while True:
        try:
            print(f"GPU run: {parts} partition(s)")
            out = {"sph": [], "sp": []}
            hist = cp.zeros(SLOTS * NB, cp.int64)
            t_sort = t_boot = 0.0
            bench, n_repl = None, 0
            host = {"st": st, "pc": pc, "hr": hr, "wd": wd, "dl": dl}
            t_host = time.time()
            bounds = eng.order_by_partition(host, parts)
            st, pc, hr, wd, dl = (host[k] for k in ("st", "pc", "hr", "wd", "dl"))
            t_host = time.time() - t_host
            sync(); t_all = time.time()
            for p in range(parts):
                lo, hi = int(bounds[p]), int(bounds[p + 1])
                g_st = cp.asarray(st[lo:hi]).astype(cp.int64)
                g_pc = cp.asarray(pc[lo:hi]).astype(cp.int64)
                g_hr = cp.asarray(hr[lo:hi]).astype(cp.int64)
                g_wd = cp.asarray(wd[lo:hi]).astype(cp.int64)
                g_dl = cp.asarray(dl[lo:hi])
                hist += cp.bincount((g_pc * 256 + g_wd * 32 + g_hr) * NB + (g_dl.astype(cp.int64) - D_MIN),
                                    minlength=SLOTS * NB)
                del g_wd
                for j, (name, keyfn) in enumerate((("sph", lambda: g_st * 512 + g_pc * 32 + g_hr),
                                                   ("sp", lambda: g_st * 16 + g_pc))):
                    sync(); t = time.time()
                    uniq, v, start, n = sorted_groups(keyfn(), g_dl)
                    sync(); t_sort += time.time() - t; t = time.time()
                    p_hat, lo, hi, se = boot_ci(kern, v, start, n, base=(p * 2 + j) << 40)
                    sync(); t_boot += time.time() - t
                    n_repl += int(n.size) * B
                    out[name].append(pl.DataFrame({"key": cp.asnumpy(uniq), "n": cp.asnumpy(n),
                                                   "p95_s": cp.asnumpy(p_hat), "lo_s": cp.asnumpy(lo),
                                                   "hi_s": cp.asnumpy(hi), "se_s": cp.asnumpy(se)}))
                    if name == "sph" and p == 0:
                        bench = (cp.asnumpy(v), cp.asnumpy(start), cp.asnumpy(n), cp.asnumpy(lo), cp.asnumpy(hi))
                    del uniq, v, start, n, p_hat, lo, hi, se
                del g_st, g_pc, g_hr, g_dl
                pool.free_all_blocks()
            sync(); t = time.time()
            keep, hn, hp, hlo, hhi, hse = hist_boot_ci(kern, hist.reshape(SLOTS, NB), base=1 << 50)
            sync(); t_hist = time.time() - t
            pwh = pl.DataFrame({"key": cp.asnumpy(keep), "n": cp.asnumpy(hn), "p95_s": cp.asnumpy(hp),
                                "lo_s": cp.asnumpy(hlo), "hi_s": cp.asnumpy(hhi), "se_s": cp.asnumpy(hse)})
            del hist
            pool.free_all_blocks()
            t_gpu = time.time() - t_all
            break
        except cp.cuda.memory.OutOfMemoryError:
            pool.free_all_blocks()
            parts *= 2
            print(f"OOM -> retry with {parts} partitions")

    # ---- CPU benchmarks + validation on partition 0 of the sph grouping ----
    v0, s0, n0, lo0, hi0 = bench
    rng = np.random.default_rng(SEED)
    t_cpu_algo, g_cpu = cpu_same_algo(v0, s0, n0, CPU_ALGO_FRACTION, rng)
    G_sph = sum(d.height for d in out["sph"])
    t_cpu_algo_full = t_cpu_algo * G_sph / g_cpu
    t_naive, sel, nlo, nhi, work = cpu_naive(v0, s0, n0, rng)
    t_naive_full = t_naive * (N * B) / work
    width = np.maximum(hi0[sel].astype(float) - lo0[sel], 1.0)
    d_lo, d_hi = np.abs(lo0[sel] - nlo), np.abs(hi0[sel] - nhi)
    tol = np.maximum(5.0, 0.15 * width)
    ok = 100.0 * np.mean((d_lo <= tol) & (d_hi <= tol))

    # ---- decode + save ----
    pmap = dprod.select(pl.col("pcode").cast(pl.Int64), "product")
    ci_cols = [(pl.col("hi_s") - pl.col("lo_s")).alias("ci_width_s")]
    sph = (pl.concat(out["sph"]).with_columns((pl.col("key") // 512).cast(pl.Int32).alias("station_uic"),
                                              ((pl.col("key") // 32) % 16).alias("pcode"),
                                              (pl.col("key") % 32).alias("hour"), *ci_cols)
           .join(pmap, on="pcode").join(names, on="station_uic", how="left"))
    sp = (pl.concat(out["sp"]).with_columns((pl.col("key") // 16).cast(pl.Int32).alias("station_uic"),
                                            (pl.col("key") % 16).alias("pcode"), *ci_cols)
          .join(pmap, on="pcode").join(names, on="station_uic", how="left"))
    pwh = (pwh.with_columns((pl.col("key") // 256).alias("pcode"), ((pl.col("key") // 32) % 8).alias("wd"),
                            (pl.col("key") % 32).alias("hour"), *ci_cols).join(pmap, on="pcode"))
    sph.write_parquet(eng.AGG / "ci_station_product_hour.parquet")
    sp.write_parquet(eng.AGG / "ci_station_product.parquet")
    pwh.write_parquet(eng.AGG / "ci_product_weekday_hour.parquet")

    fmt = lambda s: f"{s:,.1f} s" if s < 3600 else f"{s / 3600:,.1f} h"
    with pl.Config(tbl_rows=40, tbl_cols=12, fmt_str_lengths=40, thousands_separator=True,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print("\n=== Benchmark: p95 bootstrap CI, all station x product x hour groups ===")
        print(pl.DataFrame([
            ("CPU naive resample (extrapolated)", fmt(t_naive_full), f"{NAIVE_GROUPS} groups measured"),
            ("CPU NumPy Beta trick (extrapolated)", fmt(t_cpu_algo_full), f"{g_cpu:,} groups measured"),
            ("GPU kernel Beta trick (measured)", fmt(t_boot), f"{n_repl / 1e6:,.0f}M replicates"),
        ], schema=["method", "time", "basis"], orient="row"))
        print(f"speedup: algorithm {t_naive_full / t_cpu_algo_full:,.0f}x | GPU over CPU same algo "
              f"{t_cpu_algo_full / t_boot:,.1f}x | total {t_naive_full / t_boot:,.0f}x")
        print(f"GPU wall: {t_gpu:.2f}s (sort {t_sort:.2f}s, bootstrap {t_boot:.2f}s, hist CI {t_hist:.2f}s) | "
              f"host reorder {t_host:.2f}s | JIT {t_jit:.2f}s | partitions {parts}")
        print(f"validation vs naive on {len(sel)} groups: within tol {ok:.1f}% | "
              f"median |d lo| {np.median(d_lo):.1f}s, |d hi| {np.median(d_hi):.1f}s, median CI width {np.median(width):.0f}s\n")

        hubs = sp.filter(pl.col("product") == "Zug").sort("n", descending=True).head(15)
        print("=== Rail hubs: p95 with 95% CI (trains) ===")
        print(hubs.select("name", "n", "p95_s", "lo_s", "hi_s", "ci_width_s", pl.col("se_s").round(1)))
        wide = sph.filter((pl.col("product") == "Zug") & (pl.col("n") >= 30)).sort("ci_width_s", descending=True).head(10)
        print("\n=== Least certain station x hour cells (trains, n >= 30) ===")
        print(wide.select("name", "hour", "n", "p95_s", "lo_s", "hi_s", "ci_width_s"))

    z = pwh.filter((pl.col("product") == "Zug") & (pl.col("n") >= 30) & (pl.col("hour") < 24))
    p95, wid = np.full((7, 24), np.nan), np.full((7, 24), np.nan)
    for r in z.iter_rows(named=True):
        p95[r["wd"], r["hour"]] = r["p95_s"] / 60
        wid[r["wd"], r["hour"]] = r["ci_width_s"] / 60
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5))
    for ax, arr, title, lab in ((axes[0], p95, "p95 arrival delay [min]", "p95, min"),
                                (axes[1], wid, "width of 95% bootstrap CI for p95 [min]", "CI width, min")):
        im = ax.imshow(arr, aspect="auto", cmap="magma_r")
        ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xticks(range(24))
        ax.set_title(f"Swiss trains: {title}, {z['n'].sum() / 1e6:.2f}M train arrivals, {n_days} days")
        fig.colorbar(im, ax=ax, label=lab)
    axes[1].set_xlabel("scheduled arrival hour (Source: opentransportdata.swiss)")
    fig.tight_layout()
    fig.savefig(eng.OUT / "f3_zug_p95_ci_heatmap.png", dpi=130)
    print(f"\nheatmap -> {eng.OUT / 'f3_zug_p95_ci_heatmap.png'}")


if __name__ == "__main__":
    main()
