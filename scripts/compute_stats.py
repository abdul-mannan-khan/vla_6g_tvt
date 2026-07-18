#!/usr/bin/env python3
"""
Statistical analysis for VLA-6G UAV Relay System evaluation results.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("/home/it-services/ros2_ws/src/vla_6g_tvt/results")

FILES = {
    "main": RESULTS_DIR / "evaluation_20260213_100703.json",
    "sac": RESULTS_DIR / "sac_baseline_20260213_184337.json",
    "td3": RESULTS_DIR / "td3_baseline_20260213_184824.json",
    "mlp_2k": RESULTS_DIR / "mlp_2k_baseline_20260213_182549.json",
    "mlp_8k": RESULTS_DIR / "mlp_baseline_20260213_153032.json",
    "ppo": RESULTS_DIR / "drl_baseline_20260213_153943.json",
}

WEIGHT_CONFIGS = {
    "default":          {"w1": 0.60, "w2": 0.30, "w3": 0.10},
    "throughput_heavy":  {"w1": 0.80, "w2": 0.10, "w3": 0.10},
    "fairness_heavy":    {"w1": 0.30, "w2": 0.60, "w3": 0.10},
    "coverage_heavy":    {"w1": 0.30, "w2": 0.10, "w3": 0.60},
    "equal":             {"w1": 0.33, "w2": 0.33, "w3": 0.34},
}

ALL_METHODS = ["vla", "analytical", "optimized", "random", "static",
               "sac", "td3", "mlp_2k", "mlp_8k", "ppo"]


def load_data():
    with open(FILES["main"]) as f:
        main_data = json.load(f)

    methods_main = {}
    for mn in ["vla", "analytical", "optimized", "random", "static"]:
        entries = sorted(
            [r for r in main_data["results"] if r["method"] == mn],
            key=lambda r: r["scenario_id"],
        )
        methods_main[mn] = entries

    throughputs = {}
    for mn, entries in methods_main.items():
        throughputs[mn] = np.array([e["total_throughput"] for e in entries])

    for key in ["sac", "td3", "mlp_2k", "mlp_8k", "ppo"]:
        with open(FILES[key]) as f:
            d = json.load(f)
        ps = sorted(d["per_scenario"], key=lambda r: r["scenario_id"])
        throughputs[key] = np.array([r["throughput"] for r in ps])

    return throughputs, methods_main


def bootstrap_ci(data, n_resamples=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(data)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = data[rng.randint(0, n, size=n)]
        means[i] = np.mean(sample)
    alpha = (1 - ci) / 2
    lo = np.percentile(means, 100 * alpha)
    hi = np.percentile(means, 100 * (1 - alpha))
    return float(np.mean(data)), float(lo), float(hi)


def cohens_d(a, b):
    diff = a - b
    return float(np.mean(diff) / np.std(diff, ddof=1))


def interpret_d(d_val):
    d_abs = abs(d_val)
    if d_abs < 0.2: return "negligible"
    elif d_abs < 0.5: return "small"
    elif d_abs < 0.8: return "medium"
    else: return "large"


def paired_ttest(a, b):
    t_stat, p_val = stats.ttest_rel(a, b)
    return float(t_stat), float(p_val)


def composite_score(throughput, fairness, coverage, w1, w2, w3):
    return w1 * throughput + w2 * (fairness * 100) + w3 * (coverage * 100)


def section_header(title):
    print("")
    print("+" + "-" * 63 + "+")
    print("| " + title.ljust(62) + "|")
    print("+" + "-" * 63 + "+")


def main():
    print("=" * 72)
    print("  VLA-6G Statistical Analysis")
    print("=" * 72)

    throughputs, methods_main = load_data()
    output = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_bootstrap": 10000,
            "n_scenarios": 100,
            "methods": ALL_METHODS,
        }
    }

    # --- 1. Bootstrap 95% CIs ---
    section_header("1. Bootstrap 95% CIs for mean throughput (Mbps)")
    print("  {:<14} {:>9} {:>11} {:>12}".format("Method", "Mean", "CI Low", "CI High"))
    print("  " + "-" * 14 + " " + "-" * 9 + " " + "-" * 11 + " " + "-" * 12)

    boot = {}
    for name in ALL_METHODS:
        mean, lo, hi = bootstrap_ci(throughputs[name])
        boot[name] = {"mean": mean, "ci_low": lo, "ci_high": hi}
        print("  {:<14} {:9.2f} [{:9.2f}, {:9.2f}]".format(name, mean, lo, hi))
    output["bootstrap_ci"] = boot

    # --- 2. Cohen's d ---
    pairs_d = [("vla","analytical"), ("vla","sac"), ("mlp_8k","vla"), ("vla","ppo")]
    section_header("2. Cohen's d effect sizes")
    print("  {:<25} {:>8} {:<15}".format("Comparison", "d", "Interpretation"))
    print("  " + "-" * 25 + " " + "-" * 8 + " " + "-" * 15)

    cd = {}
    for a, b in pairs_d:
        d = cohens_d(throughputs[a], throughputs[b])
        interp = interpret_d(d)
        label = a + " vs " + b
        cd[label] = {"d": d, "interpretation": interp}
        print("  {:<25} {:8.4f}   {}".format(label, d, interp))
    output["cohens_d"] = cd

    # --- 3. Bonferroni-corrected paired t-tests ---
    pairs_t = [("vla","analytical"), ("vla","sac"), ("vla","td3"),
               ("vla","ppo"), ("vla","mlp_2k"), ("vla","mlp_8k")]
    nc = len(pairs_t)
    section_header("3. Bonferroni-corrected paired t-tests ({} comps)".format(nc))
    print("  {:<25} {:>9} {:>12} {:>12} {:<6}".format(
        "Comparison", "t-stat", "p (raw)", "p (Bonf)", "Sig?"))
    print("  " + "-" * 25 + " " + "-" * 9 + " " + "-" * 12 + " " + "-" * 12 + " " + "-" * 6)

    tt = {}
    for a, b in pairs_t:
        t_stat, p_raw = paired_ttest(throughputs[a], throughputs[b])
        p_bonf = min(p_raw * nc, 1.0)
        if p_bonf < 0.001: sig = "***"
        elif p_bonf < 0.01: sig = "**"
        elif p_bonf < 0.05: sig = "*"
        else: sig = "ns"
        label = a + " vs " + b
        tt[label] = {"t_statistic": t_stat, "p_raw": p_raw,
                     "p_bonferroni": p_bonf, "significant_at_0.05": p_bonf < 0.05}
        print("  {:<25} {:9.4f} {:12.6e} {:12.6e} {}".format(
            label, t_stat, p_raw, p_bonf, sig))
    output["bonferroni_ttests"] = tt

    # --- 4. MLP-2K vs MLP-8K ---
    section_header("4. MLP-2K vs MLP-8K paired t-test")
    t_s, p_v = paired_ttest(throughputs["mlp_2k"], throughputs["mlp_8k"])
    d_m = cohens_d(throughputs["mlp_2k"], throughputs["mlp_8k"])
    m2k = float(np.mean(throughputs["mlp_2k"]))
    m8k = float(np.mean(throughputs["mlp_8k"]))
    diff = m2k - m8k
    sig_m = p_v < 0.05

    mlp_cmp = {
        "mlp_2k_mean": m2k, "mlp_8k_mean": m8k, "mean_difference": diff,
        "t_statistic": t_s, "p_value": p_v, "cohens_d": d_m,
        "cohens_d_interpretation": interpret_d(d_m),
        "significant_at_0.05": sig_m,
    }
    output["mlp_2k_vs_8k"] = mlp_cmp

    print("  MLP-2K mean throughput:    {:.2f} Mbps".format(m2k))
    print("  MLP-8K mean throughput:    {:.2f} Mbps".format(m8k))
    print("  Mean difference:           {:+.2f} Mbps".format(diff))
    print("  t-statistic:               {:.4f}".format(t_s))
    print("  p-value:                   {:.6e}".format(p_v))
    print("  Cohen d:                   {:.4f} ({})".format(d_m, interpret_d(d_m)))
    print("  Significant at alpha=0.05: {}".format("YES" if sig_m else "NO"))

    # --- 5. Weight sensitivity ---
    section_header("5. Weight sensitivity analysis")

    mm = {}
    for mn in ["vla", "analytical", "optimized", "random", "static"]:
        ents = sorted(methods_main[mn], key=lambda r: r["scenario_id"])
        mm[mn] = {
            "throughput": np.array([e["total_throughput"] for e in ents]),
            "fairness": np.array([e["fairness_index"] for e in ents]),
            "coverage": np.array([e["coverage_rate"] for e in ents]),
        }
    for key in ["sac", "td3", "mlp_2k", "mlp_8k", "ppo"]:
        with open(FILES[key]) as f:
            d = json.load(f)
        ps = sorted(d["per_scenario"], key=lambda r: r["scenario_id"])
        mm[key] = {
            "throughput": np.array([r["throughput"] for r in ps]),
            "fairness": np.array([r["fairness"] for r in ps]),
            "coverage": np.array([r["coverage_rate"] for r in ps]),
        }

    sens = {}
    for cn, wts in WEIGHT_CONFIGS.items():
        w1, w2, w3 = wts["w1"], wts["w2"], wts["w3"]
        print("")
        print("  Config: {} (w1={}, w2={}, w3={})".format(cn, w1, w2, w3))
        print("  {:<6} {:<14} {:>15} {:>10}".format("Rank", "Method", "Mean Composite", "Std"))
        print("  " + "-" * 6 + " " + "-" * 14 + " " + "-" * 15 + " " + "-" * 10)

        scores = {}
        for name in ALL_METHODS:
            m = mm[name]
            c = composite_score(m["throughput"], m["fairness"], m["coverage"], w1, w2, w3)
            scores[name] = {"mean": float(np.mean(c)), "std": float(np.std(c, ddof=1))}

        ranking = sorted(scores.items(), key=lambda x: x[1]["mean"], reverse=True)
        ranked = []
        for rk, (name, vals) in enumerate(ranking, 1):
            ranked.append({"rank": rk, "method": name,
                           "mean_composite": vals["mean"], "std": vals["std"]})
            print("  {:<6} {:<14} {:15.2f} {:10.2f}".format(rk, name, vals["mean"], vals["std"]))
        sens[cn] = {"weights": wts, "ranking": ranked}

    output["weight_sensitivity"] = sens

    # Summary table
    print("")
    print("=" * 72)
    print("  Summary: Method ranking across weight configurations")
    print("=" * 72)
    hdr = "  {:<14}".format("Method")
    for cfg in WEIGHT_CONFIGS:
        hdr += " {:>16}".format(cfg)
    print(hdr)
    ln = "  " + "-" * 14
    for _ in WEIGHT_CONFIGS:
        ln += " " + "-" * 16
    print(ln)

    for name in ALL_METHODS:
        row = "  {:<14}".format(name)
        for cfg in WEIGHT_CONFIGS:
            rl = sens[cfg]["ranking"]
            rk = [r["rank"] for r in rl if r["method"] == name][0]
            row += " {:>16}".format(rk)
        print(row)

    # Save
    out_path = RESULTS_DIR / "statistical_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print("")
    print("  Results saved to: {}".format(out_path))
    print("=" * 72)


if __name__ == "__main__":
    main()
