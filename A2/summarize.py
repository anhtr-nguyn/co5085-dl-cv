"""Collect A2 results: training logs of the MonoCoP clone + analysis JSONs -> tables and figures.

    python A2/summarize.py [--monocop-root ~/MonoCoP]

Outputs
    A2/outputs/results.json     per-run val AP for every epoch, best/last summary, analysis summary
    A2/outputs/results.md       markdown tables for the landing page
    A2/outputs/figs/*.png       curves, branch ablation, error by distance, KITTI-C
    report_tex/a2/tables/*.tex  LaTeX tables used by the report
Only runs that finished all 250 epochs enter the tables; unfinished runs are listed as such.
"""
import argparse
import glob
import json
import math
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A2 = os.path.join(ROOT, "A2")
OUT = os.path.join(A2, "outputs")
FIGS = os.path.join(OUT, "figs")
ANALYSIS = os.path.join(OUT, "analysis")
TABLES = os.path.join(ROOT, "report_tex", "a2", "tables")
MAX_EPOCH = 250

# (group, run-dir glob relative to MonoCoP/outputs). The 2026-10-01 run kept the ResNet stem frozen
# (upstream freeze_at=0); every other run trains it (freeze_at=-1), so it is reported separately.
RUNS = [
    ("monocop", "monocop_car/monocop_car444_2026-10-20_*"),
    ("monocop", "monocop_car/monocop_car444_2026-10-21_*"),
    ("monocop", "monocop_car/monocop_car444_2026-10-22_*"),
    ("monocop", "monocop_car/monocop_car42_*"),
    ("monocop", "monocop_car/monocop_car100_*"),
    ("parallel", "monocop_car_parallel/monocop_car_parallel444_*"),
    ("parallel", "monocop_car_parallel/monocop_car_parallel42_*"),
    ("parallel", "monocop_car_parallel/monocop_car_parallel100_*"),
    ("monocop_frozen_stem", "monocop_car/monocop_car444_2026-10-01_08-54-17_*"),
]
GROUP_NAME = {"monocop": "MonoCoP (CoP + GS)", "parallel": "Baseline song song",
              "monocop_frozen_stem": "MonoCoP, stem đóng băng"}
COLOR = {"monocop": "tab:red", "parallel": "tab:blue", "monocop_frozen_stem": "tab:gray"}
LEVELS = ("easy", "mod", "hard")
CORRUPTION_NAME = {"brightness_1": "Brightness", "contrast_1": "Contrast", "saturate_1": "Saturate",
                   "defocus_blur_1": "Defocus blur", "glass_blur_1": "Glass blur", "motion_blur_1": "Motion blur",
                   "gaussian_noise_1": "Gaussian noise", "shot_noise_1": "Shot noise",
                   "impulse_noise_1": "Impulse noise", "pixelate_1": "Pixelate", "fog_1": "Fog",
                   "frost_1": "Frost", "snow_1": "Snow"}
BLOCK = re.compile(r"Car AP_R40@0\.70, (0\.70|0\.50), \1:\nbbox AP:(.*)\nbev  AP:(.*)\n3d   AP:(.*)\n")


def parse_log(path):
    text = open(path).read()
    stamps = re.findall(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", text, re.M)
    epochs = {}
    parts = re.split(r"INFO  Test Epoch (\d+)\n", text)
    for e, body in zip(parts[1::2], parts[2::2]):
        rec = {}
        for iou, bbox, bev, d3 in BLOCK.findall(body)[:2]:
            key = "iou70" if iou == "0.70" else "iou50"
            rec[key] = {"bbox": [float(x) for x in bbox.split(",")],
                        "bev": [float(x) for x in bev.split(",")],
                        "3d": [float(x) for x in d3.split(",")]}
        if len(rec) == 2:
            epochs[int(e)] = rec
    hours = None
    m = re.search(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*Test Epoch %d\n" % MAX_EPOCH, text, re.M)
    if stamps and m:
        fmt = "%Y-%m-%d %H:%M:%S"
        hours = (datetime.strptime(m.group(1), fmt) - datetime.strptime(stamps[0], fmt)).total_seconds() / 3600
    return epochs, hours


def collect_runs(monocop_root):
    runs = []
    for group, pattern in RUNS:
        dirs = sorted(glob.glob(os.path.join(monocop_root, "outputs", pattern)))
        for d in dirs:
            logs = sorted(glob.glob(os.path.join(d, "train.log.*")))
            if not logs:
                continue
            epochs, hours = parse_log(logs[-1])
            if not epochs:
                continue
            seed = re.search(r"(?:car|parallel)(\d+)_", os.path.basename(d)).group(1)
            best = max(epochs, key=lambda e: epochs[e]["iou70"]["3d"][1])
            last = max(epochs)
            runs.append({"group": group, "name": os.path.basename(d), "seed": int(seed),
                         "complete": last >= MAX_EPOCH, "n_epochs": last, "train_hours": hours,
                         "best_epoch": best, "best": epochs[best], "last": epochs[last],
                         "curve_mod3d": {e: epochs[e]["iou70"]["3d"][1] for e in sorted(epochs)},
                         "epochs": epochs})
    return runs


def stats(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()), "n": len(a)}


def summarize_groups(runs):
    out = {}
    for group in GROUP_NAME:
        rs = [r for r in runs if r["group"] == group and r["complete"]]
        if not rs:
            continue
        g = {"n": len(rs), "runs": [r["name"] for r in rs]}
        for which in ("best", "last"):
            for iou in ("iou70", "iou50"):
                for kind in ("3d", "bev"):
                    for i, lvl in enumerate(LEVELS):
                        g[f"{which}_{iou}_{kind}_{lvl}"] = stats([r[which][iou][kind][i] for r in rs])
        g["best_epoch"] = [r["best_epoch"] for r in rs]
        hours = [r["train_hours"] for r in rs if r["train_hours"]]
        g["train_hours"] = stats(hours) if hours else None
        out[group] = g
    return out


def load_analysis(runs, analysis_dir):
    res = {}
    for r in runs:
        p = os.path.join(analysis_dir, r["name"] + ".json")
        if os.path.exists(p):
            res[r["name"]] = json.load(open(p))
    return res


def run_date(name):
    return re.search(r"_(\d{4}-\d\d-\d\d)_", name).group(1)


def run_label(name):
    seed = re.search(r"(?:car|parallel)(\d+)_", name).group(1)
    return f"seed {seed}, {run_date(name)}"


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pm(s, nd=2):
    return f"${s['mean']:.{nd}f} \\pm {s['std']:.{nd}f}$" if s["n"] > 1 else f"${s['mean']:.{nd}f}$"


def placeholder_table():
    return "\\begin{tabular}{l}\n\\todo{chờ kết quả phân tích (A2/summarize.py)}\n\\end{tabular}\n"


def placeholder_fig(name):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, "Chờ kết quả phân tích (A2/summarize.py)", ha="center", va="center", fontsize=14)
    ax.axis("off")
    fig.savefig(os.path.join(FIGS, name), dpi=100)
    plt.close(fig)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------- tables
def table_main(groups):
    rows = []
    for group in ("parallel", "monocop"):
        if group not in groups:
            continue
        g = groups[group]
        for which, label in (("best", "best (val)"), ("last", "epoch 250")):
            cells = [pm(g[f"{which}_iou70_3d_{l}"]) for l in LEVELS] + [pm(g[f"{which}_iou70_bev_{l}"]) for l in LEVELS]
            med = fmt(g[f"{which}_iou70_3d_mod"]["median"])
            rows.append(f"{GROUP_NAME[group]} & {g['n']} & {label} & " + " & ".join(cells) + f" & {med} \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    return ("\\begin{tabular}{lrl" + "r" * 7 + "}\n\\toprule\n"
            "& & & \\multicolumn{3}{c}{AP$_{3D}$|R40} & \\multicolumn{3}{c}{AP$_{BEV}$|R40} & Mod. \\\\\n"
            "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\n"
            "Biến thể & Run & Checkpoint & Easy & Mod. & Hard & Easy & Mod. & Hard & median \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def table_runs(runs):
    rows = []
    for r in runs:
        b3, l3 = r["best"]["iou70"]["3d"], r["last"]["iou70"]["3d"]
        b5 = r["best"]["iou50"]["3d"]
        status = "" if r["complete"] else f" (đang chạy, {r['n_epochs']} epoch)"
        rows.append(f"{GROUP_NAME[r['group']]}{status} & {r['seed']} & {run_date(r['name'])} & "
                    f"{r['best_epoch']} & " + " / ".join(fmt(x) for x in b3) + " & "
                    + " / ".join(fmt(x) for x in l3) + " & " + " / ".join(fmt(x) for x in b5)
                    + f" & {fmt(r['train_hours'], 1)} \\\\")
    return ("\\begin{tabular}{llllllll}\n\\toprule\n"
            "Biến thể & Seed & Ngày & Best ep. & AP$_{3D}$ best (E/M/H) & AP$_{3D}$ ep. 250 & AP$_{3D}$@0.5 best & Giờ \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def branch_rows(analysis, runs):
    """Mean over MonoCoP checkpoints of AP and chain fraction for each selection rule (clean val)."""
    names = [r["name"] for r in runs if r["group"] == "monocop" and r["complete"] and r["name"] in analysis
             and "clean" in analysis[r["name"]]["sets"]]
    if not names:
        return None, names
    modes = {}
    for n in names:
        for m, v in analysis[n]["sets"]["clean"]["modes"].items():
            modes.setdefault(m, []).append(v)
    order = ["gs", "chain", "parallel", "min_unc"] + [f"tau@p{q}" for q in range(10, 100, 10)]
    rows = []
    for m in order:
        if m not in modes or len(modes[m]) != len(names):
            continue
        ap = np.array([v["ap"]["iou70"]["3d"] for v in modes[m]])
        frac = np.mean([v["chain_frac"] for v in modes[m]])
        rows.append({"mode": m, "chain_frac": float(frac), "ap_mean": ap.mean(0).tolist(),
                     "ap_std": ap.std(0, ddof=1).tolist() if len(ap) > 1 else [0, 0, 0]})
    return rows, names


MODE_NAME = {"gs": "GS, $\\tau=5$ (mặc định)", "chain": "Luôn chain", "parallel": "Luôn parallel",
             "min_unc": "Nhánh có uncertainty thấp hơn"}


def table_branch(rows, names):
    body = []
    for r in rows:
        name = MODE_NAME.get(r["mode"], r["mode"].replace("tau@p", "$\\tau$ = phân vị ") + "\\%")
        body.append(f"{name} & {100 * r['chain_frac']:.1f} & "
                    + " & ".join(f"${m:.2f} \\pm {s:.2f}$" for m, s in zip(r["ap_mean"], r["ap_std"])) + " \\\\")
        if r["mode"] == "min_unc":
            body.append("\\midrule")
    return ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Quy tắc chọn nhánh độ sâu & Chain (\\%) & Easy & Mod. & Hard \\\\\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"), len(names)


def distance_summary(analysis, runs):
    """Per group: mean over checkpoints of per-bin MAE (clean val)."""
    out = {}
    for group, mode in (("parallel", "parallel_baseline"), ("monocop", "gs")):
        recs = [analysis[r["name"]]["sets"]["clean"]["error_by_distance"] for r in runs
                if r["group"] == group and r["complete"] and r["name"] in analysis
                and "clean" in analysis[r["name"]]["sets"]]
        if not recs:
            continue
        g = {"n_ckpt": len(recs), "bins": recs[0]["bins_m"],
             "recall": np.mean([[m / n if n else np.nan for m, n in zip(x["n_matched"], x["n_gt"])] for x in recs], 0).tolist()}
        for q in ("depth", "size", "angle"):
            g[q] = np.nanmean([[np.nan if v is None else v for v in x["errors"][mode][q]["per_bin"]] for x in recs], 0).tolist()
            g[q + "_all"] = float(np.mean([x["errors"][mode][q]["all"] for x in recs]))
            g[q + "_median"] = np.nanmean([[np.nan if v is None else v for v in x["errors"][mode][q]["per_bin_median"]] for x in recs], 0).tolist()
        if group == "monocop":
            for m in ("chain", "parallel", "oracle"):
                g["depth_" + m] = np.nanmean([[np.nan if v is None else v for v in x["errors"][m]["depth"]["per_bin"]] for x in recs], 0).tolist()
                g["depth_all_" + m] = float(np.mean([x["errors"][m]["depth"]["all"] for x in recs]))
            g["gs_picks_better_all"] = float(np.mean([x["gs_picks_better_all"] for x in recs]))
            g["gs_chain_frac"] = np.nanmean([[np.nan if v is None else v for v in x["gs_chain_frac"]] for x in recs], 0).tolist()
        out[group] = g
    return out


def bin_labels(bins):
    return [f"{a}--{b}" if b != "inf" else f"$\\geq${a}" for a, b in zip(bins[:-1], bins[1:])]


def table_distance(dist):
    if "monocop" not in dist or "parallel" not in dist:
        return None
    labels = bin_labels(dist["monocop"]["bins"])
    rows = []
    for q, name, unit in (("depth", "Độ sâu", "m"), ("size", "Kích thước", "m"), ("angle", "Góc quay", "độ")):
        for group in ("parallel", "monocop"):
            v = dist[group][q]
            rows.append(f"{name} ({unit}) & {GROUP_NAME[group]} & " + " & ".join(fmt(x) for x in v)
                        + f" & {fmt(dist[group][q + '_all'])} \\\\")
        rows.append("\\midrule")
    m = dist["monocop"]
    for key, name in (("depth_chain", "chỉ nhánh chain"), ("depth_parallel", "chỉ nhánh parallel"),
                      ("depth_oracle", "oracle (nhánh tốt hơn)")):
        rows.append(f"Độ sâu (m) & MonoCoP, {name} & " + " & ".join(fmt(x) for x in m[key])
                    + f" & {fmt(m['depth_all_' + key.split('_')[1]])} \\\\")
    return ("\\begin{tabular}{ll" + "r" * (len(labels) + 1) + "}\n\\toprule\n"
            "Sai số & Mô hình & " + " & ".join(labels) + " & Tất cả \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def kittic_summary(analysis, runs):
    out = {}
    for group, modes in (("parallel", ["parallel_baseline"]), ("monocop", ["gs", "chain", "parallel"])):
        names = [r["name"] for r in runs if r["group"] == group and r["complete"] and r["name"] in analysis]
        for mode in modes:
            per_set = {}
            for s in ["clean"] + list(CORRUPTION_NAME):
                vals = [analysis[n]["sets"][s]["modes"][mode]["ap"]["iou70"]["3d"][1] for n in names
                        if s in analysis[n]["sets"] and mode in analysis[n]["sets"][s]["modes"]]
                if vals and len(vals) == len(names):
                    per_set[s] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
            if per_set:
                out[f"{group}:{mode}"] = {"n_ckpt": len(names), "sets": per_set}
    return out


def table_kittic(kc):
    cols = [("parallel:parallel_baseline", "Baseline"), ("monocop:gs", "MonoCoP (GS)"),
            ("monocop:chain", "MonoCoP, chain"), ("monocop:parallel", "MonoCoP, parallel")]
    cols = [(k, n) for k, n in cols if k in kc]
    if len(cols) < 2:
        return None
    rows = []
    for s in ["clean"] + list(CORRUPTION_NAME):
        if not all(s in kc[k]["sets"] for k, _ in cols):
            continue
        name = "Ảnh gốc" if s == "clean" else CORRUPTION_NAME[s]
        cells = []
        for k, _ in cols:
            v = kc[k]["sets"][s]["mean"]
            clean = kc[k]["sets"]["clean"]["mean"]
            cells.append(fmt(v) if s == "clean" else f"{fmt(v)} ({100 * (v - clean) / clean:+.0f}\\%)")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if s == "clean":
            rows.append("\\midrule")
    return ("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n"
            "Tập val & " + " & ".join(n for _, n in cols) + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def cost_summary(analysis, runs, groups):
    out = {}
    for group in ("parallel", "monocop"):
        an = [analysis[r["name"]] for r in runs if r["group"] == group and r["name"] in analysis]
        sp = [a["speed"] for a in an if "speed" in a and a["speed"].get("other_processes_on_gpu", 1) == 0]
        out[group] = {"params_M": an[0]["params_M"] if an else None,
                      "latency_ms": float(np.median([s["latency_ms_bs1_fp32"] for s in sp])) if sp else None,
                      "fps": float(np.median([s["fps"] for s in sp])) if sp else None,
                      "train_hours": groups.get(group, {}).get("train_hours")}
    return out


def table_cost(cost):
    rows = []
    for group in ("parallel", "monocop"):
        c = cost[group]
        h = c["train_hours"]
        rows.append(f"{GROUP_NAME[group]} & {fmt(c['params_M'])} & {fmt(c['latency_ms'], 1)} & {fmt(c['fps'], 1)} & "
                    + (pm(h, 1) if h else "--") + " \\\\")
    return ("\\begin{tabular}{lrrrr}\n\\toprule\n"
            "Biến thể & Tham số (M) & Latency (ms) & FPS & Thời gian train (giờ) \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------- figures
def fig_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, lo in zip(axes, (1, 100)):
        seen = set()
        for r in runs:
            e = [x for x in r["curve_mod3d"] if x >= lo]
            label = GROUP_NAME[r["group"]] if r["group"] not in seen else None
            seen.add(r["group"])
            ax.plot(e, [r["curve_mod3d"][x] for x in e], color=COLOR[r["group"]], lw=0.8, alpha=0.8, label=label)
        for d in (85, 125, 165, 205):
            if d >= lo:
                ax.axvline(d, color="k", ls=":", lw=0.6)
        ax.set_xlabel("epoch")
        ax.set_ylabel("AP$_{3D}$|R40 Car Moderate (val)")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Toàn bộ 250 epoch")
    axes[1].set_title("Epoch 100--250 (đường chấm: giảm LR)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "curves.png"), dpi=150)
    plt.close(fig)


def fig_branch(analysis, names):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n in names:
        modes = analysis[n]["sets"]["clean"]["modes"]
        pts = sorted((v["chain_frac"], v["ap"]["iou70"]["3d"][1]) for m, v in modes.items()
                     if m.startswith("tau@p") or m in ("chain", "parallel"))
        ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=1, label=run_label(n))
        gs = modes["gs"]
        ax.scatter([100 * gs["chain_frac"]], [gs["ap"]["iou70"]["3d"][1]], marker="*", s=120, color="k", zorder=5)
    ax.set_xlabel("Tỉ lệ object dùng nhánh chain (%)")
    ax.set_ylabel("AP$_{3D}$|R40 Car Moderate (val)")
    ax.set_title("Quét ngưỡng $\\tau$ của GS (sao: $\\tau=5$ mặc định)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "branch_ablation.png"), dpi=150)
    plt.close(fig)


def fig_distance(dist):
    labels = [l.replace("$\\geq$", "≥") for l in bin_labels(dist["monocop"]["bins"])]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (q, title) in zip(axes, (("depth", "Độ sâu (m)"), ("size", "Kích thước (m)"), ("angle", "Góc quay (độ)"))):
        for group in ("parallel", "monocop"):
            ax.plot(x, dist[group][q], marker="o", color=COLOR[group], label=GROUP_NAME[group])
        if q == "depth":
            ax.plot(x, dist["monocop"]["depth_oracle"], ls="--", color="k", lw=1, label="MonoCoP, oracle nhánh")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Độ sâu ground truth (m)")
        ax.set_title("MAE " + title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "error_by_distance.png"), dpi=150)
    plt.close(fig)


def fig_kittic(kc):
    cols = [k for k in ("parallel:parallel_baseline", "monocop:gs") if k in kc]
    sets = [s for s in CORRUPTION_NAME if all(s in kc[k]["sets"] for k in cols)]
    if len(cols) < 2 or not sets:
        return
    x = np.arange(len(sets))
    fig, ax = plt.subplots(figsize=(12, 4))
    for i, k in enumerate(cols):
        g = k.split(":")[0]
        ax.bar(x + (i - 0.5) * 0.38, [kc[k]["sets"][s]["mean"] for s in sets], 0.38,
               yerr=[kc[k]["sets"][s]["std"] for s in sets], color=COLOR[g], label=GROUP_NAME[g], capsize=2)
        ax.axhline(kc[k]["sets"]["clean"]["mean"], color=COLOR[g], ls="--", lw=0.8)
    ax.set_xticks(x, [CORRUPTION_NAME[s] for s in sets], rotation=30, ha="right")
    ax.set_ylabel("AP$_{3D}$|R40 Car Moderate")
    ax.set_title("KITTI-C (mức 1), trung bình qua các checkpoint; đường đứt: ảnh gốc")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "kittic.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- markdown
def markdown(runs, groups, branch, dist, kc, cost):
    md = ["# A2 results (generated by `A2/summarize.py`)", "",
          "KITTI val, Car, AP|R40 at IoU 0.7. `best` = checkpoint with the highest val AP3D Moderate "
          "(selected on the same val set, as in the paper); `last` = epoch 250.", "",
          "| Variant | Runs | Ckpt | AP3D E | AP3D M | AP3D H | APBEV M | AP3D M median |", "|---|---|---|---|---|---|---|---|"]
    for group in ("parallel", "monocop", "monocop_frozen_stem"):
        if group not in groups:
            continue
        g = groups[group]
        for which in ("best", "last"):
            md.append(f"| {GROUP_NAME[group]} | {g['n']} | {which} | "
                      + " | ".join(f"{g[f'{which}_iou70_3d_{l}']['mean']:.2f} ± {g[f'{which}_iou70_3d_{l}']['std']:.2f}" for l in LEVELS)
                      + f" | {g[f'{which}_iou70_bev_mod']['mean']:.2f} ± {g[f'{which}_iou70_bev_mod']['std']:.2f}"
                      + f" | {g[f'{which}_iou70_3d_mod']['median']:.2f} |")
    md += ["", "## Runs", "", "| Variant | Seed | Run | Epochs | Best ep. | AP3D best E/M/H | AP3D ep.250 E/M/H | Hours |",
           "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        md.append(f"| {GROUP_NAME[r['group']]} | {r['seed']} | `{r['name']}` | {r['n_epochs']} | {r['best_epoch']} | "
                  + " / ".join(fmt(x) for x in r["best"]["iou70"]["3d"]) + " | "
                  + " / ".join(fmt(x) for x in r["last"]["iou70"]["3d"]) + f" | {fmt(r['train_hours'], 1)} |")
    if branch:
        md += ["", "## Branch selection at inference (MonoCoP, clean val, mean ± std over checkpoints)", "",
               "| Rule | Chain % | AP3D E | AP3D M | AP3D H |", "|---|---|---|---|---|"]
        for r in branch:
            md.append(f"| {r['mode']} | {100 * r['chain_frac']:.1f} | "
                      + " | ".join(f"{m:.2f} ± {s:.2f}" for m, s in zip(r["ap_mean"], r["ap_std"])) + " |")
    if dist.get("monocop") and dist.get("parallel"):
        md += ["", "## MAE by GT depth (matched Cars, clean val)", "",
               "| Error | Model | " + " | ".join(f"{a}-{b}" for a, b in zip(dist["monocop"]["bins"][:-1], dist["monocop"]["bins"][1:])) + " | all |",
               "|---|---|" + "---|" * len(dist["monocop"]["bins"])]
        for q in ("depth", "size", "angle"):
            for group in ("parallel", "monocop"):
                md.append(f"| {q} | {GROUP_NAME[group]} | " + " | ".join(fmt(x) for x in dist[group][q]) + f" | {fmt(dist[group][q + '_all'])} |")
        md.append(f"\nGS picks the branch with the lower depth error for {100 * dist['monocop']['gs_picks_better_all']:.1f}% of matched Cars.")
    if kc:
        md += ["", "## KITTI-C (AP3D Moderate, mean over checkpoints)", "", "| Set | " + " | ".join(kc) + " |",
               "|---|" + "---|" * len(kc)]
        for s in ["clean"] + list(CORRUPTION_NAME):
            if all(s in v["sets"] for v in kc.values()):
                md.append(f"| {s} | " + " | ".join(fmt(v["sets"][s]["mean"]) for v in kc.values()) + " |")
    md += ["", "## Cost", "", "| Variant | Params (M) | Latency bs1 fp32 (ms) | FPS |", "|---|---|---|---|"]
    for group, c in cost.items():
        md.append(f"| {GROUP_NAME[group]} | {fmt(c['params_M'])} | {fmt(c['latency_ms'], 1)} | {fmt(c['fps'], 1)} |")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monocop-root", default=os.path.expanduser("~/MonoCoP"))
    ap.add_argument("--analysis-dir", default=ANALYSIS)
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    runs = collect_runs(args.monocop_root)
    for r in runs:
        if not r["complete"]:
            print(f"unfinished: {r['name']} ({r['n_epochs']} epochs) - excluded from group statistics")
    groups = summarize_groups(runs)
    analysis = load_analysis(runs, args.analysis_dir)
    branch, branch_names = branch_rows(analysis, runs)
    dist = distance_summary(analysis, runs)
    kc = kittic_summary(analysis, runs)
    cost = cost_summary(analysis, runs, groups)

    write(os.path.join(TABLES, "main.tex"), table_main(groups))
    write(os.path.join(TABLES, "runs.tex"), table_runs(runs))
    write(os.path.join(TABLES, "cost.tex"), table_cost(cost))
    fig_curves(runs)
    if branch:
        tex, n = table_branch(branch, branch_names)
        write(os.path.join(TABLES, "branch.tex"), tex)
        fig_branch(analysis, branch_names)
    else:
        write(os.path.join(TABLES, "branch.tex"), placeholder_table())
        placeholder_fig("branch_ablation.png")
    if table_distance(dist):
        write(os.path.join(TABLES, "distance.tex"), table_distance(dist))
        fig_distance(dist)
    else:
        write(os.path.join(TABLES, "distance.tex"), placeholder_table())
        placeholder_fig("error_by_distance.png")
    if table_kittic(kc):
        write(os.path.join(TABLES, "kittic.tex"), table_kittic(kc))
        fig_kittic(kc)
    else:
        write(os.path.join(TABLES, "kittic.tex"), placeholder_table())
        placeholder_fig("kittic.png")

    slim = [{k: v for k, v in r.items() if k != "epochs"} | {"epochs": {e: r["epochs"][e] for e in sorted(r["epochs"])}}
            for r in runs]
    json.dump({"runs": slim, "groups": groups, "branch_selection": branch, "branch_checkpoints": branch_names,
               "error_by_distance": dist, "kittic": kc, "cost": cost}, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    write(os.path.join(OUT, "results.md"), markdown(runs, groups, branch, dist, kc, cost))


if __name__ == "__main__":
    main()
