"""Inference-only analysis of a MonoCoP / parallel-baseline checkpoint on KITTI val.

1. Branch-selection ablation (MonoCoP only): the depth head has a chain and a parallel branch. One
   inference pass captures both, then every selection rule (always chain, always parallel, GS with a
   threshold sweep, lower-uncertainty branch) is decoded and evaluated offline.
2. Error by distance: MAE of depth / size / angle on detections matched to GT Cars, binned by GT depth.
   For MonoCoP it also reports the per-object oracle branch and how often GS picks the better branch.
3. KITTI-C: the same evaluation with val images replaced by corrupted copies.

Run from the MonoCoP root (needs its `lib/` and the monocop conda env; without
NUMBA_CUDA_USE_NVIDIA_BINDING=1 the numba-based KITTI evaluator segfaults on this machine):
    NUMBA_CUDA_USE_NVIDIA_BINDING=1 python /path/to/A2/monocop/analyze.py --config config/monocop_car.yaml \
        --ckpt outputs/monocop_car/<run>/checkpoint_best.pth --out /path/to/<run>.json [--sets all] [--speed]
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import tqdm
import yaml

sys.path.insert(0, os.getcwd())
from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.decode_helper import extract_dets_from_outputs, decode_detections
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti

CORRUPTIONS = ["brightness_1", "contrast_1", "saturate_1", "defocus_blur_1", "glass_blur_1", "motion_blur_1",
               "gaussian_noise_1", "shot_noise_1", "impulse_noise_1", "pixelate_1", "fog_1", "frost_1", "snow_1"]
DIST_BINS = [0, 10, 20, 30, 40, 50, math.inf]
OUT_KEYS = ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth", "pred_angle")
# decode_detections row: cls, alpha, x1, y1, x2, y2, h, w, l, x, y, z, ry, score
H, W, L, Z, RY, SCORE = 6, 7, 8, 11, 12, 13


def parse_ap(text):
    out = {}
    for iou, key in (("0.70, 0.70, 0.70", "iou70"), ("0.70, 0.50, 0.50", "iou50")):
        m = re.search(r"Car AP_R40@%s:\nbbox AP:(.*)\nbev  AP:(.*)\n3d   AP:(.*)\n" % re.escape(iou), text)
        out[key] = {k: [round(float(x), 4) for x in v.split(",")] for k, v in zip(("bbox", "bev", "3d"), m.groups())}
    return out


@torch.no_grad()
def run_inference(model, loader, device, desc):
    captured, hooks = {}, []
    if model.chain_prediction:
        hooks.append(model.depth_embed[-1].register_forward_hook(lambda m, i, o: captured.__setitem__("chain", o)))
        hooks.append(model.depth_embed_para[-1].register_forward_hook(lambda m, i, o: captured.__setitem__("para", o)))
    batches = []
    for inputs, calibs, targets, info in tqdm.tqdm(loader, desc=desc, leave=False):
        out = model(inputs.to(device), calibs.to(device), targets, info["img_size"].to(device), dn_args=0)
        b = {k: out[k].float().cpu() for k in OUT_KEYS}
        if model.chain_prediction:
            b["chain"], b["para"] = captured["chain"].float().cpu(), captured["para"].float().cpu()
        b["img_id"], b["img_size"] = info["img_id"].numpy(), info["img_size"].numpy()
        batches.append(b)
    for h in hooks:
        h.remove()
    return batches


def make_modes(batches, model, taus, threshold):
    """Return {mode: per-batch bool mask [B, Q, 1] of queries that use the chain branch}."""
    if not model.chain_prediction:
        return {"parallel_baseline": None}
    modes = {"chain": lambda c, p: torch.ones_like(c[..., 1:2], dtype=torch.bool),
             "parallel": lambda c, p: torch.zeros_like(c[..., 1:2], dtype=torch.bool),
             "gs": lambda c, p, t=model.uncertainty_tau: c[..., 1:2] < t,
             "min_unc": lambda c, p: c[..., 1:2] <= p[..., 1:2]}
    for t in taus or []:
        modes["tau=%g" % t] = lambda c, p, t=t: c[..., 1:2] < t
    if taus is None:
        kept = torch.cat([b["chain"][..., 1][b["pred_logits"].sigmoid().amax(-1) >= threshold] for b in batches])
        for q in range(10, 100, 10):
            t = torch.quantile(kept, q / 100).item()
            modes["tau@p%d" % q] = lambda c, p, t=t: c[..., 1:2] < t
    return {name: [fn(b["chain"], b["para"]) for b in batches] for name, fn in modes.items()}


def depth_for_mode(b, mask, default_tau):
    if mask is None:
        return b["pred_depth"]
    chain, para = b["chain"], b["para"]
    geo = b["pred_depth"][..., 0:1] - torch.where(chain[..., 1:2] < default_tau, chain, para)[..., 0:1]
    d = torch.where(mask, chain, para)
    return torch.cat([geo + d[..., 0:1], d[..., 1:2]], -1)


def decode(batches, masks, dataset, calibs, cfg_test, default_tau):
    results = {}
    for k, b in enumerate(batches):
        outs = {key: b[key] for key in OUT_KEYS}
        outs["pred_depth"] = depth_for_mode(b, None if masks is None else masks[k], default_tau)
        dets = extract_dets_from_outputs(outs, K=dataset.max_objs, topk=cfg_test["topk"]).numpy()
        info = {"img_id": b["img_id"], "img_size": b["img_size"]}
        results.update(decode_detections(dets, info, [calibs[i] for i in b["img_id"]],
                                         dataset.cls_mean_size, cfg_test.get("threshold", 0.2)))
    return results


def evaluate(results, dataset, gt_annos, img_ids, tmpdir):
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir)
    for img_id, preds in results.items():
        with open(os.path.join(tmpdir, "%06d.txt" % img_id), "w") as f:
            for p in preds:
                f.write("{} 0.0 0".format(dataset.class_name[int(p[0])]))
                f.write("".join(" {:.2f}".format(v) for v in p[1:]) + "\n")
    dt_annos = kitti.get_label_annos(tmpdir, img_ids)
    text, _, _ = get_official_eval_result(gt_annos, dt_annos, 0)
    return parse_ap(text)


def chain_fraction(batches, masks, threshold):
    if masks is None:
        return None
    sel = [m[..., 0][b["pred_logits"].sigmoid().amax(-1) >= threshold] for b, m in zip(batches, masks)]
    return round(torch.cat(sel).float().mean().item(), 4)


def iou2d(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def load_gt_cars(dataset, img_ids):
    """GT Cars that count for KITTI AP (Easy/Moderate/Hard)."""
    gts = {}
    for i in img_ids:
        gts[i] = [o for o in dataset.get_label(i) if o.cls_type == "Car" and o.level in (1, 2, 3)]
    return gts


def error_by_distance(results, gts, ref_mode, car_id, iou_thr=0.5):
    """Greedy 2D-IoU matching on ref_mode; rows of all modes are aligned (same top-k and threshold)."""
    modes = list(results)
    bins = len(DIST_BINS) - 1
    rec = {"n_gt": [0] * bins, "n_matched": [0] * bins}
    err = {m: {"depth": [[] for _ in range(bins)], "size": [[] for _ in range(bins)],
               "angle": [[] for _ in range(bins)]} for m in modes}
    branch = "chain" in results and "parallel" in results
    if branch:
        err["oracle"] = {"depth": [[] for _ in range(bins)]}
        gs_pick = {"chain": [0] * bins, "better": [0] * bins}

    for img_id, objs in gts.items():
        ref = results[ref_mode][img_id]
        rows = [j for j, p in enumerate(ref) if int(p[0]) == car_id]
        rows.sort(key=lambda j: -ref[j][SCORE])
        used = set()
        for o in objs:
            rec["n_gt"][np.digitize(o.pos[2], DIST_BINS) - 1] += 1
        for j in rows:
            best, best_iou = None, iou_thr
            for g, o in enumerate(objs):
                if g in used:
                    continue
                v = iou2d(ref[j][2:6], o.box2d)
                if v >= best_iou:
                    best, best_iou = g, v
            if best is None:
                continue
            used.add(best)
            o = objs[best]
            k = np.digitize(o.pos[2], DIST_BINS) - 1
            rec["n_matched"][k] += 1
            for m in modes:
                p = results[m][img_id][j]
                err[m]["depth"][k].append(abs(p[Z] - o.pos[2]))
                err[m]["size"][k].append((abs(p[H] - o.h) + abs(p[W] - o.w) + abs(p[L] - o.l)) / 3)
                err[m]["angle"][k].append(math.degrees(abs((p[RY] - o.ry + math.pi) % (2 * math.pi) - math.pi)))
            if branch:
                ec = abs(results["chain"][img_id][j][Z] - o.pos[2])
                ep = abs(results["parallel"][img_id][j][Z] - o.pos[2])
                err["oracle"]["depth"][k].append(min(ec, ep))
                if "gs" in results:
                    gs_chain = abs(results["gs"][img_id][j][Z] - results["chain"][img_id][j][Z]) < 1e-6
                    gs_pick["chain"][k] += int(gs_chain)
                    gs_pick["better"][k] += int(gs_chain == (ec <= ep))

    def summ(lists):
        flat = [x for l in lists for x in l]
        return {"per_bin": [round(float(np.mean(l)), 4) if l else None for l in lists],
                "per_bin_median": [round(float(np.median(l)), 4) if l else None for l in lists],
                "all": round(float(np.mean(flat)), 4) if flat else None}

    out = {"bins_m": [b if b != math.inf else "inf" for b in DIST_BINS], **rec,
           "errors": {m: {q: summ(v) for q, v in e.items()} for m, e in err.items()}}
    if branch and "gs" in results:
        n = rec["n_matched"]
        out["gs_chain_frac"] = [round(c / x, 4) if x else None for c, x in zip(gs_pick["chain"], n)]
        out["gs_picks_better_frac"] = [round(c / x, 4) if x else None for c, x in zip(gs_pick["better"], n)]
        out["gs_picks_better_all"] = round(sum(gs_pick["better"]) / max(1, sum(n)), 4)
    return out


@torch.no_grad()
def measure_speed(model, dataset, device, warmup=20, iters=200):
    inputs, calib, targets, info = torch.utils.data.default_collate([dataset[0]])
    inputs, calib, size = inputs.to(device), calib.to(device), info["img_size"].to(device)
    for _ in range(warmup):
        model(inputs, calib, targets, size, dn_args=0)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(inputs, calib, targets, size, dn_args=0)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    gpu_index = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    apps = subprocess.run(["nvidia-smi", "-i", gpu_index, "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split()
    return {"latency_ms_bs1_fp32": round(ms, 2), "fps": round(1000 / ms, 1),
            "gpu": torch.cuda.get_device_name(), "other_processes_on_gpu": len(apps) - 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="clean", help="'clean', 'all' or comma list of clean/KITTI-C folders")
    ap.add_argument("--kittic-root", default="/localhome/local-tronganhn/dataset/KITTI/kitti_c")
    ap.add_argument("--taus", default=None, help="comma list of fixed tau; default: percentiles of chain log-variance")
    ap.add_argument("--corruption-modes", default="gs,chain,parallel")
    ap.add_argument("--subset", type=int, default=0, help="debug: only the first N val images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="recompute sets already present in --out")
    args = ap.parse_args()

    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    device = torch.device("cuda")
    _, loader = build_dataloader(cfg["dataset"], workers=args.workers)
    dataset = loader.dataset
    if args.subset:
        dataset.idx_list = dataset.idx_list[:args.subset]
    img_ids = [int(i) for i in dataset.idx_list]

    model, _ = build_model(cfg["model"])
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    calibs = {i: dataset.get_calib(i) for i in img_ids}
    gt_annos = kitti.get_label_annos(dataset.label_dir, img_ids)
    gts = load_gt_cars(dataset, img_ids)
    car_id = dataset.class_name.index("Car")
    taus = None if args.taus is None else [float(t) for t in args.taus.split(",")]
    tmpdir = os.path.splitext(args.out)[0] + "_tmp"

    report = json.load(open(args.out)) if os.path.exists(args.out) else {}
    report.update({"config": args.config, "ckpt": args.ckpt, "epoch": ckpt.get("epoch"),
                   "variant": "monocop" if model.chain_prediction else "parallel_baseline",
                   "uncertainty_tau": model.uncertainty_tau if model.chain_prediction else None,
                   "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
                   "n_images": len(img_ids)})
    report.setdefault("sets", {})
    if args.speed:
        report["speed"] = measure_speed(model, dataset, device)
        print("speed", report["speed"])

    sets = ["clean"] + CORRUPTIONS if args.sets == "all" else args.sets.split(",")
    clean_dir = dataset.image_dir
    for s in sets:
        if s in report["sets"] and not args.overwrite:
            print(s, "already in", args.out, "- skipped")
            continue
        dataset.image_dir = clean_dir if s == "clean" else os.path.join(args.kittic_root, s)
        t0 = time.time()
        batches = run_inference(model, loader, device, s)
        masks = make_modes(batches, model, taus, cfg["tester"].get("threshold", 0.2))
        if s != "clean" and model.chain_prediction:
            masks = {m: masks[m] for m in args.corruption_modes.split(",")}
        default_tau = getattr(model, "uncertainty_tau", 0)
        results, entry = {}, {"modes": {}}
        for name, mask in masks.items():
            results[name] = decode(batches, mask, dataset, calibs, cfg["tester"], default_tau)
            entry["modes"][name] = {"ap": evaluate(results[name], dataset, gt_annos, img_ids, tmpdir),
                                    "chain_frac": chain_fraction(batches, mask, cfg["tester"].get("threshold", 0.2))}
            print(s, name, "AP3D R40 IoU0.7:", entry["modes"][name]["ap"]["iou70"]["3d"],
                  "chain frac:", entry["modes"][name]["chain_frac"])
        ref = "gs" if "gs" in results else next(iter(results))
        mae_modes = {m: results[m] for m in ("gs", "chain", "parallel", "min_unc", "parallel_baseline") if m in results}
        entry["error_by_distance"] = error_by_distance(mae_modes, gts, ref, car_id)
        if model.chain_prediction and s == "clean":
            kept = torch.cat([b["chain"][..., 1][b["pred_logits"].sigmoid().amax(-1) >= 0.2] for b in batches])
            kept_p = torch.cat([b["para"][..., 1][b["pred_logits"].sigmoid().amax(-1) >= 0.2] for b in batches])
            qs = torch.tensor([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            entry["logvar_quantiles"] = {"q": qs.tolist(),
                                         "chain": [round(v, 4) for v in torch.quantile(kept, qs).tolist()],
                                         "parallel": [round(v, 4) for v in torch.quantile(kept_p, qs).tolist()]}
        entry["seconds"] = round(time.time() - t0, 1)
        report["sets"][s] = entry
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=1)
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
