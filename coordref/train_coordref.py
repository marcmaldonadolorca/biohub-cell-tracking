#!/usr/bin/env python3
"""Train the coordinate-refinement head from captured detector features.

Inputs: <captures>/<stem>/<t>.npz written by coordref.capture() (integer coords on the (1,4,4) grid
plus 224 features), and the ground-truth <train-dir>/<stem>.geff of the competition train set.
Each detection is matched to a GT cell per frame with the Hungarian algorithm, gated at 7 um (the
matching radius of the official metric). The head (Linear 224->32, SiLU, Linear 32->3, last layer
zero-initialised so it starts as the identity) predicts the shift in um, bounded to 2 um.

The split is BY VIDEO, never by detection. The number that decides is the change of the mean
detection->GT distance on the held-out videos; the head is marked `puerta_ok` (gate passed) only if
it drops by >= 5 % pooled AND improves >= 60 % of the held-out videos.

  # held-out measurement (default: the last videos of each embryo prefix)
  python train_coordref.py --captures CAP --train-dir TRAIN --holdout 6 --out head_holdout.pt
  # unseen embryo: hold out every video of one prefix
  python train_coordref.py --captures CAP --train-dir TRAIN --val-prefix 44b6 --out head_44b6.pt
  # final refit on every video, sealed with the held-out evidence measured before
  python train_coordref.py --captures CAP --train-dir TRAIN --final --seal=-20.48,6/6 --out head.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

VOXEL_UM = np.array([1.625, 0.40625, 0.40625])  # um per voxel on the ORIGINAL grid (z, y, x)
GRID_UM = VOXEL_UM * np.array([1.0, 4.0, 4.0])  # um per voxel on the predictor's downsampled grid
MATCH_UM = 7.0
MAX_SHIFT_UM = 2.0


def make_head(in_dim=224, hidden=32):
    head = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    d = head(x)
    return MAX_SHIFT_UM * d / (1.0 + torch.linalg.vector_norm(d, dim=-1, keepdim=True))


def ridge_fit(X, Y, lam):
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1)
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1], dtype=Xb.dtype), Xb.T @ Y)


def ridge_pred(W, X):
    d = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], axis=1) @ W
    return MAX_SHIFT_UM * d / (1.0 + np.linalg.norm(d, axis=1, keepdims=True))


def gt_per_frame(geff):
    """{t: (n, 3) GT centres in um}."""
    import tracksdata as td
    g = td.graph.IndexedRXGraph.from_geff(geff)
    g = g[0] if isinstance(g, tuple) else g
    per_t = {}
    for row in g.node_attrs().iter_rows(named=True):
        per_t.setdefault(int(row["t"]), []).append((float(row["z"]), float(row["y"]), float(row["x"])))
    return {t: np.array(v) * VOXEL_UM for t, v in per_t.items()}


def video_pairs(geff, cap_dir):
    """(features, shift_um, initial_distance_um) of the detections matched to a GT cell."""
    gt = gt_per_frame(geff)
    X, Y, D0 = [], [], []
    for f in sorted(Path(cap_dir).glob("*.npz")):
        g = gt.get(int(f.stem))
        z = np.load(f)
        if g is None or not len(g) or not len(z["coords"]):
            continue
        det_um = z["coords"][:, 1:].astype(np.float64) * GRID_UM
        d = np.linalg.norm(det_um[:, None, :] - g[None, :, :], axis=-1)
        d_gated = np.where(d <= MATCH_UM, d, 1e6)
        feats = z["feats"].astype(np.float32)
        for r, c in zip(*linear_sum_assignment(d_gated)):
            if d_gated[r, c] < 1e6:
                X.append(feats[r]); Y.append(g[c] - det_um[r]); D0.append(d[r, c])
    return (np.stack(X), np.stack(Y).astype(np.float32), np.array(D0)) if X else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", required=True, help="directory with <stem>/<t>.npz")
    ap.add_argument("--train-dir", required=True, help="competition train dir with <stem>.geff")
    ap.add_argument("--out", required=True, help="output .pt")
    ap.add_argument("--holdout", type=int, default=6, help="held-out videos, split evenly across embryo prefixes")
    ap.add_argument("--val-stems", help="explicit comma-separated held-out videos (overrides --holdout)")
    ap.add_argument("--val-prefix", help="hold out every video of this embryo prefix (unseen-embryo test)")
    ap.add_argument("--final", action="store_true", help="train on ALL videos; gate evidence comes from --seal")
    ap.add_argument("--seal", help="held-out evidence measured before, 'pct,improved/total' (with --final)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    train_dir = Path(a.train_dir)

    stems = sorted(d.name for d in Path(a.captures).iterdir()
                   if d.is_dir() and (train_dir / f"{d.name}.geff").exists())
    data = {}
    for s in stems:
        r = video_pairs(train_dir / f"{s}.geff", Path(a.captures) / s)
        if r is not None:
            data[s] = r
            print(f"  {s}: {len(r[0]):6d} pairs | mean dist {r[2].mean():.3f} um", flush=True)
    if not data:
        raise SystemExit("no data")
    print(f"{len(data)} videos, {sum(len(v[0]) for v in data.values())} pairs")

    if a.final:
        val = []
    elif a.val_prefix:
        val = [s for s in data if s.startswith(a.val_prefix)]
    elif a.val_stems:
        val = [s for s in a.val_stems.split(",") if s in data]
    else:  # stratified by embryo prefix: the last videos of each, not the alphabetical tail
        by_prefix = {}
        for s in data:
            by_prefix.setdefault(s.split("_")[0], []).append(s)
        val = [s for lst in by_prefix.values() for s in lst[-max(1, a.holdout // len(by_prefix)):]]
    tr = [s for s in data if s not in val]
    Xtr = torch.tensor(np.concatenate([data[s][0] for s in tr]))
    Ytr = torch.tensor(np.concatenate([data[s][1] for s in tr]))
    print(f"train on {len(tr)} videos ({len(Xtr)} pairs), hold out {len(val)}: {val}")

    mean, scale = Xtr.mean(0), Xtr.std(0).clamp_min(1e-3)
    head = make_head(Xtr.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    n = len(Xtr)
    for ep in range(a.epochs):
        perm, tot = torch.randperm(n), 0.0
        for i in range(0, n, 4096):
            idx = perm[i:i + 4096]
            loss = torch.nn.functional.smooth_l1_loss(bounded(head, (Xtr[idx] - mean) / scale), Ytr[idx], beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d} loss {tot / n:.4f}", flush=True)
    head.eval()

    saved = {"mean": mean, "scale": scale, "max_shift_um": MAX_SHIFT_UM, "grid_um": GRID_UM.tolist(),
             "train_stems": tr, "val_stems": val}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.final:
        pct, vids = (a.seal or "-99,0/0").split(",")
        saved.update(kind="mlp", state_dict=head.state_dict(), holdout_pct=float(pct),
                     holdout_videos_mejoran=vids, puerta_ok=a.seal is not None,
                     nota="trained on ALL videos; the gate was measured with a by-video split (see --seal)")
        torch.save(saved, out)
        print(f"final head, {len(tr)} videos ({n} pairs) -> {out}  [seal {pct} %, {vids}]")
        return

    def change_pct(pred_fn, stems_eval):
        d0 = np.concatenate([data[s][2] for s in stems_eval])
        d1 = np.concatenate([np.linalg.norm(data[s][1] - pred_fn(((torch.tensor(data[s][0]) - mean) / scale).numpy()),
                                            axis=1) for s in stems_eval])
        return float((d1.mean() - d0.mean()) / d0.mean() * 100)

    Xtr_n, Ytr_n = ((Xtr - mean) / scale).numpy(), Ytr.numpy()
    mlp_fn = lambda x: bounded(head, torch.tensor(x)).detach().numpy()
    results = {"mlp": (change_pct(mlp_fn, val), mlp_fn, None)}
    for lam in (1.0, 10.0, 100.0, 1000.0):
        W = ridge_fit(Xtr_n, Ytr_n, lam)
        fn = lambda x, W=W: ridge_pred(W, x)
        results[f"ridge{lam:g}"] = (change_pct(fn, val), fn, W)
    print("\nheld-out videos, change of the mean distance to GT (negative = better):")
    for k, (pct, _, _) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"  {k:10s} {pct:+6.2f} %")
    best = min(results, key=lambda k: results[k][0])
    pct, fn, W = results[best]
    improved = 0
    for s in val:
        d0 = data[s][2].mean()
        d1 = np.linalg.norm(data[s][1] - fn(((torch.tensor(data[s][0]) - mean) / scale).numpy()), axis=1).mean()
        improved += d1 < d0
        print(f"  {s}: {d0:.3f} -> {d1:.3f} um ({(d1 - d0) / d0 * 100:+.1f} %)  n={len(data[s][0])}")
    gate = pct <= -5.0 and improved / max(len(val), 1) >= 0.6
    print(f"\nBEST: {best}  POOLED {pct:+.2f} %  |  {improved}/{len(val)} videos improve  |  gate {'PASSED' if gate else 'FAILED'}")
    saved.update(kind="ridge" if W is not None else "mlp", holdout_pct=pct,
                 holdout_videos_mejoran=f"{improved}/{len(val)}", puerta_ok=bool(gate))
    if W is not None:
        saved.update(ridge_W=torch.tensor(W, dtype=torch.float32), ridge_lambda=float(best[5:]))
    else:
        saved["state_dict"] = head.state_dict()
    torch.save(saved, out)
    print("head ->", out)


if __name__ == "__main__":
    main()
