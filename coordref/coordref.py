"""Sub-voxel refinement of detection coordinates for the BioHub cell-tracking pipeline.

The detector in pilkwang's support pack returns INTEGER cell centres on the downsampled (1, 4, 4)
grid, which at 1.625 um/voxel is isotropic. A tiny MLP reads the UNet features at each detection and
at its 6 face neighbours (32 channels x 7 = 224 inputs) and predicts the shift to the true centre,
bounded to 2 um. Three changes to scripts/predict_unet_transformer.py go together:
  1. refine the detections right after they are found;
  2. do NOT cast the rescaled coordinates to int16 (the sub-voxel part must survive);
  3. index the transformer features TRILINEARLY: the stock integer gather truncates 12.7 -> 12,
     so float coordinates would read the wrong cell.
(3) is not optional: without it, float coordinates are truncated when the edge features are gathered.
`patch_predictor()` applies all three.

Idea and architecture: anvithpothula (public notebook "biohub-0-953-lb-original", formerly
"biohub-x138"). This file is an independent reimplementation.
"""
import os
from pathlib import Path

import numpy as np
import torch

OFFSETS = ((0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
GRID_UM = np.array([1.625, 1.625, 1.625], dtype=np.float32)  # downsampled (1,4,4) grid, (z, y, x)
_CACHE = None


def neighbour_features(arr, feature):
    """(n, 224) inputs: features at the detection plus the 6 neighbour-minus-centre differences.
    arr: (n, 4) integer (t, z, y, x) on the downsampled grid. feature: (1, C, Z, Y, X)."""
    xyz = torch.as_tensor(np.asarray(arr)[:, 1:].astype("int64"), device=feature.device)
    blocks = []
    for off in OFFSETS:
        loc = xyz + torch.tensor(off, device=feature.device)
        for axis, size in enumerate(feature.shape[-3:]):
            loc[:, axis].clamp_(0, size - 1)
        blocks.append(feature[0, :, loc[:, 0], loc[:, 1], loc[:, 2]].T)
    return torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1)


def load_head(path, require_gate=True):
    """Returns (fn, mean, scale, max_shift_um). fn maps normalised (n, 224) inputs to raw shifts."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if require_gate and not saved.get("puerta_ok", False):  # puerta_ok = passed the held-out gate
        raise RuntimeError(("coordref: head did not pass the held-out gate",
                            saved.get("holdout_pct"), saved.get("holdout_videos_mejoran")))
    if saved.get("kind", "mlp") == "ridge":
        W = saved["ridge_W"]
        fn = lambda x: torch.cat([x, torch.ones(len(x), 1)], dim=1) @ W
    else:
        fn = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
        fn.load_state_dict(saved["state_dict"])
        fn.eval()
    return fn, saved["mean"], saved["scale"], float(saved.get("max_shift_um", 2.0))


def refine(arr, feature):
    """Shifted float copy of arr. Head path from $BIOHUB_COORDREF_HEAD_PATH."""
    global _CACHE
    if not len(arr):
        return arr
    if _CACHE is None:
        _CACHE = load_head(os.environ["BIOHUB_COORDREF_HEAD_PATH"],
                           os.environ.get("BIOHUB_COORDREF_REQUIRE_GATE", "1") == "1")
    head, mean, scale, max_um = _CACHE
    x = neighbour_features(arr, feature).float().cpu()
    with torch.no_grad():
        d = head((x - mean) / scale)
        shift_um = max_um * d / (1.0 + torch.linalg.vector_norm(d, dim=-1, keepdim=True))
    out = np.asarray(arr).astype(np.float32).copy()
    out[:, 1:] += shift_um.numpy() / GRID_UM
    for axis, size in enumerate(feature.shape[-3:]):
        out[:, axis + 1] = np.clip(out[:, axis + 1], 0, size - 1)
    moved = np.linalg.norm((out[:, 1:] - np.asarray(arr)[:, 1:]) * GRID_UM, axis=1)
    if not np.isfinite(out).all() or moved.max() > max_um + 1e-4:
        raise RuntimeError("coordref: invalid shift")
    return out


def index_features(self, feat_maps, coords, mask):
    """Drop-in for UNetNodeTransformer._index_features: TRILINEAR gather. With integer coordinates
    it reproduces the stock integer gather exactly."""
    B, C = feat_maps.shape[:2]
    spatial = feat_maps.shape[2:]
    out = torch.zeros(B, coords.shape[1], C, device=feat_maps.device, dtype=feat_maps.dtype)
    for b in range(B):
        n = int(mask[b].sum().item())
        if n == 0:
            continue
        q = coords[b, :n].clone().float()
        for axis, size in enumerate(spatial):
            q[:, axis].clamp_(0, size - 1)
        low = q.floor().long()
        frac = q - low
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    shift = torch.tensor([dz, dy, dx], device=feat_maps.device)
                    loc = low + shift
                    for axis, size in enumerate(spatial):
                        loc[:, axis].clamp_(0, size - 1)
                    w = torch.where(shift.bool(), frac, 1.0 - frac).prod(dim=1)
                    out[b, :n] += feat_maps[b, :, loc[:, 0], loc[:, 1], loc[:, 2]].T * w[:, None].to(out.dtype)
    return out


def capture(stem, t, arr, feature, out_dir=None):
    """Training data: writes <out_dir>/<stem>/<t>.npz with integer coords and the 224 inputs.
    out_dir defaults to $BIOHUB_COORDREF_CAPTURE. Predictions are left unchanged."""
    out_dir = out_dir or os.environ.get("BIOHUB_COORDREF_CAPTURE", "")
    if not out_dir or not len(arr):
        return
    out = Path(out_dir) / str(stem)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{int(t):04d}.npz", coords=np.asarray(arr).astype("int16"),
                        feats=neighbour_features(arr, feature).detach().float().cpu().numpy().astype("float16"))


_IMPORT = "import tracksdata as td\n"
_DETECT = "                coord_offset[t] = (global_node_count, global_node_count + len(arr))"
_INT16 = "    coords = coords.astype(np.int16)\n"
_LOAD = "    model, window_size, downsample = load_model(weights_path, device)"


def patch_predictor(script, mode="refine"):
    """Rewrite scripts/predict_unet_transformer.py in place; copy this file next to it first.
    mode="refine": the three coupled changes. mode="capture": only dump training features."""
    script = Path(script)
    src = script.read_text()

    def sub(anchor, new):
        nonlocal src
        if src.count(anchor) != 1:
            raise RuntimeError(f"coordref: anchor found {src.count(anchor)} times: {anchor!r}")
        src = src.replace(anchor, new, 1)

    if mode == "capture":
        sub(_IMPORT, _IMPORT + "from coordref import capture as _coordref_capture\n")
        sub(_DETECT, "                _coordref_capture(ds_path.name.replace('.zarr', ''), t, arr, unet_out[:, f_idx])\n"
            + _DETECT)
    elif mode == "refine":
        sub(_IMPORT, _IMPORT + "from coordref import refine as _coordref_refine, index_features as _coordref_index\n")
        sub(_DETECT, "                arr = _coordref_refine(arr, unet_out[:, f_idx])\n" + _DETECT)
        sub(_INT16, "    # coordref: coordinates stay float (sub-voxel)\n")
        sub(_LOAD, _LOAD + "\n    UNetNodeTransformer._index_features = _coordref_index")
    else:
        raise ValueError(mode)
    compile(src, str(script), "exec")
    script.write_text(src)
