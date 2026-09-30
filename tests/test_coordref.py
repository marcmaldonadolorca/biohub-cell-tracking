"""Pruebas de la cabeza de refinado sin datos de la competición: comprueban el contrato del módulo."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coordref"))
import coordref  # noqa: E402


def _cabeza_temporal(tmp_path, gate=True):
    torch.manual_seed(0)
    h = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    p = tmp_path / "head.pt"
    torch.save({"state_dict": h.state_dict(), "mean": torch.zeros(224), "scale": torch.ones(224),
                "max_shift_um": 2.0, "kind": "mlp", "puerta_ok": gate}, p)
    return p


def test_features_224():
    feat = torch.randn(1, 32, 8, 16, 16)
    arr = np.array([[0, 3, 5, 7], [0, 0, 0, 0]], dtype=np.int16)
    x = coordref.neighbour_features(arr, feat)
    assert x.shape == (2, 224)
    assert torch.allclose(x[0, :32], feat[0, :, 3, 5, 7])


def test_desplazamiento_acotado(tmp_path, monkeypatch):
    monkeypatch.setenv("BIOHUB_COORDREF_HEAD_PATH", str(_cabeza_temporal(tmp_path)))
    coordref._CACHE = None
    feat = torch.randn(1, 32, 8, 16, 16) * 50
    arr = np.array([[0, 4, 8, 8], [1, 2, 3, 4], [2, 6, 12, 1]], dtype=np.int16)
    out = coordref.refine(arr, feat)
    movido = np.linalg.norm((out[:, 1:] - arr[:, 1:]) * coordref.GRID_UM, axis=1)
    assert out.dtype == np.float32 and movido.max() <= 2.0 + 1e-4


def test_puerta_rechaza_cabeza_no_validada(tmp_path):
    p = _cabeza_temporal(tmp_path, gate=False)
    try:
        coordref.load_head(p)
    except RuntimeError:
        return
    raise AssertionError("una cabeza sin puerta_ok debería rechazarse")


def test_trilineal_igual_a_gather_en_enteros():
    feat = torch.randn(1, 32, 8, 16, 16)
    coords = torch.tensor([[[3.0, 5.0, 7.0], [0.0, 15.0, 2.0]]])
    mask = torch.tensor([[True, True]])
    out = coordref.index_features(None, feat, coords, mask)
    esperado = feat[0, :, [3, 0], [5, 15], [7, 2]].T
    assert torch.allclose(out[0], esperado, atol=1e-6)
