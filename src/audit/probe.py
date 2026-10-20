"""Linear probe on frozen intermediate features (H1).

If the forget-class concept is still linearly decodable from the unlearned model's
mid-block features, "unlearning" did not change the representation.  The probe is
an L2-regularised logistic regression (forget class vs the rest for diffusion;
forget author vs retain for the LLM), trained and tested on disjoint splits.
``probe_accuracy_np`` is the numpy/sklearn core used by the tests.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def probe_accuracy_np(feats_train: np.ndarray, y_train: Sequence[int], feats_test: np.ndarray, y_test: Sequence[int], C: float = 1.0, seed: int = 0) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(feats_train)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(feats_train), np.asarray(y_train))
    acc = float(clf.score(sc.transform(feats_test), np.asarray(y_test)))
    # balanced accuracy too, since the forget class is a minority
    pred = clf.predict(sc.transform(feats_test))
    yt = np.asarray(y_test)
    tpr = float((pred[yt == 1] == 1).mean()) if (yt == 1).any() else float("nan")
    tnr = float((pred[yt == 0] == 0).mean()) if (yt == 0).any() else float("nan")
    return {"probe_accuracy": acc, "probe_balanced_accuracy": float(np.nanmean([tpr, tnr])), "probe_forget_recall": tpr}


def extract_diffusion_features(model, diff, dataset, indices, timestep: int, device, batch: int = 128) -> np.ndarray:
    import torch

    from ..diffusion.data import make_loader

    loader = make_loader(dataset, indices, batch, False, 0, 2, drop_last=False)
    feats = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t = torch.full((x.shape[0],), int(timestep), device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            xt = diff.q_sample(x, t, noise)
            feats.append(model.features(xt, t, y).float().cpu().numpy())
    return np.concatenate(feats)


def probe_diffusion(model, diff, dataset, labels: np.ndarray, forget_class: int, cfg: Dict, device, seed: int = 0) -> Dict[str, float]:
    p = cfg["audit"]["probe"]
    rng = np.random.default_rng(seed)
    fc_idx = np.where(labels == forget_class)[0]
    other = np.where(labels != forget_class)[0]
    n = min(int(p.get("n_per_class", 1500)), len(fc_idx))
    fsel = rng.choice(fc_idx, size=n, replace=False)
    osel = rng.choice(other, size=n, replace=False)
    idx = np.concatenate([fsel, osel])
    y = np.concatenate([np.ones(len(fsel)), np.zeros(len(osel))]).astype(int)
    feats = extract_diffusion_features(model, diff, dataset, idx.tolist(), int(p.get("timestep", 200)), device)
    perm = rng.permutation(len(idx))
    cut = int(0.7 * len(idx))
    tr, te = perm[:cut], perm[cut:]
    return probe_accuracy_np(feats[tr], y[tr], feats[te], y[te], float(p.get("C", 1.0)), seed)
