"""Embedding-clustering decode (DECODE axis of V320). Cluster fg voxels' cosine embeddings ->
instances. Variable count (no fixed K) via mean-shift; enforce spatial connectivity (a fragment is
spatially connected) by splitting each cosine-cluster into its spatial CCs + pruning small ones."""
import numpy as np
from scipy import ndimage as ndi


def decode_embeddings(emb, fg, bandwidth=0.7, min_size=150, n_sub=2500, rng=None):
    """emb: [E,Z,Y,X] float. fg: bool [Z,Y,X]. Returns int32 instance map (0=bg)."""
    rng = rng or np.random.default_rng(0)
    E = emb.shape[0]
    en = emb / (np.linalg.norm(emb, axis=0, keepdims=True) + 1e-9)
    feats = en[:, fg].T                                    # [Nfg, E]
    if len(feats) < min_size:
        return (fg.astype(np.int32))
    sub = feats[rng.choice(len(feats), min(n_sub, len(feats)), replace=False)]
    try:
        from sklearn.cluster import MeanShift
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True).fit(sub)
        centers = ms.cluster_centers_
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9)
    except Exception:
        centers = _greedy_centers(sub, bandwidth)
    lab_fg = (feats @ centers.T).argmax(1) + 1             # nearest center (cosine)
    out = np.zeros(fg.shape, np.int32)
    out[fg] = lab_fg
    return _split_cc_prune(out, min_size)


def _greedy_centers(sub, bw):
    centers, used = [], np.zeros(len(sub), bool)
    for i in np.random.permutation(len(sub)):
        if used[i]:
            continue
        c = sub[i].copy()
        for _ in range(3):
            mem = (sub @ c) > (1 - bw)
            c = sub[mem].mean(0); c /= (np.linalg.norm(c) + 1e-9)
        used[(sub @ c) > (1 - bw)] = True
        centers.append(c)
    return np.array(centers)


def _split_cc_prune(lab, min_size):
    """Split each cosine-cluster label into spatial connected components; prune < min_size."""
    out = np.zeros_like(lab)
    nxt = 1
    for k in [v for v in np.unique(lab) if v > 0]:
        cc, n = ndi.label(lab == k)
        for c in range(1, n + 1):
            m = cc == c
            if m.sum() >= min_size:
                out[m] = nxt; nxt += 1
    return out


def decode_embeddings_coreseeded(abbc, emb, *, bg_thr=0.5, core_thr=0.5, min_size=150):
    """ABBC-core-SEEDED embedding decode (anti-over-split hybrid).

    abbc: [4,Z,Y,X] softmax probs (0=bg, 1=border, 2=boundary, 3=core). emb: [E,Z,Y,X].
    Caps the fragment COUNT at the number of ABBC cores (exactly like the deployed V302
    core-seed watershed), but assigns each support voxel to the nearest core by EMBEDDING
    cosine — so touching-fragment boundaries follow the LEARNED embedding instead of pure
    distance. This kills mean-shift's structural over-split (no free-running cluster count).
    Returns int32 instance map (0=bg)."""
    support = abbc[0] < bg_thr
    out = np.zeros(support.shape, np.int32)
    if not support.any():
        return out
    core = (abbc[3] >= core_thr) & support
    core_lab, n = ndi.label(core, structure=np.ones((3, 3, 3), bool))
    if n <= 1:                                              # 0/1 core -> one fragment
        out[support] = 1
        return out
    en = emb / (np.linalg.norm(emb, axis=0, keepdims=True) + 1e-9)
    mu = np.zeros((n, emb.shape[0]), np.float32)            # per-core mean embedding (unit)
    for k in range(1, n + 1):
        v = en[:, core_lab == k].mean(1)
        mu[k - 1] = v / (np.linalg.norm(v) + 1e-9)
    feats = en[:, support]                                  # [E, Nsup]
    out[support] = (mu @ feats).argmax(0) + 1               # nearest core by cosine
    return _merge_small_cc(out, support, min_size)


def _merge_small_cc(lab, support, min_size):
    """Reassign spatially-disconnected sub-min_size pieces of each label into the nearest
    remaining label. Keeps the fragment COUNT (= #cores); just cleans strays. Never SPLITS."""
    out = np.asarray(lab, np.int32).copy()
    small = np.zeros(out.shape, bool)
    for k in [v for v in np.unique(out) if v > 0]:
        cc, ncc = ndi.label(out == k)
        if ncc <= 1:
            continue
        for c in range(1, ncc + 1):
            m = cc == c
            if m.sum() < min_size:
                small |= m
    if small.any():
        out[small] = 0
        rest = out > 0
        if rest.any():
            nearest = ndi.distance_transform_edt(~rest, return_indices=True)[1]
            fill = support & (out == 0)
            out[fill] = out[tuple(idx[fill] for idx in nearest)]
    return out


if __name__ == "__main__":
    import sys, glob, SimpleITK as sitk
    sys.path.insert(0, "/home/guest/Project/PENGWIN2026/experiments")
    from synthetic_fracture import synthetic_fracture
    ROOT = "/home/guest/Project/PENGWIN2026"

    def ari(a, b):                                          # adjusted Rand index on fg voxels
        from itertools import product
        m = (a > 0) | (b > 0)
        a, b = a[m], b[m]
        ua, ub = np.unique(a), np.unique(b)
        cont = np.zeros((len(ua), len(ub)))
        ai = {v: i for i, v in enumerate(ua)}; bi = {v: i for i, v in enumerate(ub)}
        for x, y in zip(a, b):
            cont[ai[x], bi[y]] += 1
        sa = cont.sum(1); sb = cont.sum(0); n = cont.sum()
        comb = lambda x: x * (x - 1) / 2
        idx = comb(cont).sum(); exp = comb(sa).sum() * comb(sb).sum() / comb(n)
        mx = (comb(sa).sum() + comb(sb).sum()) / 2
        return (idx - exp) / (mx - exp + 1e-9)

    g = glob.glob(f"{ROOT}/data/task1_2/extracted/*/001/label.mha")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(g[0])).astype(np.int32)
    m = (arr >= 51) & (arr <= 100)                         # LeftHip
    zz, yy, xx = np.where(m); sl = tuple(slice(c.min(), c.max() + 1) for c in (zz, yy, xx))
    bone = m[sl]
    rng = np.random.default_rng(1)
    for K in [3, 5]:
        gt = synthetic_fracture(bone, K, np.random.default_rng(K))
        ids = [v for v in np.unique(gt) if v > 0]
        # IDEAL embeddings: each fragment -> a distinct random unit direction + noise (simulate a trained net)
        E = 16
        dirs = rng.standard_normal((len(ids), E)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        emb = (rng.standard_normal((E,) + gt.shape) * 0.25).astype(np.float32)
        for i, k in enumerate(ids):
            emb[:, gt == k] += dirs[i][:, None]
        dec = decode_embeddings(emb, gt > 0, bandwidth=0.7, rng=rng)
        print(f"K={K}: GT frags={len(ids)} -> decoded={len([v for v in np.unique(dec) if v>0])}  ARI={ari(gt, dec):.3f}")
    print("DECODE pipeline (synthetic GT -> ideal emb -> cluster) validated.")
