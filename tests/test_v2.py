"""Unit tests for the v2 retrieval maths (no GPU or model download needed).

python tests/test_v2.py        (or: python -m pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mmrag.fusion import (apply_hubness, bank_log_normaliser, collapse_max, early_fuse, early_fused_scores, l2norm,
                          max_pool_by_owner)
from mmrag.metrics import retrieval_metrics

rng = np.random.default_rng(0)


def test_early_fused_scores_matches_explicit_fusion():
    Q, I, C = l2norm(rng.normal(size=(7, 16))), l2norm(rng.normal(size=(11, 16))), l2norm(rng.normal(size=(11, 16)))
    for a in (0.0, 0.3, 0.85, 1.0):
        ref = Q @ early_fuse(I, C, a).T
        got = early_fused_scores(Q @ I.T, Q @ C.T, (I * C).sum(1), a)
        assert np.allclose(ref, got, atol=1e-5), a


def test_max_pool_by_owner():
    S = rng.normal(size=(5, 9)).astype(np.float32)
    owner = np.array([2, 0, 2, 1, 0, 0, 3, 1, 2])
    got = max_pool_by_owner(S, owner, 5)
    for g in range(4):
        assert np.allclose(got[:, g], S[:, owner == g].max(1))
    assert np.all(got[:, 4] == -1.0)   # item without vectors keeps the fill value


def test_bank_normaliser_brute_force():
    n_bank, n = 12, 6
    S = rng.normal(size=(n_bank, n))
    owner = np.tile(np.arange(n), 2)
    beta = 7.0
    got = bank_log_normaliser(S, owner, beta)
    for g in range(n):
        vals = beta * S[owner != g, g]
        assert np.isclose(got[g], np.log(np.mean(np.exp(vals))), atol=1e-4)
    adj = apply_hubness(S[:3], got, beta, 1.0)
    assert np.allclose(adj, S[:3] - (got - got.mean()) / beta, atol=1e-6)
    assert apply_hubness(S, got, beta, 0.0) is S


def test_collapse_max():
    assert collapse_max([(3, 0.1), (1, 0.9), (3, 0.7), (1, 0.2)]) == [(1, 0.9), (3, 0.7)]


def test_fast_ranks_match_reference_metrics():
    from experiments_v2 import gt_rank, metrics_from_ranks

    S = rng.normal(size=(200, 40)).astype(np.float32)
    gt = rng.integers(0, 40, size=200)
    ref = retrieval_metrics(S, gt)
    got = metrics_from_ranks(gt_rank(S, gt))
    for k in ("R@1", "R@5", "R@10", "MRR"):
        assert np.isclose(ref[k], got[k]), k


def test_rerank_ranks_match_bruteforce():
    from experiments_v2 import gt_rank, rerank_ranks_sub

    n_q, n, K = 300, 50, 8
    S = rng.normal(size=(n_q, n)).astype(np.float32)
    gt = rng.integers(0, n, size=n_q)
    tk = np.argsort(-S, 1)[:, :K]
    comb = rng.normal(size=(n_q, K)).astype(np.float32)
    got = rerank_ranks_sub(comb, tk, gt, gt_rank(S, gt))
    for q in range(n_q):
        order = list(tk[q][np.argsort(-comb[q], kind="stable")]) + [i for i in np.argsort(-S[q]) if i not in tk[q]]
        assert got[q] == order.index(gt[q]), q
    # all-tied rerank scores must keep the first-stage order (no credit for ties)
    tied = rerank_ranks_sub(np.zeros_like(comb), tk, gt, gt_rank(S, gt))
    assert np.array_equal(tied, gt_rank(S, gt))


def test_cv_never_uses_tuning_fold_for_reporting():
    from experiments_v2 import CV

    n_q, n = 40, 10
    gt = rng.integers(0, n, size=n_q)
    fold = (np.arange(n_q) % 2)
    seen = []

    def score(P, m):
        seen.append((P["x"], int(m.sum()), bool(m[0])))
        return np.where(np.arange(n)[None, :] == gt[m][:, None], P["x"], 0.0).astype(np.float32)

    cv = CV(fold, gt)
    S, picks = cv.pick("x", [0.0, 1.0], score)
    assert picks == [1.0, 1.0]
    assert np.all(S[np.arange(n_q), gt] == 1.0)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
