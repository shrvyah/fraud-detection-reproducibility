"""Hybrid interpretable fraud detection: ADRG, RuleFit, Z-score anomaly layer.

Reference implementation for the paper. All randomness is seeded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(y, pred):
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    tp = int((y & pred).sum()); fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum()); tn = int((~y & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return dict(precision=prec, recall=rec, f1=f1, fpr=fpr, alerts=tp + fp, tp=tp)


def score_metrics(y, s):
    out = {}
    if len(np.unique(s)) > 1:
        out["auprc"] = float(average_precision_score(y, s))
        out["auroc"] = float(roc_auc_score(y, s))
    else:
        out["auprc"] = float(np.mean(y)); out["auroc"] = 0.5
    return out


def best_f1_threshold(y, s):
    """Threshold on score s that maximises F1 (chosen on validation only)."""
    order = np.argsort(-s, kind="stable")
    ys = np.asarray(y)[order].astype(float); ss = s[order]
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    prec = tp / k; rec = tp / max(ys.sum(), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
    # only consider cut points where the score changes
    valid = np.r_[ss[1:] != ss[:-1], True]
    i = int(np.argmax(np.where(valid, f1, -1)))
    return float(ss[i])


# ---------------------------------------------------------------------------
# Rule utilities
# ---------------------------------------------------------------------------

def _node_paths(tree):
    """Return dict node_id -> list of (feature, op, threshold) from root."""
    t = tree.tree_
    paths = {0: []}
    stack = [0]
    while stack:
        n = stack.pop()
        l, r = t.children_left[n], t.children_right[n]
        if l == -1:
            continue
        f, th = int(t.feature[n]), float(t.threshold[n])
        paths[l] = paths[n] + [(f, "<=", th)]
        paths[r] = paths[n] + [(f, ">", th)]
        stack += [l, r]
    return paths


def simplify(conds):
    """Merge redundant bounds on the same feature."""
    lo, hi = {}, {}
    for f, op, th in conds:
        if op == "<=":
            hi[f] = min(hi.get(f, np.inf), th)
        else:
            lo[f] = max(lo.get(f, -np.inf), th)
    out = []
    for f in sorted(set(lo) | set(hi)):
        if f in lo:
            out.append((f, ">", lo[f]))
        if f in hi:
            out.append((f, "<=", hi[f]))
    return tuple(out)


def rule_to_text(conds, names):
    return " AND ".join(f"{names[f]} {op} {th:.4g}" for f, op, th in conds)


# ---------------------------------------------------------------------------
# ADRG
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "max_depth": [4, 5, 6, 7, 8],
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": ["best", "random"],
    "max_features": ["sqrt", "log2", None],
}


def sample_configs(n, rng):
    keys = list(PARAM_GRID)
    cfgs = []
    for _ in range(n):
        cfgs.append({k: PARAM_GRID[k][rng.integers(len(PARAM_GRID[k]))] for k in keys})
    return cfgs


@dataclass
class ADRG:
    ratios: tuple = (1, 2, 3, 5, 8, 10)
    n_configs: int = 6
    min_alerts: int = 5
    min_precision: float = 0.15   # "fraud rate" threshold
    max_fpr: float = 0.08         # false positive rate threshold
    seed: int = 0
    rules_: list = field(default_factory=list)
    stats_: dict = field(default_factory=dict)

    def fit(self, Xtr, ytr, Xval, yval):
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        Xtr = np.asarray(Xtr, dtype=np.float32); ytr = np.asarray(ytr)
        Xval = np.asarray(Xval, dtype=np.float32); yval = np.asarray(yval).astype(bool)
        pos = np.flatnonzero(ytr == 1); neg = np.flatnonzero(ytr == 0)
        cfgs = sample_configs(self.n_configs, rng)
        cands = []  # (conds, val_mask)
        n_trees = 0
        for ratio in self.ratios:
            for cfg in cfgs:
                nneg = min(len(neg), ratio * len(pos))
                idx = np.r_[pos, rng.choice(neg, nneg, replace=False)]
                dt = DecisionTreeClassifier(random_state=int(rng.integers(1 << 30)), **cfg)
                dt.fit(Xtr[idx], ytr[idx]); n_trees += 1
                t = dt.tree_
                leaves_val = dt.apply(Xval)
                paths = _node_paths(dt)
                for leaf in np.flatnonzero(t.children_left == -1):
                    v = t.value[leaf][0]
                    if v[1] <= v[0]:
                        continue  # not a fraud leaf
                    cands.append((simplify(paths[leaf]), leaves_val == leaf))
        t_gen = time.perf_counter() - t0
        n_raw = len(cands)
        # evaluate on validation, dedupe identical rule logic
        seen = set(); evaluated = []
        nneg_val = max((~yval).sum(), 1)
        for conds, m in cands:
            if conds in seen:
                continue
            seen.add(conds)
            a = int(m.sum())
            if a < self.min_alerts:
                continue
            tp = int((m & yval).sum())
            evaluated.append(dict(conds=conds, mask=m, alerts=a, tp=tp,
                                  precision=tp / a, fpr=(a - tp) / nneg_val,
                                  recall=tp / max(yval.sum(), 1)))
        n_unique = len(seen); n_minvol = len(evaluated)
        passing = [r for r in evaluated
                   if r["precision"] >= self.min_precision and r["fpr"] <= self.max_fpr]
        n_thresh = len(passing)
        # greedy coverage: highest precision first, keep rule if it adds frauds
        # and its marginal alerts still meet the precision threshold;
        # rules with identical alert sets are treated as duplicates.
        passing.sort(key=lambda r: (-r["precision"], -r["tp"]))
        covered = np.zeros(len(yval), bool); sel = []; sigs = set()
        for r in passing:
            sig = hash(np.packbits(r["mask"]).tobytes())
            if sig in sigs:
                continue
            sigs.add(sig)
            new = r["mask"] & ~covered
            ntp = int((new & yval).sum()); na = int(new.sum())
            if ntp == 0 or ntp / na < self.min_precision:
                continue
            sel.append(r); covered |= r["mask"]
        self.rules_ = [dict(conds=r["conds"], val_precision=r["precision"],
                            val_tp=r["tp"], val_alerts=r["alerts"], val_fpr=r["fpr"])
                       for r in sel]
        self.stats_ = dict(n_trees=n_trees, n_raw=n_raw, n_unique=n_unique,
                           n_min_volume=n_minvol, n_threshold=n_thresh,
                           n_selected=len(sel), t_generate=t_gen,
                           t_total=time.perf_counter() - t0)
        return self

    def _masks(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = []
        for r in self.rules_:
            m = np.ones(len(X), bool)
            for f, op, th in r["conds"]:
                m &= (X[:, f] <= th) if op == "<=" else (X[:, f] > th)
            out.append(m)
        return out

    def score(self, X):
        """Max validation precision among firing rules (0 if none fire)."""
        s = np.zeros(len(X))
        for r, m in zip(self.rules_, self._masks(X)):
            s = np.where(m, np.maximum(s, r["val_precision"]), s)
        return s

    def predict(self, X):
        return self.score(X) > 0


# ---------------------------------------------------------------------------
# RuleFit (Friedman & Popescu 2008), classification variant
# ---------------------------------------------------------------------------

@dataclass
class RuleFit:
    n_estimators: int = 100
    max_depth: int = 3
    Cs: tuple = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
    seed: int = 0

    def _features(self, X):
        R = self.forest_.decision_path(np.asarray(X, dtype=np.float32))[0]
        R = R[:, self.rule_cols_].astype(np.float32)
        L = (np.clip(X, self.lo_, self.hi_) - self.mu_) / self.sd_
        return sparse.hstack([R, sparse.csr_matrix(L.astype(np.float32))]).tocsr()

    def fit(self, Xtr, ytr, Xval, yval):
        t0 = time.perf_counter()
        Xtr = np.asarray(Xtr, dtype=np.float64)
        self.forest_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight="balanced", random_state=self.seed, n_jobs=1,
            max_features="sqrt")
        self.forest_.fit(Xtr.astype(np.float32), ytr)
        # rule columns: every non-root node of every tree
        offsets = self.forest_.decision_path(Xtr[:1].astype(np.float32))[1]
        cols, conds = [], []
        for ti, est in enumerate(self.forest_.estimators_):
            paths = _node_paths(est)
            for node, c in paths.items():
                if node == 0:
                    continue
                cols.append(offsets[ti] + node); conds.append(simplify(c))
        # drop duplicate rules (identical simplified conditions)
        uniq = {}
        for c, k in zip(cols, conds):
            uniq.setdefault(k, c)
        self.rule_conds_ = list(uniq.keys()); self.rule_cols_ = np.array(list(uniq.values()))
        self.lo_ = np.quantile(Xtr, 0.025, axis=0); self.hi_ = np.quantile(Xtr, 0.975, axis=0)
        W = np.clip(Xtr, self.lo_, self.hi_)
        self.mu_ = W.mean(0); self.sd_ = W.std(0) + 1e-9
        Ftr = self._features(Xtr); Fval = self._features(np.asarray(Xval, dtype=np.float64))
        best = None
        for C in self.Cs:
            lr = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                    class_weight="balanced", max_iter=200,
                                    random_state=self.seed)
            lr.fit(Ftr, ytr)
            ap = average_precision_score(yval, lr.decision_function(Fval))
            if best is None or ap > best[0]:
                best = (ap, C, lr)
        self.C_, self.lr_ = best[1], best[2]
        coef = self.lr_.coef_[0]
        nr = len(self.rule_cols_)
        self.n_rules_selected_ = int((coef[:nr] != 0).sum())
        self.n_linear_selected_ = int((coef[nr:] != 0).sum())
        self.n_candidate_rules_ = nr
        self.t_total_ = time.perf_counter() - t0
        return self

    def score(self, X):
        return self.lr_.decision_function(self._features(np.asarray(X, dtype=np.float64)))

    def top_rules(self, names, k=5):
        coef = self.lr_.coef_[0][: len(self.rule_cols_)]
        idx = np.argsort(-np.abs(coef))[:k]
        return [(rule_to_text(self.rule_conds_[i], names), float(coef[i])) for i in idx if coef[i] != 0]


# ---------------------------------------------------------------------------
# Z-score anomaly layer
# ---------------------------------------------------------------------------

@dataclass
class ZScore:
    """Composite Z-score. Population baselines are estimated on non-fraud
    training rows; entity-level (personalised) z-scores are supplied as
    precomputed columns, computed causally from each entity's own history."""
    pop_cols: list
    entity_cols: list = field(default_factory=list)

    def fit(self, X: pd.DataFrame, y):
        legit = X[np.asarray(y) == 0]
        self.mu_ = legit[self.pop_cols].mean(); self.sd_ = legit[self.pop_cols].std() + 1e-9
        return self

    def score(self, X: pd.DataFrame):
        z = ((X[self.pop_cols] - self.mu_) / self.sd_).abs()
        if self.entity_cols:
            z = pd.concat([z, X[self.entity_cols].abs()], axis=1)
        # composite = root-mean-square of the per-feature |z|
        return np.sqrt((z.values ** 2).mean(1))
