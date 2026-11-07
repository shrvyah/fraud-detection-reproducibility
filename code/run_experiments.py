"""Run all experiments reported in the paper. Usage: python run_experiments.py <data_dir> <out_dir>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from frauddet import (ADRG, RuleFit, ZScore, best_f1_threshold, binary_metrics,
                      rule_to_text, score_metrics)

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
NEG_CAP = 250_000  # negatives kept for supervised training on PaySim


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_ulb():
    csv_path = DATA / "creditcard.csv"
    parquet_path = DATA / "ulb.parquet"
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.read_parquet(parquet_path)
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df["hour"] = (df["Time"] // 3600) % 24
    df["log_amount"] = np.log1p(df["Amount"])
    feats = [f"V{i}" for i in range(1, 29)] + ["Amount", "hour"]
    zpop = [f"V{i}" for i in range(1, 29)] + ["log_amount"]
    return dict(name="ULB", df=df, y=df["Class"].values, feats=feats,
                zpop=zpop, zent=[], time=df["Time"].values)


def load_paysim():
    df = pd.read_parquet(DATA / "paysim.parquet")
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    df["log_amount"] = np.log1p(df["amount"])
    # causal, entity-level statistics of each destination account, using only
    # that account's *previous* transactions (all transaction types)
    g = df.groupby("nameDest", sort=False)["log_amount"]
    n_prev = g.cumcount()
    s1 = g.cumsum() - df["log_amount"]
    s2 = df.assign(sq=df["log_amount"] ** 2).groupby("nameDest", sort=False)["sq"].cumsum() - df["log_amount"] ** 2
    mean_prev = s1 / n_prev.replace(0, np.nan)
    var_prev = (s2 / n_prev.replace(0, np.nan)) - mean_prev ** 2
    sd_prev = np.sqrt(var_prev.clip(lower=0)) + 0.1
    df["dest_prev_cnt"] = n_prev
    df["dest_z_amount"] = np.where(n_prev >= 2, (df["log_amount"] - mean_prev) / sd_prev, 0.0)
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["hour"] = df["step"] % 24
    df["amt_to_orig_bal"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["log_old_orig"] = np.log1p(df["oldbalanceOrg"])
    df["log_old_dest"] = np.log1p(df["oldbalanceDest"])
    df["log_ratio"] = np.log1p(df["amt_to_orig_bal"])
    feats = ["is_transfer", "amount", "oldbalanceOrg", "newbalanceOrig",
             "oldbalanceDest", "newbalanceDest", "hour", "amt_to_orig_bal",
             "dest_prev_cnt", "dest_z_amount"]
    zpop = ["log_amount", "log_old_orig", "log_old_dest", "log_ratio"]
    return dict(name="PaySim", df=df, y=df["isFraud"].values, feats=feats,
                zpop=zpop, zent=["dest_z_amount"], time=df["step"].values)


def time_split(n, a=0.6, b=0.8):
    i, j = int(n * a), int(n * b)
    return np.arange(0, i), np.arange(i, j), np.arange(j, n)


def sup_subsample(y_idx, y, seed):
    """All frauds + up to NEG_CAP negatives (row order preserved)."""
    rng = np.random.default_rng(1000 + seed)
    pos = y_idx[y[y_idx] == 1]; neg = y_idx[y[y_idx] == 0]
    if len(neg) > NEG_CAP:
        neg = rng.choice(neg, NEG_CAP, replace=False)
    return np.sort(np.r_[pos, neg])


def evaluate(name, yv, sv, yt, st, extra=None):
    thr = best_f1_threshold(yv, sv)
    m = binary_metrics(yt, st >= thr)
    m.update(score_metrics(yt, st)); m["threshold"] = thr; m["model"] = name
    if extra:
        m.update(extra)
    return m


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark(D):
    df, y, feats = D["df"], D["y"], D["feats"]
    X = df[feats].values.astype(np.float64)
    tr, va, te = time_split(len(df))
    info = dict(dataset=D["name"], n=len(df), n_fraud=int(y.sum()),
                fraud_rate=float(y.mean()),
                train=(len(tr), int(y[tr].sum())), val=(len(va), int(y[va].sum())),
                test=(len(te), int(y[te].sum())))
    print(info, flush=True)
    rows, rule_tables, adrg_stats, rulefit_info = [], [], [], []

    # Z-score (deterministic)
    z = ZScore(D["zpop"], D["zent"]).fit(df.iloc[tr], y[tr])
    zv, zt = z.score(df.iloc[va]), z.score(df.iloc[te])
    rows.append(evaluate("Z-score (val-tuned)", y[va], zv, y[te], zt, dict(seed=-1)))
    m = binary_metrics(y[te], zt >= 2.5); m.update(score_metrics(y[te], zt))
    m.update(model="Z-score (|z|>=2.5)", threshold=2.5, seed=-1); rows.append(m)

    # Logistic regression (deterministic)
    sub = sup_subsample(tr, y, 0)
    sc = StandardScaler().fit(X[sub])
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(sc.transform(X[sub]), y[sub])
    rows.append(evaluate("Logistic regression", y[va], lr.decision_function(sc.transform(X[va])),
                         y[te], lr.decision_function(sc.transform(X[te])), dict(seed=-1)))

    if D["name"] == "PaySim":
        m = binary_metrics(y[te], df["isFlaggedFraud"].values[te] == 1)
        m.update(model="Simulator rule (isFlaggedFraud)", seed=-1); rows.append(m)

    for seed in SEEDS:
        print(f"  seed {seed}", flush=True)
        # ADRG: own sampling over the full training window
        ad = ADRG(seed=seed).fit(X[tr], y[tr], X[va], y[va])
        sv, st = ad.score(X[va]), ad.score(X[te])
        m = binary_metrics(y[te], st > 0); m.update(score_metrics(y[te], st))
        m.update(model="ADRG", seed=seed, runtime_s=ad.stats_["t_total"],
                 n_rules=ad.stats_["n_selected"]); rows.append(m)
        adrg_stats.append(dict(seed=seed, **ad.stats_))
        if seed == 0:
            masks = ad._masks(X[te])
            for r, mk in sorted(zip(ad.rules_, masks), key=lambda t: -t[0]["val_tp"])[:5]:
                bm = binary_metrics(y[te], mk)
                rule_tables.append(dict(rule=rule_to_text(r["conds"], feats),
                                        n_conditions=len(r["conds"]),
                                        val_precision=r["val_precision"], val_tp=r["val_tp"],
                                        **{f"test_{k}": v for k, v in bm.items()}))
        # Hybrid: ADRG rules OR composite z above a validation-tuned threshold
        best = None
        for q in np.quantile(zv, [0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]):
            f1 = binary_metrics(y[va], (sv > 0) | (zv >= q))["f1"]
            if best is None or f1 > best[0]:
                best = (f1, q)
        m = binary_metrics(y[te], (st > 0) | (zt >= best[1]))
        m.update(model="Hybrid (ADRG OR Z)", seed=seed, threshold=float(best[1]))
        m["added_tp_over_adrg"] = int(((zt >= best[1]) & ~(st > 0) & (y[te] == 1)).sum())
        rows.append(m)

        sub = sup_subsample(tr, y, seed)
        rf_rule = RuleFit(seed=seed).fit(X[sub], y[sub], X[va], y[va])
        rows.append(evaluate("RuleFit", y[va], rf_rule.score(X[va]), y[te], rf_rule.score(X[te]),
                             dict(seed=seed, runtime_s=rf_rule.t_total_,
                                  n_rules=rf_rule.n_rules_selected_)))
        rulefit_info.append(dict(seed=seed, C=rf_rule.C_, candidate_rules=rf_rule.n_candidate_rules_,
                                 selected_rules=rf_rule.n_rules_selected_,
                                 selected_linear=rf_rule.n_linear_selected_,
                                 runtime_s=rf_rule.t_total_,
                                 top_rules=rf_rule.top_rules(feats) if seed == 0 else None))

        t0 = time.perf_counter()
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                    random_state=seed, n_jobs=1).fit(X[sub], y[sub])
        rows.append(evaluate("Random forest", y[va], rf.predict_proba(X[va])[:, 1], y[te],
                             rf.predict_proba(X[te])[:, 1],
                             dict(seed=seed, runtime_s=time.perf_counter() - t0)))
        t0 = time.perf_counter()
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                             early_stopping=False, random_state=seed).fit(X[sub], y[sub])
        rows.append(evaluate("Gradient boosting (HGB)", y[va], hgb.predict_proba(X[va])[:, 1], y[te],
                             hgb.predict_proba(X[te])[:, 1],
                             dict(seed=seed, runtime_s=time.perf_counter() - t0)))

    pd.DataFrame(rows).to_csv(OUT / f"{D['name']}_results.csv", index=False)
    pd.DataFrame(rule_tables).to_csv(OUT / f"{D['name']}_adrg_top_rules.csv", index=False)
    pd.DataFrame(adrg_stats).to_csv(OUT / f"{D['name']}_adrg_stats.csv", index=False)
    json.dump(dict(info=info, rulefit=rulefit_info), open(OUT / f"{D['name']}_info.json", "w"),
              indent=2, default=str)


# ---------------------------------------------------------------------------
# Periodic retraining (PaySim, weekly cycles over 31 simulated days)
# ---------------------------------------------------------------------------

def retraining(D):
    df, y, feats = D["df"], D["y"], D["feats"]
    X = df[feats].values.astype(np.float64)
    day = (df["step"].values - 1) // 24
    periods = [(7, 14), (14, 21), (21, 28), (28, 31)]
    rows = []
    for seed in [0, 1, 2]:
        def fit_window(d0, d1, seed):
            w = np.flatnonzero((day >= d0) & (day < d1))
            cut = np.flatnonzero(day[w] >= d1 - 2)[0]  # last 2 days of window = validation
            trw, vaw = w[:cut], w[cut:]
            ad = ADRG(seed=seed).fit(X[trw], y[trw], X[vaw], y[vaw])
            sub = sup_subsample(trw, y, seed)
            rf_ = RuleFit(seed=seed).fit(X[sub], y[sub], X[vaw], y[vaw])
            thr = best_f1_threshold(y[vaw], rf_.score(X[vaw]))
            return ad, rf_, thr
        static = fit_window(0, 7, seed)
        prev_sigs = {tuple(sorted({c[0] for c in r["conds"]})) for r in static[0].rules_}
        prev_rules = {r["conds"] for r in static[0].rules_}
        for (a, b) in periods:
            te = np.flatnonzero((day >= a) & (day < b))
            retr = fit_window(a - 7, a, seed)
            for mode, (ad, rf_, thr) in [("static", static), ("rolling 7-day", retr)]:
                m = binary_metrics(y[te], ad.predict(X[te]))
                rows.append(dict(seed=seed, period=f"days {a}-{b-1}", mode=mode, model="ADRG",
                                 n_test=len(te), n_fraud=int(y[te].sum()), **m))
                m = binary_metrics(y[te], rf_.score(X[te]) >= thr)
                rows.append(dict(seed=seed, period=f"days {a}-{b-1}", mode=mode, model="RuleFit",
                                 n_test=len(te), n_fraud=int(y[te].sum()), **m))
            new_rules = [r for r in retr[0].rules_ if r["conds"] not in prev_rules]
            rows.append(dict(seed=seed, period=f"days {a}-{b-1}", mode="rolling 7-day", model="ADRG-novelty",
                             n_selected=len(retr[0].rules_), n_new_exact=len(new_rules),
                             n_new_feature_set=sum(tuple(sorted({c[0] for c in r["conds"]})) not in prev_sigs
                                                   for r in retr[0].rules_)))
            prev_rules = {r["conds"] for r in retr[0].rules_}
            prev_sigs = {tuple(sorted({c[0] for c in r["conds"]})) for r in retr[0].rules_}
        print("  retraining seed", seed, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "PaySim_retraining.csv", index=False)


# ---------------------------------------------------------------------------
# Runtime scaling (PaySim)
# ---------------------------------------------------------------------------

def scaling(D):
    df, y, feats = D["df"], D["y"], D["feats"]
    X = df[feats].values.astype(np.float64)
    rows = []
    for n in [50_000, 200_000, 800_000, len(df)]:
        rng = np.random.default_rng(7)
        idx = np.sort(rng.choice(len(df), n, replace=False)) if n < len(df) else np.arange(len(df))
        tr, va, _ = time_split(len(idx))
        tr, va = idx[tr], idx[va]
        for seed in [0, 1, 2]:
            ad = ADRG(seed=seed).fit(X[tr], y[tr], X[va], y[va])
            rf_ = RuleFit(seed=seed, Cs=(0.01,)).fit(X[tr], y[tr], X[va], y[va])
            rows.append(dict(n=n, n_train=len(tr), n_val=len(va), seed=seed,
                             adrg_s=ad.stats_["t_total"], adrg_generate_s=ad.stats_["t_generate"],
                             adrg_candidates=ad.stats_["n_raw"], rulefit_s=rf_.t_total_))
        print("  scaling", n, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "PaySim_scaling.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    with threadpool_limits(1):  # single CPU core for all timed components
        if which in ("all", "ulb"):
            benchmark(load_ulb())
        if which in ("all", "paysim", "retrain", "scaling"):
            P = load_paysim()
            if which in ("all", "paysim"):
                benchmark(P)
            if which in ("all", "retrain"):
                retraining(P)
            if which in ("all", "scaling"):
                scaling(P)
