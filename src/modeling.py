"""Temporal split, baseline + three models, validation-based tuning/threshold, single test evaluation."""
import itertools
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C
from . import features as F
from .io_utils import md_table
from .numpy_ops import log1p_transform

TARGET = "is_late"


# ----------------------------------------------------------------------------- split
def temporal_split(df: pd.DataFrame, train_frac=C.TRAIN_FRAC, val_frac=C.VAL_FRAC):
    """Split by order_approved_at into train < val < test (test = most recent). No shuffling."""
    d = df.sort_values(["order_approved_at", "order_id"]).reset_index(drop=True)
    n = len(d)
    t1 = d["order_approved_at"].iloc[int(n * train_frac)]
    t2 = d["order_approved_at"].iloc[int(n * (train_frac + val_frac))]
    tr = d[d["order_approved_at"] < t1]
    va = d[(d["order_approved_at"] >= t1) & (d["order_approved_at"] < t2)]
    te = d[d["order_approved_at"] >= t2]
    return tr, va, te


def _X(d: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix; categorical columns as object with np.nan (sklearn-friendly)."""
    X = d[F.FEATURES].copy()
    for c in F.CATEGORICAL:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in F.NUMERIC:
        X[c] = X[c].astype("float64")
    return X


# ----------------------------------------------------------------------------- pipelines
def make_preprocessor(scale_and_log: bool) -> ColumnTransformer:
    log_cols = [c for c in F.LOG_COLS]
    plain = [c for c in F.NUMERIC if c not in log_cols]
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                          sparse_output=False))])
    if scale_and_log:
        num_log = Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("log", FunctionTransformer(log1p_transform, feature_names_out="one-to-one")),
                            ("sc", StandardScaler())])
        num_plain = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    else:
        num_log = Pipeline([("imp", SimpleImputer(strategy="median"))])
        num_plain = Pipeline([("imp", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("num_log", num_log, log_cols), ("num", num_plain, plain),
                              ("cat", cat, F.CATEGORICAL)])


def make_model(kind: str, params: dict) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=C.SEED, **params)
        return Pipeline([("prep", make_preprocessor(True)), ("clf", clf)])
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, early_stopping=False,
                                             random_state=C.SEED, **params)
        return Pipeline([("prep", make_preprocessor(False)), ("clf", clf)])
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=C.SEED, **params)
        return Pipeline([("prep", make_preprocessor(False)), ("clf", clf)])
    raise ValueError(kind)


GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0]},
    "hgb": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]},
    "rf": {"min_samples_leaf": [10, 30, 100], "max_depth": [10, None]},
}


def grid(g: dict):
    keys = list(g)
    for vals in itertools.product(*g.values()):
        yield dict(zip(keys, vals))


# ----------------------------------------------------------------------------- metrics
def metrics_at(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "average_precision": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(proba)) > 1 else 0.5,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "threshold": float(thr),
    }


def best_f1_threshold(y, proba) -> float:
    p, r, t = precision_recall_curve(y, proba)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.nanargmax(f1))])


# ----------------------------------------------------------------------------- main
def run_modeling(df: pd.DataFrame) -> dict:
    # ---- forbidden-column guard (also unit-tested) ----
    assert not (set(F.FEATURES) & C.FORBIDDEN_COLUMNS), "forbidden column among features"

    tr, va, te = temporal_split(df)
    assert tr.order_approved_at.max() < va.order_approved_at.min() <= va.order_approved_at.max() < te.order_approved_at.min()
    split_tbl = pd.DataFrame([
        {"split": n, "start": d.order_approved_at.min(), "end": d.order_approved_at.max(), "n": len(d),
         "n_late": int(d[TARGET].sum()), "late_rate": d[TARGET].mean()}
        for n, d in (("train", tr), ("validation", va), ("test (final)", te))])
    split_tbl.to_csv(C.TABLE_DIR / "temporal_split.csv", index=False)

    Xtr, ytr = _X(tr), tr[TARGET].astype(int).to_numpy()
    Xva, yva = _X(va), va[TARGET].astype(int).to_numpy()
    Xte, yte = _X(te), te[TARGET].astype(int).to_numpy()

    log_rows = []

    def log(model, stage, params, val, test=None):
        row = {"model": model, "stage": stage, "params": json.dumps(params, default=str), "seed": C.SEED,
               "n_features": len(F.FEATURES), "features": ";".join(F.FEATURES)}
        row.update({f"val_{k}": v for k, v in val.items()})
        row.update({f"test_{k}": v for k, v in (test or {}).items()})
        log_rows.append(row)

    # ---- baseline (Dummy prior; hard prediction = majority class, no threshold tuning) ----
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    d_val, d_te = dummy.predict_proba(Xva)[:, 1], dummy.predict_proba(Xte)[:, 1]
    base_thr = 0.5  # prior ~7% < 0.5 => predicts "not late" for everyone
    base_val, base_test = metrics_at(yva, d_val, base_thr), metrics_at(yte, d_te, base_thr)
    log("baseline_dummy_prior", "final", {"strategy": "prior"}, base_val, base_test)
    allpos_test = metrics_at(yte, np.ones_like(yte, dtype=float), 0.5)
    allpos_val = metrics_at(yva, np.ones_like(yva, dtype=float), 0.5)
    log("baseline_always_late", "final", {"rule": "predict late for every order"}, allpos_val, allpos_test)

    # ---- tuning on validation only ----
    best = {}
    val_proba = {}
    for kind, g in GRIDS.items():
        cands = []
        for params in grid(g):
            pipe = make_model(kind, params).fit(Xtr, ytr)
            p = pipe.predict_proba(Xva)[:, 1]
            thr = best_f1_threshold(yva, p)
            m = metrics_at(yva, p, thr)
            log(kind, "tuning", params, m)
            cands.append((m["average_precision"], params, thr, pipe, p, m))
        cands.sort(key=lambda c: c[0], reverse=True)
        ap, params, thr, pipe, p, m = cands[0]
        best[kind] = {"params": params, "thr": thr, "pipe": pipe, "val": m}
        val_proba[kind] = p
        print(f"   best {kind}: val AP={ap:.4f}  params={params}  thr={thr:.3f}")

    # ---- final test evaluation: ONE shot per selected model, threshold from validation ----
    test_proba = {}
    for kind, b in best.items():
        p = b["pipe"].predict_proba(Xte)[:, 1]
        test_proba[kind] = p
        b["test"] = metrics_at(yte, p, b["thr"])
        log(kind, "final", b["params"], b["val"], b["test"])

    selected = max(best, key=lambda k: best[k]["val"]["average_precision"])  # chosen on VALIDATION only
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(C.OUT / "experiments_log.csv", index=False)

    # ---- comparison table ----
    comp = [{"model": "Baseline: Dummy(prior)", **{k: base_test[k] for k in ("accuracy", "precision", "recall", "f1", "average_precision")},
             "val_average_precision": base_val["average_precision"]},
            {"model": "Baseline: always late", **{k: allpos_test[k] for k in ("accuracy", "precision", "recall", "f1", "average_precision")},
             "val_average_precision": allpos_val["average_precision"]}]
    names = {"logreg": "Logistic Regression", "hgb": "HistGradientBoosting", "rf": "Random Forest"}
    for k, b in best.items():
        comp.append({"model": names[k] + (" (selected)" if k == selected else ""),
                     **{m: b["test"][m] for m in ("accuracy", "precision", "recall", "f1", "average_precision")},
                     "val_average_precision": b["val"]["average_precision"]})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(C.TABLE_DIR / "model_comparison_test.csv", index=False)
    pd.DataFrame([{"model": names[k], **b["val"], "params": json.dumps(b["params"])} for k, b in best.items()]).to_csv(
        C.TABLE_DIR / "model_validation_metrics.csv", index=False)
    cm_rows = [{"model": names[k], "split": s, "tn": m["tn"], "fp": m["fp"], "fn": m["fn"], "tp": m["tp"], "threshold": m["threshold"]}
               for k, b in best.items() for s, m in (("validation", b["val"]), ("test", b["test"]))]
    pd.DataFrame(cm_rows).to_csv(C.TABLE_DIR / "confusion_matrices.csv", index=False)

    beats = best[selected]["test"]["average_precision"] > base_test["average_precision"]

    # ---- figures ----
    _plot_pr(yva, yte, val_proba, test_proba, names, base_test["average_precision"])
    _plot_confusions(best, names)

    # ---- importance ----
    imp_tables = _importance(best, selected, Xva, yva, names)
    # ---- error analysis ----
    err = _error_analysis(te, test_proba[selected], best[selected]["thr"], names[selected])

    decisions_md = "\n".join([
        "Temporal split on `order_approved_at` (no shuffling), proportions 70/15/15 of eligible orders:",
        "",
        md_table(split_tbl.assign(start=split_tbl.start.astype(str), end=split_tbl.end.astype(str))),
        "",
        f"- Preprocessing (imputer, log1p, scaler, one-hot) lives inside sklearn Pipelines fitted on TRAIN only.",
        f"- Hyper-parameters: grid search; every candidate fitted on train, scored on validation (Average Precision). "
        f"Threshold = F1-optimal on validation. Test used once per selected model.",
        f"- Selected model (by validation AP): **{names[selected]}**. Beats Dummy on test AP: **{beats}**.",
        "- Final models are NOT refit on train+validation, so the validation-derived threshold remains valid.",
    ])
    summary = {
        "split": split_tbl.assign(start=split_tbl.start.astype(str), end=split_tbl.end.astype(str)).to_dict("records"),
        "selected_model": names[selected], "selected_beats_baseline_test_AP": bool(beats),
        "baseline_test": base_test, "always_late_test": allpos_test,
        "models": {names[k]: {"params": b["params"], "threshold": b["thr"], "val": b["val"], "test": b["test"]} for k, b in best.items()},
        "n_features": len(F.FEATURES),
    }
    return {"summary": summary, "decisions_md": decisions_md, "importance": imp_tables, "errors": err}


# ----------------------------------------------------------------------------- figures / analysis
def _plot_pr(yva, yte, val_proba, test_proba, names, prior):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, y, probs, title in ((axes[0], yva, val_proba, "Validation"), (axes[1], yte, test_proba, "Test (final)")):
        for k, p in probs.items():
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, label=f"{names[k]} (AP={average_precision_score(y, p):.3f})")
        ax.axhline(y.mean(), color="gray", ls="--", label=f"Baseline prevalence ({y.mean():.3f})")
        ax.set_xlabel("Recall (late class)"); ax.set_title(f"Precision-Recall curve - {title}")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    axes[0].set_ylabel("Precision (late class)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_pr_curves.png", dpi=200); plt.close(fig)


def _plot_confusions(best, names):
    fig, axes = plt.subplots(1, len(best), figsize=(4.6 * len(best), 4.2))
    for ax, (k, b) in zip(np.atleast_1d(axes), best.items()):
        m = b["test"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Pred on-time", "Pred late"]); ax.set_yticks([0, 1], ["Actual on-time", "Actual late"])
        ax.set_title(f"{names[k]}\nthr={b['thr']:.2f} (test, n={cm.sum():,})", fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_confusion_matrices_test.png", dpi=200); plt.close(fig)


def _importance(best, selected, Xva, yva, names):
    out = {}
    lr = best["logreg"]["pipe"]
    coef = pd.DataFrame({"feature": lr.named_steps["prep"].get_feature_names_out(),
                         "coefficient": lr.named_steps["clf"].coef_[0]})
    coef["abs"] = coef.coefficient.abs()
    coef = coef.sort_values("abs", ascending=False).drop(columns="abs")
    coef.to_csv(C.TABLE_DIR / "logreg_coefficients.csv", index=False)
    out["logreg"] = coef

    pipe = best[selected]["pipe"]
    pi = permutation_importance(pipe, Xva, yva, scoring="average_precision", n_repeats=3,
                                random_state=C.SEED, n_jobs=-1)
    pit = pd.DataFrame({"feature": Xva.columns, "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std}).sort_values("importance_mean", ascending=False)
    pit.to_csv(C.TABLE_DIR / "permutation_importance_selected.csv", index=False)
    out["permutation"] = pit

    top = pit.head(12).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(top.feature, top.importance_mean, xerr=top.importance_std, color="#2a6f97")
    axes[0].set_xlabel("Drop in validation Average Precision when shuffled")
    axes[0].set_title(f"Permutation importance - {names[selected]}")
    c = coef.head(12).iloc[::-1]
    axes[1].barh(c.feature, c.coefficient, color=np.where(c.coefficient > 0, "#c1121f", "#2a6f97"))
    axes[1].set_xlabel("Standardised coefficient (+ = higher late risk)"); axes[1].set_title("Logistic Regression coefficients (top 12 by |coef|)")
    for ax in axes: ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig_feature_importance.png", dpi=200); plt.close(fig)
    return out


def _error_analysis(te, proba, thr, name):
    d = te.copy()
    d["proba"] = proba
    d["pred"] = (proba >= thr).astype(int)
    d["y"] = d[TARGET].astype(int)
    d["outcome"] = np.select([(d.y == 1) & (d.pred == 1), (d.y == 0) & (d.pred == 1), (d.y == 1) & (d.pred == 0)],
                             ["TP", "FP", "FN"], "TN")
    d["order_size"] = pd.cut(d["n_items"], [0, 1, 2, 100], labels=["1 item", "2 items", "3+ items"])

    def summarise(col, min_n=C.MIN_GROUP_N):
        g = d.groupby(col, observed=True).agg(n=("y", "size"), late_rate=("y", "mean"), predicted_late_rate=("pred", "mean"),
                                              tp=("outcome", lambda s: (s == "TP").sum()), fp=("outcome", lambda s: (s == "FP").sum()),
                                              fn=("outcome", lambda s: (s == "FN").sum()))
        g["precision"] = g.tp / (g.tp + g.fp).replace(0, np.nan)
        g["recall"] = g.tp / (g.tp + g.fn).replace(0, np.nan)
        g["reliable_n_ge_100"] = g.n >= min_n
        return g.reset_index().sort_values("n", ascending=False)

    for col, fn in (("customer_state", "errors_by_customer_state"), ("seller_state", "errors_by_seller_state"),
                    ("main_category", "errors_by_category"), ("order_size", "errors_by_order_size"),
                    ("multi_seller", "errors_by_multi_seller")):
        summarise(col).to_csv(C.TABLE_DIR / f"{fn}.csv", index=False)
    prof_cols = ["planned_days", "total_price", "total_freight", "total_weight_g", "same_state", "approval_delay_hours", "seller_prior_orders"]
    prof = d.groupby("outcome")[prof_cols].median().T
    prof.insert(0, "feature", prof.index)
    prof.to_csv(C.TABLE_DIR / "error_profile_by_outcome_median.csv", index=False)
    d[["order_id", "customer_state", "seller_state", "main_category", "n_items", "planned_days", "y", "pred", "proba", "outcome"]].to_csv(
        C.DATA_PROCESSED / "test_predictions.csv", index=False)
    return {"profile": prof}

