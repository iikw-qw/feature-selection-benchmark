import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import is_real


def selectors(k, seed=0):
    return {
        "none": None,
        "anova_f": SelectKBest(f_classif, k=k),
        "mutual_info": SelectKBest(lambda x, y: mutual_info_classif(x, y, random_state=seed), k=k),
        "l1_logistic": SelectFromModel(
            LogisticRegression(l1_ratio=1.0, solver="saga", C=0.15, max_iter=5000), max_features=k, threshold=-np.inf
        ),
        "rfe_logistic": RFE(LogisticRegression(max_iter=2000), n_features_to_select=k, step=0.2),
        "forest_importance": SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1), max_features=k, threshold=-np.inf
        ),
    }


def make_pipeline(selector):
    steps = [("scale", StandardScaler())]
    if selector is not None:
        steps.append(("select", selector))
    steps.append(("clf", LogisticRegression(max_iter=2000)))
    return Pipeline(steps)


def cv_score(x, y, selector, seed=0):
    """Selection happens inside every fold, so the held-out part never influences it."""
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_score(make_pipeline(selector), x, y, cv=cv, scoring="roc_auc")


def selected_names(x, y, selector):
    pipe = make_pipeline(selector).fit(x, y)
    mask = pipe.named_steps["select"].get_support()
    return list(x.columns[mask])


def benchmark(x, y, k=10, seed=0):
    rows = {}
    for name, sel in selectors(k, seed).items():
        scores = cv_score(x, y, sel, seed)
        row = {"auc_mean": scores.mean(), "auc_std": scores.std()}
        if sel is None:
            row["n_features"] = x.shape[1]
            row["real_share"] = float(np.mean([is_real(c) for c in x.columns]))
        else:
            names = selected_names(x, y, sel)
            row["n_features"] = len(names)
            row["real_share"] = float(np.mean([is_real(c) for c in names]))
            row["noise_picked"] = int(sum(c.startswith("noise_") for c in names))
        rows[name] = row
    return pd.DataFrame(rows).T


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def stability(x, y, name, k=10, n_boot=8, seed=0):
    """Average pairwise Jaccard similarity of the selected sets over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n_boot):
        idx = rng.choice(len(x), len(x), replace=True)
        sets.append(selected_names(x.iloc[idx], y.iloc[idx], selectors(k, seed)[name]))
    sims = [jaccard(sets[i], sets[j]) for i in range(n_boot) for j in range(i + 1, n_boot)]
    return float(np.mean(sims))


def score_vs_k(x, y, name, ks=(2, 5, 10, 20, 40), seed=0):
    return pd.Series({k: cv_score(x, y, selectors(k, seed)[name], seed).mean() for k in ks})
