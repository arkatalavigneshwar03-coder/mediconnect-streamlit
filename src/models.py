"""
Train and apply the four MediConnect models.

  1. Classification : will the respondent subscribe? (interested = Q29 >= 4)
  2. Clustering     : K-means segments ranked High / Medium / Low probable
  3. Association    : Apriori rules over the 10-service baskets (E1-E10)
  4. Regression     : monthly willingness to pay in AED (Q31)

No Streamlit import here, so the module can be tested and reused on its own.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import apriori
from .config import (CATEG_AD, CATEG_REG, CLU, LIVE_CSV, NUMERIC_AD, NUMERIC_REG, SEED, SEGS, SERVICES, SURVEY_CSV,
                     SVC_LABEL, pretty)


# ------------------------------------------------------------------ data
def load_data():
    """Read the survey CSVs; generate them first if they are missing (same seed -> same data)."""
    if not SURVEY_CSV.exists() or not LIVE_CSV.exists():
        from .data_gen import generate
        s, l = generate()
        SURVEY_CSV.parent.mkdir(parents=True, exist_ok=True)
        s.to_csv(SURVEY_CSV, index=False)
        l.to_csv(LIVE_CSV, index=False)
    survey = pd.read_csv(SURVEY_CSV, keep_default_na=False)
    live = pd.read_csv(LIVE_CSV, keep_default_na=False)
    return survey, live


def feature_names(cats: dict, numeric: list, categ: list) -> list:
    return list(numeric) + [f"{c}={v}" for c in categ for v in cats[c]]


def encode(frame: pd.DataFrame, names: list) -> np.ndarray:
    """Encode rows using names like 'income_band' (numeric) or 'insurance=Basic' (one-hot)."""
    cols = []
    for n in names:
        if "=" in n:
            c, v = n.split("=", 1)
            cols.append((frame[c].astype(str) == v).astype(float).values)
        else:
            cols.append(frame[n].astype(float).values)
    return np.column_stack(cols)


@dataclass
class Models:
    cats: dict
    # classification
    cls_names: list = field(default_factory=list)
    cls_scaler: StandardScaler = None
    logit: LogisticRegression = None
    forest: RandomForestClassifier = None
    cls_test: dict = field(default_factory=dict)       # y_test, p_logit, p_forest
    cls_drivers: pd.DataFrame = None
    # clustering
    clu_scaler: StandardScaler = None
    kmeans: KMeans = None
    tier_of_cluster: dict = field(default_factory=dict)  # cluster id -> tier index (0 = High)
    pca: PCA = None
    silhouette: dict = field(default_factory=dict)
    inertia: dict = field(default_factory=dict)
    # association
    freq: dict = field(default_factory=dict)
    rules: pd.DataFrame = None
    lift: pd.DataFrame = None
    # regression
    reg_names: list = field(default_factory=list)
    reg_scaler: StandardScaler = None
    ridge: Ridge = None
    gbr: GradientBoostingRegressor = None
    reg_test: dict = field(default_factory=dict)
    # survey with segment labels added
    survey: pd.DataFrame = None


def train_all(survey: pd.DataFrame) -> Models:
    df = survey.copy()
    cats = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEG_REG}
    M = Models(cats=cats)

    # ---------- 1. classification
    M.cls_names = feature_names(cats, NUMERIC_AD, CATEG_AD)
    X = encode(df, M.cls_names)
    y = df["interested"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    M.cls_scaler = StandardScaler().fit(Xtr)
    M.cls_scaler.scale_[M.cls_scaler.scale_ == 0] = 1
    M.logit = LogisticRegression(C=0.5, max_iter=2000).fit(M.cls_scaler.transform(Xtr), ytr)
    M.forest = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    M.cls_test = {"y": yte, "p_logit": M.logit.predict_proba(M.cls_scaler.transform(Xte))[:, 1],
                  "p_forest": M.forest.predict_proba(Xte)[:, 1]}
    M.cls_drivers = (pd.DataFrame({"feature": [pretty(n) for n in M.cls_names],
                                   "importance": M.forest.feature_importances_,
                                   "logit_coef": M.logit.coef_[0]})
                     .sort_values("importance", ascending=False).reset_index(drop=True))

    # ---------- 2. clustering
    M.clu_scaler = StandardScaler().fit(df[CLU].astype(float))
    Z = M.clu_scaler.transform(df[CLU].astype(float))
    for k in range(2, 7):
        km_k = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
        M.silhouette[k] = float(silhouette_score(Z, km_k.labels_))
        M.inertia[k] = float(km_k.inertia_)
    M.kmeans = KMeans(n_clusters=3, n_init=50, random_state=SEED).fit(Z)
    df["cluster"] = M.kmeans.labels_
    order = df.groupby("cluster")["interested"].mean().sort_values(ascending=False).index.tolist()
    M.tier_of_cluster = {int(c): i for i, c in enumerate(order)}
    df["segment"] = [SEGS[M.tier_of_cluster[c]] for c in df["cluster"]]
    M.pca = PCA(n_components=2, random_state=SEED).fit(Z)
    P = M.pca.transform(Z)
    df["pc1"], df["pc2"] = P[:, 0], P[:, 1]

    # ---------- 3. association
    M.freq = apriori.frequent_itemsets(df[SERVICES], min_support=0.05)
    M.rules = apriori.association_rules(M.freq, len(df), min_confidence=0.3, min_lift=1.0)
    M.lift = apriori.lift_matrix(df[SERVICES])

    # ---------- 4. regression
    M.reg_names = feature_names(cats, NUMERIC_REG, CATEG_REG)
    XR = encode(df, M.reg_names)
    yr = df["wtp_monthly_aed"].values.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(XR, yr, test_size=0.25, random_state=SEED)
    M.reg_scaler = StandardScaler().fit(Xtr)
    M.reg_scaler.scale_[M.reg_scaler.scale_ == 0] = 1
    M.ridge = Ridge(alpha=5.0).fit(M.reg_scaler.transform(Xtr), ytr)
    M.gbr = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                      random_state=SEED).fit(Xtr, ytr)
    M.reg_test = {"y": yte, "p_ridge": np.clip(M.ridge.predict(M.reg_scaler.transform(Xte)), 0, None),
                  "p_gbr": np.clip(M.gbr.predict(Xte), 0, None)}
    M.survey = df
    return M


# ------------------------------------------------------------------ metrics helpers
def classification_metrics(y, p, threshold=0.5) -> dict:
    pred = (p >= threshold).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1 score": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p), "confusion": confusion_matrix(y, pred, labels=[0, 1])}


def roc_points(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return fpr, tpr


def regression_metrics(y, p) -> dict:
    return {"R²": r2_score(y, p), "MAE (AED)": mean_absolute_error(y, p),
            "RMSE (AED)": float(np.sqrt(mean_squared_error(y, p)))}


def cluster_any_k(M: Models, k: int):
    """Re-cluster with a chosen k; clusters ranked by subscribe rate (Tier 1 = most probable)."""
    df = M.survey
    Z = M.clu_scaler.transform(df[CLU].astype(float))
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z)
    lab = pd.Series(km.labels_, index=df.index)
    order = df.groupby(lab)["interested"].mean().sort_values(ascending=False).index.tolist()
    rank = {c: i for i, c in enumerate(order)}
    names = SEGS if k == 3 else [f"Tier {i + 1}" for i in range(k)]
    name_of = {c: names[rank[c]] for c in range(k)}
    return lab.map(name_of), names, km, name_of


def segment_profiles(df: pd.DataFrame, seg_col="segment", order=SEGS) -> pd.DataFrame:
    rows = []
    for s in order:
        g = df[df[seg_col] == s]
        if not len(g):
            continue
        rows.append({
            "Segment": s, "Respondents": len(g), "Share": len(g) / len(df),
            "Would subscribe (Q29 4–5)": g.interested.mean(), "Avg spend, AED/mo": g.wtp_monthly_aed.mean(),
            "Patient aged 60+": (g.patient_age_group >= 4).mean(), "Chronic conditions": g.n_conditions.mean(),
            "Missed doses (0–3)": g.missed_doses_week.mean(), "Tech comfort (1–5)": g.tech_comfort.mean(),
            "Caregiver respondents": (g.respondent_role != "Myself").mean(),
            "Services wanted (of 10)": g.n_services_selected.mean(),
            "Top sign-up channel": g.purchase_channel.mode()[0], "Top plan": g.preferred_plan.mode()[0]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scoring new respondents
def _prep(rows) -> pd.DataFrame:
    df = pd.DataFrame([rows]) if isinstance(rows, dict) else rows.copy()
    df["n_services_selected"] = df[SERVICES].astype(int).sum(axis=1)
    return df


def predict_subscribe(M: Models, rows) -> np.ndarray:
    df = _prep(rows)
    return M.logit.predict_proba(M.cls_scaler.transform(encode(df, M.cls_names)))[:, 1]


def predict_segment(M: Models, rows) -> np.ndarray:
    """Tier index: 0 High, 1 Medium, 2 Low."""
    df = _prep(rows)
    labels = M.kmeans.predict(M.clu_scaler.transform(df[CLU].astype(float)))
    return np.array([M.tier_of_cluster[int(c)] for c in labels])


def pca_coords(M: Models, rows) -> np.ndarray:
    df = _prep(rows)
    return M.pca.transform(M.clu_scaler.transform(df[CLU].astype(float)))


def predict_wtp(M: Models, rows) -> np.ndarray:
    df = _prep(rows)
    return np.clip(M.gbr.predict(encode(df, M.reg_names)), 0, None)


def next_offer(M: Models, row: dict, min_conf=0.6, min_lift=1.15):
    """Best service to offer next: highest-confidence rule whose antecedent the person already wants."""
    have = {s for s in SERVICES if int(row.get(s, 0)) == 1}
    r = M.rules
    r = r[(r.confidence >= min_conf) & (r.lift >= min_lift) & (r.n_antecedent <= 2)]
    best = None
    for _, x in r.iterrows():
        if set(x.antecedent) <= have and x.consequent[0] not in have:
            if best is None or (x.n_antecedent, x.confidence) > (best.n_antecedent, best.confidence):
                best = x
    if best is None:
        return None
    return {"service": SVC_LABEL[best.consequent[0]], "because": [SVC_LABEL[a] for a in best.antecedent],
            "confidence": float(best.confidence), "lift": float(best.lift)}
