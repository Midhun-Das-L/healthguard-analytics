"""
Training and scoring for both models used by the dashboard.

* Provider fraud model  - trained on provider-level features (fraud_features.py).
  Logistic Regression, Random Forest and SVC (the notebook's candidates) are
  compared on a stratified hold-out split and the best ROC-AUC wins.
* Readmission model     - logistic regression on the prepared diabetes encounter
  data (data/readmission/model_data.csv) with the prototype's feature set.

Fixes relative to the notebook / prototype:
  - the scaler is fitted on training data only (inside a Pipeline); the notebook
    re-fitted StandardScaler on the validation set;
  - labels are joined to features by Provider id instead of relying on row order;
  - class imbalance is handled with class_weight="balanced";
  - no hard-coded absolute CSV paths.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .fraud_features import FEATURE_COLUMNS, FEATURE_LABELS

FRAUD_MODEL_FILE = "fraud_model.joblib"
READMISSION_MODEL_FILE = "readmission_model.joblib"

READMISSION_FEATURES = [
    'age', 'time_in_hospital', 'num_procedures', 'num_medications', 'number_outpatient_log1p',
    'number_emergency_log1p', 'number_inpatient_log1p', 'number_diagnoses', 'metformin',
    'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
    'pioglitazone', 'rosiglitazone', 'acarbose', 'tolazamide', 'insulin', 'glyburide-metformin',
    'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other', 'gender_1',
    'admission_type_id_3', 'admission_type_id_5', 'discharge_disposition_id_2', 'discharge_disposition_id_7',
    'discharge_disposition_id_10', 'discharge_disposition_id_18', 'admission_source_id_4',
    'admission_source_id_7', 'admission_source_id_9', 'max_glu_serum_0', 'max_glu_serum_1', 'A1Cresult_0',
    'A1Cresult_1', 'num_medications|time_in_hospital', 'num_medications|num_procedures',
    'time_in_hospital|num_lab_procedures', 'num_medications|num_lab_procedures', 'num_medications|number_diagnoses',
    'age|number_diagnoses', 'change|num_medications', 'number_diagnoses|time_in_hospital',
    'num_medications|numchange', 'level1_diag1_1.0', 'level1_diag1_2.0', 'level1_diag1_3.0', 'level1_diag1_4.0',
    'level1_diag1_5.0', 'level1_diag1_6.0', 'level1_diag1_7.0', 'level1_diag1_8.0',
]

READMISSION_LABELS = {
    'age': 'Age (bucket mid-point)', 'time_in_hospital': 'Time in hospital (std.)',
    'num_procedures': 'Procedures (std.)', 'num_medications': 'Medications (std.)',
    'number_outpatient_log1p': 'Prior outpatient visits', 'number_emergency_log1p': 'Prior emergency visits',
    'number_inpatient_log1p': 'Prior inpatient visits', 'number_diagnoses': 'Number of diagnoses (std.)',
    'insulin': 'On insulin', 'metformin': 'On metformin', 'gender_1': 'Gender: male',
    'admission_type_id_3': 'Admission: elective', 'admission_type_id_5': 'Admission: not available',
    'discharge_disposition_id_2': 'Discharged to short-term hospital',
    'discharge_disposition_id_7': 'Left against medical advice',
    'discharge_disposition_id_10': 'Discharged to neonatal unit',
    'discharge_disposition_id_18': 'Discharge: not available',
    'admission_source_id_4': 'Source: transfer from hospital', 'admission_source_id_7': 'Source: emergency room',
    'admission_source_id_9': 'Source: not available',
    'max_glu_serum_0': 'Max glucose serum normal', 'max_glu_serum_1': 'Max glucose serum high',
    'A1Cresult_0': 'A1C normal', 'A1Cresult_1': 'A1C high',
    'num_medications|time_in_hospital': 'Medications x stay length',
    'num_medications|num_procedures': 'Medications x procedures',
    'time_in_hospital|num_lab_procedures': 'Stay length x lab procedures',
    'num_medications|num_lab_procedures': 'Medications x lab procedures',
    'num_medications|number_diagnoses': 'Medications x diagnoses',
    'age|number_diagnoses': 'Age x diagnoses', 'change|num_medications': 'Medication change x medications',
    'number_diagnoses|time_in_hospital': 'Diagnoses x stay length',
    'num_medications|numchange': 'Medications x number of changes',
    'level1_diag1_1.0': 'Primary diagnosis: circulatory', 'level1_diag1_2.0': 'Primary diagnosis: respiratory',
    'level1_diag1_3.0': 'Primary diagnosis: digestive', 'level1_diag1_4.0': 'Primary diagnosis: diabetes',
    'level1_diag1_5.0': 'Primary diagnosis: injury', 'level1_diag1_6.0': 'Primary diagnosis: musculoskeletal',
    'level1_diag1_7.0': 'Primary diagnosis: genitourinary', 'level1_diag1_8.0': 'Primary diagnosis: neoplasms',
}


FRAUD_CUTOFFS = (0.7, 0.4)        # (high, medium) on fraud probability
READMISSION_CUTOFFS = (0.65, 0.5)  # balanced-weight LR scores; High ~3x readmit rate of Low


def risk_level(prob: float, cutoffs=FRAUD_CUTOFFS) -> str:
    high, medium = cutoffs
    if prob >= high:
        return "High"
    if prob >= medium:
        return "Medium"
    return "Low"


def _metrics(y_true, prob, threshold=0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, prob)
    idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 40)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr[idx]],
                      "tpr": [round(float(x), 4) for x in tpr[idx]]},
        "n_test": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


# --------------------------------------------------------------------------- fraud
def train_fraud_model(features: pd.DataFrame, out_dir: Path, source: str = "sample", random_state: int = 1) -> dict:
    """``features`` must contain FEATURE_COLUMNS and a binary ``target``."""
    df = features.dropna(subset=["target"])
    X = df[FEATURE_COLUMNS].astype(float)
    y = df["target"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("Training data needs both fraud and non-fraud providers.")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                random_state=random_state, n_jobs=-1),
        "SVC (RBF)": SVC(probability=True, class_weight="balanced", random_state=random_state),
    }
    comparison, best = [], None
    for name, clf in candidates.items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        m = _metrics(y_te, prob)
        comparison.append({"model": name, **{k: m[k] for k in ("roc_auc", "accuracy", "precision", "recall", "f1")}})
        if best is None or m["roc_auc"] > best[2]["roc_auc"]:
            best = (name, pipe, m)
    name, _, metrics = best

    # refit the winner on all labelled data for production scoring
    final = Pipeline([("scale", StandardScaler()), ("clf", candidates[name].__class__(**candidates[name].get_params()))])
    final.fit(X, y)
    winner_holdout = Pipeline([("scale", StandardScaler()),
                               ("clf", candidates[name].__class__(**candidates[name].get_params()))]).fit(X_tr, y_tr)
    imp = permutation_importance(winner_holdout, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                 random_state=random_state, n_jobs=-1)
    importances = sorted(
        ({"feature": c, "label": FEATURE_LABELS.get(c, c), "importance": round(float(v), 5)}
         for c, v in zip(FEATURE_COLUMNS, imp.importances_mean)),
        key=lambda d: d["importance"], reverse=True)

    bundle = {
        "pipeline": final,
        "features": FEATURE_COLUMNS,
        "model_name": name,
        "metrics": metrics,
        "comparison": comparison,
        "importances": importances,
        "baseline_honest": X[y == 0].median().to_dict(),
        "baseline_fraud": X[y == 1].median().to_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_source": source,
        "n_train_providers": int(len(df)),
        "fraud_rate": round(float(y.mean()), 4),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_dir / FRAUD_MODEL_FILE)
    return bundle


def score_providers(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["fraud_probability"] = bundle["pipeline"].predict_proba(out[bundle["features"]].astype(float))[:, 1]
    out["risk_level"] = out["fraud_probability"].map(risk_level)
    return out.sort_values("fraud_probability", ascending=False).reset_index(drop=True)


def explain_provider(bundle: dict, row: dict, top_n: int = 8) -> list[dict]:
    """Compare a provider's top-importance features with the typical honest provider."""
    out = []
    for item in bundle["importances"][:top_n]:
        f = item["feature"]
        val = float(row.get(f, 0) or 0)
        honest = float(bundle["baseline_honest"].get(f, 0) or 0)
        fraud = float(bundle["baseline_fraud"].get(f, 0) or 0)
        ratio = (val / honest) if honest else None
        out.append({"feature": f, "label": item["label"], "value": round(val, 2),
                    "honest_median": round(honest, 2), "fraud_median": round(fraud, 2),
                    "ratio": round(ratio, 2) if ratio is not None else None,
                    "importance": item["importance"]})
    return out


# --------------------------------------------------------------------- readmission
def train_readmission_model(csv_path: Path, out_dir: Path, random_state: int = 0) -> dict:
    df = pd.read_csv(csv_path)
    X = df[READMISSION_FEATURES].astype(float)
    y = df["readmitted"].astype(int).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
    make = lambda: Pipeline([("scale", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1))])
    holdout = make().fit(X_tr, y_tr)
    metrics = _metrics(y_te, holdout.predict_proba(X_te)[:, 1])
    final = make().fit(X, y)
    coefs = final.named_steps["clf"].coef_[0]
    drivers = sorted(({"feature": f, "label": READMISSION_LABELS.get(f, f), "coef": round(float(c), 4)}
                      for f, c in zip(READMISSION_FEATURES, coefs)), key=lambda d: abs(d["coef"]), reverse=True)
    bundle = {
        "pipeline": final, "features": READMISSION_FEATURES, "model_name": "Logistic Regression",
        "metrics": metrics, "drivers": drivers,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(df)), "readmit_rate": round(float(y.mean()), 4),
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, Path(out_dir) / READMISSION_MODEL_FILE)
    return bundle


def score_encounters(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"File is missing {len(missing)} model columns, e.g. {', '.join(missing[:5])}")
    out = df.copy()
    out["readmission_probability"] = bundle["pipeline"].predict_proba(out[bundle["features"]].astype(float))[:, 1]
    out["risk_level"] = out["readmission_probability"].map(lambda p: risk_level(p, READMISSION_CUTOFFS))
    return out


def explain_encounter(bundle: dict, row: dict, top_n: int = 8) -> list[dict]:
    pipe = bundle["pipeline"]
    scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    x = np.array([[float(row.get(f, 0) or 0) for f in bundle["features"]]])
    z = scaler.transform(pd.DataFrame(x, columns=bundle["features"]))[0]
    contrib = z * clf.coef_[0]
    order = np.argsort(-np.abs(contrib))[:top_n]
    return [{"feature": bundle["features"][i], "label": READMISSION_LABELS.get(bundle["features"][i], bundle["features"][i]),
             "value": round(float(x[0, i]), 3), "contribution": round(float(contrib[i]), 3)} for i in order]


def load_bundle(path: Path):
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def public_info(bundle: dict | None) -> dict:
    """JSON-safe subset of a bundle (drops the fitted pipeline and baselines)."""
    if not bundle:
        return {}
    skip = {"pipeline", "baseline_honest", "baseline_fraud"}
    return json.loads(json.dumps({k: v for k, v in bundle.items() if k not in skip}, default=str))
