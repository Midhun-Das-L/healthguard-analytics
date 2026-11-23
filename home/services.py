"""Glue between the ML engine (home/ml) and the Django models / views."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import transaction
from django.db.models import Avg, Count, IntegerField, Q, Sum
from django.db.models.functions import Cast

from .ml import engine, fraud_features
from .models import EncounterScore, ProviderScore, ScoringBatch

_lock = threading.Lock()
READMIT_INT = Cast("actual_readmitted", IntegerField())  # AVG(boolean) is not portable (PostgreSQL)
_cache: dict[str, tuple[float, dict]] = {}


class ModelNotTrained(RuntimeError):
    pass


def _load(filename: str) -> dict | None:
    path = Path(settings.ML_MODELS_DIR) / filename
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    with _lock:
        cached = _cache.get(filename)
        if cached and cached[0] == mtime:
            return cached[1]
        bundle = engine.load_bundle(path)
        _cache[filename] = (mtime, bundle)
        return bundle


def fraud_bundle(required=True):
    b = _load(engine.FRAUD_MODEL_FILE)
    if b is None and required:
        raise ModelNotTrained("Fraud model not trained yet. Run: python manage.py setup_demo")
    return b


def readmission_bundle(required=True):
    b = _load(engine.READMISSION_MODEL_FILE)
    if b is None and required:
        raise ModelNotTrained("Readmission model not trained yet. Run: python manage.py setup_demo")
    return b


def read_csv(fileobj) -> pd.DataFrame:
    try:
        return pd.read_csv(fileobj, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        raise fraud_features.FeatureError(f"Could not read CSV '{getattr(fileobj, 'name', '')}': {exc}") from exc


# ------------------------------------------------------------------ fraud batches
@transaction.atomic
def create_batch(name, bene, inpatient, outpatient, labels=None, source=ScoringBatch.SOURCE_UPLOAD, user=None):
    bundle = fraud_bundle()
    feats = fraud_features.build_provider_features(bene, inpatient, outpatient)
    feats = fraud_features.attach_labels(feats, labels)
    scored = engine.score_providers(bundle, feats)

    batch = ScoringBatch.objects.create(
        name=name, source=source, created_by=user if (user and user.is_authenticated) else None,
        model_name=bundle["model_name"], n_providers=len(scored),
        n_claims=int(len(inpatient) + len(outpatient)), n_beneficiaries=int(bene["BeneID"].nunique()),
        total_reimbursed=float(scored["InscClaimAmtReimbursed"].sum()),
        has_labels=bool(scored["target"].notna().any()),
    )
    rows = []
    for rec in scored.to_dict("records"):
        target = rec.get("target")
        rows.append(ProviderScore(
            batch=batch, provider_id=str(rec["Provider"]),
            fraud_probability=float(rec["fraud_probability"]), risk_level=rec["risk_level"],
            actual_fraud=None if target is None or pd.isna(target) else bool(target),
            n_beneficiaries=int(rec["BeneID"]), n_claims=int(rec["ClaimID"]),
            inpatient_claims=int(rec["Hospt"]), total_reimbursed=float(rec["InscClaimAmtReimbursed"]),
            avg_patient_age=float(rec["Age"]),
            features={c: round(float(rec[c]), 4) for c in fraud_features.FEATURE_COLUMNS},
        ))
    ProviderScore.objects.bulk_create(rows, batch_size=1000)
    return batch


def create_sample_batch(user=None):
    from .ml.sample_data import find_kaggle_files

    files = find_kaggle_files(settings.FRAUD_SAMPLE_DIR)
    frames = {k: pd.read_csv(v, low_memory=False) for k, v in files.items()}
    return create_batch("Sample Medicare claims (training set)", frames["beneficiary"], frames["inpatient"],
                        frames["outpatient"], frames.get("labels"), source=ScoringBatch.SOURCE_SAMPLE, user=user)


def latest_batch():
    return ScoringBatch.objects.order_by("-created_at").first()


def _histogram(values, bins=10):
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bins, range=(0, 1))
    labels = [f"{int(i * 100 / bins)}-{int((i + 1) * 100 / bins)}%" for i in range(bins)]
    return {"labels": labels, "counts": [int(c) for c in counts]}


def fraud_dashboard(batch: ScoringBatch, threshold: float) -> dict:
    qs = ProviderScore.objects.filter(batch=batch)
    flagged = qs.filter(fraud_probability__gte=threshold)
    agg = qs.aggregate(total=Sum("total_reimbursed"), claims=Sum("n_claims"))
    fagg = flagged.aggregate(total=Sum("total_reimbursed"), claims=Sum("n_claims"), n=Count("id"))
    total = agg["total"] or 0
    probs = list(qs.values_list("fraud_probability", flat=True))

    by_level = {lvl: qs.filter(risk_level=lvl).aggregate(n=Count("id"), amt=Sum("total_reimbursed"))
                for lvl in ("High", "Medium", "Low")}
    status_counts = dict(flagged.order_by().values_list("review_status").annotate(n=Count("id")))
    top = list(qs.order_by("-fraud_probability")[:10].values(
        "id", "provider_id", "fraud_probability", "risk_level", "n_claims", "total_reimbursed", "review_status"))

    # bubble/scatter: claims vs reimbursed coloured by flag (cap points for the browser)
    sample = qs.filter(n_claims__gt=0, total_reimbursed__gt=0).order_by("-fraud_probability")[:800].values_list(
        "id", "provider_id", "n_claims", "total_reimbursed", "fraud_probability")
    scatter = {"flagged": [], "normal": []}
    for pk, prov, n, amt, p in sample:
        scatter["flagged" if p >= threshold else "normal"].append(
            {"x": n, "y": round(amt, 2), "id": pk, "provider": prov, "p": round(p, 4)})

    evaluation = None
    if batch.has_labels:
        lab = qs.exclude(actual_fraud=None)
        tp = lab.filter(fraud_probability__gte=threshold, actual_fraud=True).count()
        fp = lab.filter(fraud_probability__gte=threshold, actual_fraud=False).count()
        fn = lab.filter(fraud_probability__lt=threshold, actual_fraud=True).count()
        tn = lab.filter(fraud_probability__lt=threshold, actual_fraud=False).count()
        n = tp + fp + fn + tn
        evaluation = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round((tp + tn) / n, 4) if n else None,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }

    return {
        "batch": {"id": batch.id, "name": batch.name, "created_at": batch.created_at.isoformat(),
                  "model_name": batch.model_name, "has_labels": batch.has_labels},
        "threshold": threshold,
        "kpis": {
            "providers": batch.n_providers, "claims": agg["claims"] or 0, "beneficiaries": batch.n_beneficiaries,
            "flagged": fagg["n"] or 0,
            "flagged_pct": round(100 * (fagg["n"] or 0) / batch.n_providers, 1) if batch.n_providers else 0,
            "total_reimbursed": round(total, 2),
            "at_risk_amount": round(fagg["total"] or 0, 2),
            "at_risk_pct": round(100 * (fagg["total"] or 0) / total, 1) if total else 0,
            "avg_probability": round(float(np.mean(probs)), 4) if probs else 0,
            "under_review": status_counts.get("investigating", 0),
            "confirmed": qs.filter(review_status="confirmed").count(),
            "unreviewed_flagged": status_counts.get("new", 0),
        },
        "histogram": _histogram(probs),
        "risk_levels": {k: {"n": v["n"] or 0, "amount": round(v["amt"] or 0, 2)} for k, v in by_level.items()},
        "review_status": {label: qs.filter(review_status=key).count() for key, label in ProviderScore.STATUS_CHOICES},
        "top_providers": top,
        "scatter": scatter,
        "evaluation": evaluation,
    }


# ---------------------------------------------------------------- readmission
RACE_COLUMNS = ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]


def _race(rec):
    for c in RACE_COLUMNS:
        if rec.get(c) == 1:
            return "African American" if c == "AfricanAmerican" else c
    return "Unknown"


@transaction.atomic
def seed_encounters(csv_path=None) -> int:
    bundle = readmission_bundle()
    df = pd.read_csv(csv_path or settings.READMISSION_DATA)
    scored = engine.score_encounters(bundle, df).drop_duplicates("encounter_id")
    EncounterScore.objects.all().delete()
    rows = []
    for rec in scored.to_dict("records"):
        rows.append(EncounterScore(
            encounter_id=int(rec["encounter_id"]), patient_nbr=int(rec["patient_nbr"]), age=int(rec["age"]),
            gender="Male" if rec.get("gender_1") == 1 else "Female", race=_race(rec),
            on_insulin=bool(rec.get("insulin")), medication_changed=bool(rec.get("change")),
            readmission_probability=float(rec["readmission_probability"]), risk_level=rec["risk_level"],
            actual_readmitted=bool(rec["readmitted"]) if "readmitted" in rec and not pd.isna(rec["readmitted"]) else None,
            features={c: float(rec[c]) for c in bundle["features"]},
        ))
    EncounterScore.objects.bulk_create(rows, batch_size=1000)
    return len(rows)


def readmission_dashboard(qs=None) -> dict:
    qs = qs if qs is not None else EncounterScore.objects.all()
    total = qs.count()
    probs = list(qs.values_list("readmission_probability", flat=True))
    by_age = list(qs.values("age").annotate(n=Count("id"), avg=Avg("readmission_probability"),
                                            actual=Avg(READMIT_INT)).order_by("age"))
    by_level = {lvl: qs.filter(risk_level=lvl).aggregate(n=Count("id"), actual=Avg(READMIT_INT))
                for lvl in ("High", "Medium", "Low")}
    return {
        "kpis": {
            "encounters": total,
            "patients": qs.values("patient_nbr").distinct().count(),
            "high_risk": by_level["High"]["n"] or 0,
            "high_risk_pct": round(100 * (by_level["High"]["n"] or 0) / total, 1) if total else 0,
            "actual_rate": round(100 * (qs.aggregate(r=Avg(READMIT_INT))["r"] or 0), 1),
            "follow_ups": qs.filter(follow_up_scheduled=True).count(),
            "high_risk_no_follow_up": qs.filter(risk_level="High", follow_up_scheduled=False).count(),
        },
        "histogram": _histogram(probs),
        "by_age": {"labels": [f"{r['age'] - 5}-{r['age'] + 5}" for r in by_age],
                   "predicted": [round(100 * (r["avg"] or 0), 1) for r in by_age],
                   "actual": [round(100 * (r["actual"] or 0), 1) for r in by_age],
                   "n": [r["n"] for r in by_age]},
        "risk_levels": {k: {"n": v["n"] or 0, "actual_rate": round(100 * (v["actual"] or 0), 1)}
                        for k, v in by_level.items()},
        "insulin": {
            "on": round(100 * (qs.filter(on_insulin=True).aggregate(a=Avg("readmission_probability"))["a"] or 0), 1),
            "off": round(100 * (qs.filter(on_insulin=False).aggregate(a=Avg("readmission_probability"))["a"] or 0), 1),
        },
    }


def filter_encounters(params):
    qs = EncounterScore.objects.all()
    q = (params.get("q") or "").strip()
    if q.isdigit():
        qs = qs.filter(Q(encounter_id=int(q)) | Q(patient_nbr=int(q)))
    if params.get("risk") in ("High", "Medium", "Low"):
        qs = qs.filter(risk_level=params["risk"])
    if params.get("gender") in ("Male", "Female"):
        qs = qs.filter(gender=params["gender"])
    if params.get("follow_up") == "no":
        qs = qs.filter(follow_up_scheduled=False)
    elif params.get("follow_up") == "yes":
        qs = qs.filter(follow_up_scheduled=True)
    return qs


def filter_providers(batch, params):
    qs = ProviderScore.objects.filter(batch=batch)
    q = (params.get("q") or "").strip()
    if q:
        qs = qs.filter(provider_id__icontains=q)
    if params.get("risk") in ("High", "Medium", "Low"):
        qs = qs.filter(risk_level=params["risk"])
    if params.get("status") in dict(ProviderScore.STATUS_CHOICES):
        qs = qs.filter(review_status=params["status"])
    try:
        min_p = float(params.get("min_prob") or 0)
        if min_p:
            qs = qs.filter(fraud_probability__gte=min_p / 100 if min_p > 1 else min_p)
    except ValueError:
        pass
    sort = params.get("sort") or "-fraud_probability"
    allowed = {"fraud_probability", "-fraud_probability", "total_reimbursed", "-total_reimbursed",
               "n_claims", "-n_claims", "provider_id", "-provider_id"}
    return qs.order_by(sort if sort in allowed else "-fraud_probability")
