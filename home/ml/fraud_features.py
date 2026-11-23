"""
Provider-level feature engineering for Medicare provider fraud detection.

This is a faithful, vectorised port of the pipeline in
``notebooks/fraud_detection.ipynb`` / ``mc-termpaper.ipynb``:

  1. Clean the beneficiary table (chronic conditions / gender 2 -> 0,
     renal 'Y' -> 1, WhetherDead flag from DOD).
  2. Stack inpatient + outpatient claims and join beneficiary details.
  3. Per claim: NumPhysicians, NumProc, NumUniqueClaims, ExtraClm,
     AdmissionDays, ClaimDays, Age, Hospt.
  4. Aggregate per Provider: unique beneficiaries / claims, sums and means.

The result has one row per provider and the 31 model features in
``FEATURE_COLUMNS`` (same order as the notebook).

The module only depends on pandas / numpy so it can be tested outside Django.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHRONIC_COLUMNS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]
PHYSICIAN_COLUMNS = ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]
PROCEDURE_COLUMNS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
DIAGNOSIS_COLUMNS = ["ClmAdmitDiagnosisCode"] + [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]

SUM_COLUMNS = [
    "InscClaimAmtReimbursed", "DeductibleAmtPaid", "RenalDiseaseIndicator",
    *CHRONIC_COLUMNS,
    "WhetherDead", "NumPhysicians", "NumProc", "NumUniqueClaims", "ExtraClm",
    "AdmissionDays", "ClaimDays", "Hospt",
]
MEAN_COLUMNS = [
    "NoOfMonths_PartACov", "NoOfMonths_PartBCov",
    "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt", "Age",
]
COUNT_COLUMNS = ["BeneID", "ClaimID"]
FEATURE_COLUMNS = COUNT_COLUMNS + SUM_COLUMNS + MEAN_COLUMNS  # 31 features

REQUIRED_BENEFICIARY = [
    "BeneID", "DOB", "DOD", "Gender", "RenalDiseaseIndicator",
    "NoOfMonths_PartACov", "NoOfMonths_PartBCov", *CHRONIC_COLUMNS,
    "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
]
REQUIRED_CLAIMS = [
    "BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider",
    "InscClaimAmtReimbursed", "DeductibleAmtPaid",
]

FEATURE_LABELS = {
    "BeneID": "Unique beneficiaries",
    "ClaimID": "Unique claims",
    "InscClaimAmtReimbursed": "Total amount reimbursed ($)",
    "DeductibleAmtPaid": "Total deductible paid ($)",
    "RenalDiseaseIndicator": "Claims for renal-disease patients",
    "ChronicCond_Alzheimer": "Claims: Alzheimer's",
    "ChronicCond_Heartfailure": "Claims: heart failure",
    "ChronicCond_KidneyDisease": "Claims: kidney disease",
    "ChronicCond_Cancer": "Claims: cancer",
    "ChronicCond_ObstrPulmonary": "Claims: obstructive pulmonary",
    "ChronicCond_Depression": "Claims: depression",
    "ChronicCond_Diabetes": "Claims: diabetes",
    "ChronicCond_IschemicHeart": "Claims: ischemic heart",
    "ChronicCond_Osteoporasis": "Claims: osteoporosis",
    "ChronicCond_rheumatoidarthritis": "Claims: rheumatoid arthritis",
    "ChronicCond_stroke": "Claims: stroke",
    "WhetherDead": "Claims for deceased beneficiaries",
    "NumPhysicians": "Physician involvements",
    "NumProc": "Procedures billed",
    "NumUniqueClaims": "Unique diagnosis codes billed",
    "ExtraClm": "Duplicated diagnosis codes",
    "AdmissionDays": "Total admission days",
    "ClaimDays": "Total claim days",
    "Hospt": "Inpatient (hospitalised) claims",
    "NoOfMonths_PartACov": "Avg Part A coverage (months)",
    "NoOfMonths_PartBCov": "Avg Part B coverage (months)",
    "IPAnnualReimbursementAmt": "Avg patient IP annual reimbursement ($)",
    "IPAnnualDeductibleAmt": "Avg patient IP annual deductible ($)",
    "OPAnnualReimbursementAmt": "Avg patient OP annual reimbursement ($)",
    "OPAnnualDeductibleAmt": "Avg patient OP annual deductible ($)",
    "Age": "Avg patient age",
}


class FeatureError(ValueError):
    """Raised when uploaded files do not match the expected schema."""


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise FeatureError(f"{name} file is missing columns: {', '.join(missing)}")


def _row_unique_count(frame: pd.DataFrame) -> np.ndarray:
    """Number of distinct non-null values per row (vectorised `uniq()` from the notebook)."""
    n = len(frame)
    if n == 0 or frame.shape[1] == 0:
        return np.zeros(n, dtype=int)
    values = frame.to_numpy(dtype=object)
    rows, cols = np.nonzero(pd.notna(values))
    if len(rows) == 0:
        return np.zeros(n, dtype=int)
    long = pd.DataFrame({"row": rows, "val": values[rows, cols].astype(str)})
    counts = long.drop_duplicates().groupby("row").size()
    out = np.zeros(n, dtype=int)
    out[counts.index.to_numpy()] = counts.to_numpy()
    return out


def prepare_beneficiaries(bene: pd.DataFrame) -> pd.DataFrame:
    _check_columns(bene, REQUIRED_BENEFICIARY, "Beneficiary")
    b = bene.copy()
    for col in CHRONIC_COLUMNS + ["Gender"]:
        b[col] = pd.to_numeric(b[col], errors="coerce").replace(2, 0).fillna(0).astype(int)
    renal = b["RenalDiseaseIndicator"].astype(str).str.strip().str.upper()
    b["RenalDiseaseIndicator"] = np.where(renal.isin(["Y", "1"]), 1, 0)
    b["DOB"] = pd.to_datetime(b["DOB"], errors="coerce")
    b["DOD"] = pd.to_datetime(b["DOD"], errors="coerce")
    b["WhetherDead"] = b["DOD"].notna().astype(int)
    return b


def build_claim_level(bene: pd.DataFrame, inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> pd.DataFrame:
    _check_columns(inpatient, REQUIRED_CLAIMS, "Inpatient")
    _check_columns(outpatient, REQUIRED_CLAIMS, "Outpatient")
    b = prepare_beneficiaries(bene)

    claims = pd.concat([inpatient, outpatient], ignore_index=True, sort=False)
    for col in PHYSICIAN_COLUMNS + PROCEDURE_COLUMNS + DIAGNOSIS_COLUMNS + [
        "AdmissionDt", "DischargeDt", "DiagnosisGroupCode"
    ]:
        if col not in claims.columns:
            claims[col] = np.nan

    data = claims.merge(b, on="BeneID", how="inner")
    if data.empty:
        raise FeatureError("No claims matched a beneficiary (check that BeneID values line up across files).")

    data["NumPhysicians"] = _row_unique_count(data[PHYSICIAN_COLUMNS])
    data["NumProc"] = data[PROCEDURE_COLUMNS].notna().sum(axis=1)
    num_claims = data[DIAGNOSIS_COLUMNS].notna().sum(axis=1)
    data["NumUniqueClaims"] = _row_unique_count(data[DIAGNOSIS_COLUMNS])
    data["ExtraClm"] = num_claims - data["NumUniqueClaims"]

    for col in ["AdmissionDt", "DischargeDt", "ClaimStartDt", "ClaimEndDt"]:
        data[col] = pd.to_datetime(data[col], errors="coerce")
    data["AdmissionDays"] = (data["DischargeDt"] - data["AdmissionDt"]).dt.days + 1
    data["ClaimDays"] = (data["ClaimEndDt"] - data["ClaimStartDt"]).dt.days + 1
    data["Age"] = (((data["ClaimStartDt"] - data["DOB"]).dt.days + 1) / 365.25).round()
    data["Hospt"] = data["DiagnosisGroupCode"].notna().astype(int)

    for col in ["InscClaimAmtReimbursed", "DeductibleAmtPaid", *MEAN_COLUMNS[:-1]]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    num_cols = SUM_COLUMNS + MEAN_COLUMNS
    data[num_cols] = data[num_cols].fillna(0)
    return data


def build_provider_features(bene: pd.DataFrame, inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> pd.DataFrame:
    """Return one row per Provider with ``Provider`` + FEATURE_COLUMNS."""
    data = build_claim_level(bene, inpatient, outpatient)
    sums = data.groupby("Provider")[SUM_COLUMNS].sum()
    counts = data.groupby("Provider")[COUNT_COLUMNS].nunique()
    means = data.groupby("Provider")[MEAN_COLUMNS].mean()
    df = counts.join(sums).join(means).reset_index()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    return df[["Provider", *FEATURE_COLUMNS]]


def attach_labels(features: pd.DataFrame, labels: pd.DataFrame | None) -> pd.DataFrame:
    """Add a ``target`` column (1/0, or NaN when unknown) from a Provider/PotentialFraud file."""
    out = features.copy()
    if labels is None or labels.empty:
        out["target"] = np.nan
        return out
    _check_columns(labels, ["Provider", "PotentialFraud"], "Labels")
    lab = labels[["Provider", "PotentialFraud"]].drop_duplicates("Provider").copy()
    lab["target"] = (lab["PotentialFraud"].astype(str).str.strip().str.lower() == "yes").astype(int)
    out = out.merge(lab[["Provider", "target"]], on="Provider", how="left")
    return out
