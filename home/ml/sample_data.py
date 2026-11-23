"""
Synthetic Medicare claims generator.

The notebook was built on the Kaggle "Healthcare Provider Fraud Detection
Analysis" files (Train_Beneficiarydata-*.csv, Train_Inpatientdata-*.csv,
Train_Outpatientdata-*.csv, Train-*.csv). Those raw files are not part of the
repository, so this module produces files with exactly the same columns and
plausible distributions so the app runs out of the box.

Fraudulent providers are generated with the behaviours the notebook's
features are designed to catch: more claims and patients, more inpatient
stays, higher reimbursement, more physicians and procedures per claim and
padded (duplicated) diagnosis codes. Signals overlap with honest providers so
the task is realistic rather than trivial.

Replace these files with the real Kaggle data and run
``python manage.py train_models --fraud-dir <folder>`` to retrain on it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fraud_features import CHRONIC_COLUMNS

KAGGLE_SUFFIX = "1542865627584"

_DIAG_POOL = [
    "4019", "25000", "2724", "V5869", "4011", "42731", "V5861", "2720", "2449", "4280",
    "53081", "41401", "496", "2859", "41400", "78079", "5990", "28521", "3051", "2809",
    "311", "71590", "V5883", "29620", "5849", "49121", "V4581", "2768", "51881", "486",
    "78650", "V5866", "7242", "73300", "7231", "V1582", "6186", "2948", "42732", "59970",
]
_PROC_POOL = ["4019", "9904", "8154", "66", "3995", "4516", "3893", "5123", "8151", "9671"]
_DRG_POOL = [f"{n:03d}" for n in range(1, 999, 7)]


def _dates(rng, start, end, n):
    start_ts = pd.Timestamp(start).value // 10**9
    end_ts = pd.Timestamp(end).value // 10**9
    return pd.to_datetime(rng.integers(start_ts, end_ts, n), unit="s").normalize()


def _beneficiaries(rng, n, start_id):
    ids = [f"BENE{start_id + i}" for i in range(n)]
    dob = _dates(rng, "1915-01-01", "1983-12-31", n)
    dead = rng.random(n) < 0.012
    dod = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns]")
    dod[dead] = _dates(rng, "2009-07-01", "2009-12-01", int(dead.sum()))
    df = pd.DataFrame({
        "BeneID": ids,
        "DOB": dob.strftime("%Y-%m-%d"),
        "DOD": dod.dt.strftime("%Y-%m-%d"),
        "Gender": rng.choice([1, 2], n, p=[0.43, 0.57]),
        "Race": rng.choice([1, 2, 3, 5], n, p=[0.85, 0.10, 0.03, 0.02]),
        "RenalDiseaseIndicator": rng.choice(["0", "Y"], n, p=[0.85, 0.15]),
        "State": rng.integers(1, 55, n),
        "County": rng.integers(0, 999, n),
        "NoOfMonths_PartACov": np.where(rng.random(n) < 0.97, 12, rng.integers(0, 12, n)),
        "NoOfMonths_PartBCov": np.where(rng.random(n) < 0.97, 12, rng.integers(0, 12, n)),
    })
    probs = [0.33, 0.49, 0.39, 0.15, 0.30, 0.35, 0.60, 0.67, 0.27, 0.25, 0.08]
    for col, p in zip(CHRONIC_COLUMNS, probs):
        df[col] = np.where(rng.random(n) < p, 1, 2)
    df["IPAnnualReimbursementAmt"] = np.where(rng.random(n) < 0.25, rng.gamma(1.5, 6000, n).round(-1), 0)
    df["IPAnnualDeductibleAmt"] = np.where(df["IPAnnualReimbursementAmt"] > 0, 1068, 0)
    df["OPAnnualReimbursementAmt"] = rng.gamma(1.3, 1000, n).round(-1)
    df["OPAnnualDeductibleAmt"] = rng.gamma(1.2, 250, n).round()
    return df


def _codes(rng, n_rows, max_codes, pool, fill_probs, dup_prob=0.0):
    out = np.full((n_rows, max_codes), np.nan, dtype=object)
    k = np.minimum((rng.random((n_rows, max_codes)) < fill_probs).cumprod(axis=1).sum(axis=1), max_codes)
    for i in range(n_rows):
        if k[i] == 0:
            continue
        vals = list(rng.choice(pool, k[i]))
        if dup_prob and k[i] > 1 and rng.random() < dup_prob:
            vals[-1] = vals[0]  # padded / repeated diagnosis code
        out[i, : k[i]] = vals
    return out


def _claims(rng, provider_rows, bene_ids, inpatient: bool, claim_start_id):
    rows = []
    for provider, n, fraud in provider_rows:
        if n == 0:
            continue
        rows.append(pd.DataFrame({"Provider": provider, "_fraud": fraud}, index=range(n)))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    n = len(df)
    fraud = df["_fraud"].to_numpy()
    df.insert(0, "ClaimID", [f"CLM{claim_start_id + i}" for i in range(n)])
    df.insert(0, "BeneID", rng.choice(bene_ids, n))
    start = _dates(rng, "2009-01-01", "2009-12-31", n)
    if inpatient:
        stay = rng.poisson(np.where(fraud, 6.5, 4.5)) + 1
        end = start + pd.to_timedelta(stay, unit="D")
        amt = rng.gamma(2.0, np.where(fraud, 6500, 4800)).round(-2)
        ded = np.full(n, 1068.0)
    else:
        stay = rng.poisson(np.where(fraud, 1.2, 0.8))
        end = start + pd.to_timedelta(stay, unit="D")
        amt = rng.gamma(1.1, np.where(fraud, 420, 260)).round(-1)
        ded = np.where(rng.random(n) < 0.03, rng.choice([10, 20, 50, 100], n), 0).astype(float)
    df["ClaimStartDt"] = start.strftime("%Y-%m-%d")
    df["ClaimEndDt"] = end.strftime("%Y-%m-%d")
    df["InscClaimAmtReimbursed"] = amt
    phys = rng.integers(100000, 999999, (n, 3))
    df["AttendingPhysician"] = np.where(rng.random(n) < 0.997, [f"PHY{p}" for p in phys[:, 0]], None)
    op_p = np.where(fraud, 0.55, 0.30) if inpatient else np.where(fraud, 0.28, 0.14)
    ot_p = np.where(fraud, 0.45, 0.25) if inpatient else np.where(fraud, 0.48, 0.32)
    df["OperatingPhysician"] = np.where(rng.random(n) < op_p, [f"PHY{p}" for p in phys[:, 1]], None)
    df["OtherPhysician"] = np.where(rng.random(n) < ot_p, [f"PHY{p}" for p in phys[:, 2]], None)
    if inpatient:
        df["AdmissionDt"] = df["ClaimStartDt"]
    df["ClmAdmitDiagnosisCode"] = np.where(rng.random(n) < (0.95 if inpatient else 0.25),
                                           rng.choice(_DIAG_POOL, n), None)
    df["DeductibleAmtPaid"] = np.where(rng.random(n) < 0.002, np.nan, ded)
    if inpatient:
        df["DischargeDt"] = df["ClaimEndDt"]
        df["DiagnosisGroupCode"] = rng.choice(_DRG_POOL, n)
    fill = np.where(fraud[:, None], 0.90, 0.84) if inpatient else np.where(fraud[:, None], 0.62, 0.52)
    diag = _codes(rng, n, 10, _DIAG_POOL, fill, dup_prob=0.0)
    # padded billing: repeat the admit code inside the diagnosis list
    pad = rng.random(n) < np.where(fraud, 0.55, 0.30)
    admit = df["ClmAdmitDiagnosisCode"].to_numpy(dtype=object)
    for i in np.nonzero(pad & pd.notna(admit))[0]:
        diag[i, 0] = admit[i]
    for j in range(10):
        df[f"ClmDiagnosisCode_{j + 1}"] = diag[:, j]
    proc_fill = (np.where(fraud[:, None], 0.62, 0.50) * np.array([[1, 0.45, 0.2, 0.05, 0.01, 0.0]])) if inpatient \
        else np.where(fraud[:, None], 0.004, 0.002) * np.ones((1, 6))
    procs = _codes(rng, n, 6, _PROC_POOL, proc_fill)
    for j in range(6):
        df[f"ClmProcedureCode_{j + 1}"] = procs[:, j]
    return df.drop(columns="_fraud")


def generate(n_providers=600, n_beneficiaries=9000, fraud_rate=0.10, seed=42,
             provider_start=51001, bene_start=11001, claim_start=100000):
    """Return (beneficiary, inpatient, outpatient, labels) DataFrames."""
    rng = np.random.default_rng(seed)
    bene = _beneficiaries(rng, n_beneficiaries, bene_start)
    providers = [f"PRV{provider_start + i}" for i in range(n_providers)]
    fraud = rng.random(n_providers) < fraud_rate
    size = rng.lognormal(mean=np.where(fraud, 4.6, 3.3), sigma=np.where(fraud, 0.9, 1.0))
    size = np.clip(size, 1, 2500).astype(int)
    ip_share = np.clip(rng.beta(np.where(fraud, 2.2, 0.6), np.where(fraud, 3.5, 6.0)), 0, 0.9)
    n_ip = rng.binomial(size, ip_share)
    n_op = size - n_ip
    ip = _claims(rng, zip(providers, n_ip, fraud), bene["BeneID"].to_numpy(), True, claim_start)
    op = _claims(rng, zip(providers, n_op, fraud), bene["BeneID"].to_numpy(), False, claim_start + 5_000_000)
    labels = pd.DataFrame({"Provider": providers, "PotentialFraud": np.where(fraud, "Yes", "No")})
    # keep only providers that actually have claims
    active = set(ip["Provider"]).union(op["Provider"])
    labels = labels[labels["Provider"].isin(active)].reset_index(drop=True)
    return bene, ip, op, labels


def write_kaggle_style(folder: Path, prefix="Train", **kwargs) -> dict[str, Path]:
    """Generate and write the four CSVs using the Kaggle file names."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    bene, ip, op, labels = generate(**kwargs)
    paths = {
        "beneficiary": folder / f"{prefix}_Beneficiarydata-{KAGGLE_SUFFIX}.csv",
        "inpatient": folder / f"{prefix}_Inpatientdata-{KAGGLE_SUFFIX}.csv",
        "outpatient": folder / f"{prefix}_Outpatientdata-{KAGGLE_SUFFIX}.csv",
        "labels": folder / f"{prefix}-{KAGGLE_SUFFIX}.csv",
    }
    bene.to_csv(paths["beneficiary"], index=False)
    ip.to_csv(paths["inpatient"], index=False)
    op.to_csv(paths["outpatient"], index=False)
    labels.to_csv(paths["labels"], index=False)
    return paths


def find_kaggle_files(folder: Path) -> dict[str, Path]:
    """Locate beneficiary / inpatient / outpatient / label files in a folder by name."""
    folder = Path(folder)
    found = {}
    for p in sorted(folder.glob("*.csv")):
        name = p.name.lower()
        if "beneficiary" in name:
            found.setdefault("beneficiary", p)
        elif "inpatient" in name:
            found.setdefault("inpatient", p)
        elif "outpatient" in name:
            found.setdefault("outpatient", p)
        elif name.startswith(("train-", "test-", "labels")):
            found.setdefault("labels", p)
    return found
