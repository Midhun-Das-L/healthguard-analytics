from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management import BaseCommand, CommandError

from home.ml import engine, fraud_features, sample_data


class Command(BaseCommand):
    help = ("Train the provider-fraud and readmission models. Point --fraud-dir at a folder with the Kaggle "
            "Train_Beneficiarydata / Train_Inpatientdata / Train_Outpatientdata / Train-*.csv files to train on "
            "real data (default: the bundled sample).")

    def add_arguments(self, parser):
        parser.add_argument("--fraud-dir", default=None)
        parser.add_argument("--readmission-csv", default=None)
        parser.add_argument("--only", choices=["fraud", "readmission"], default=None)

    def handle(self, *args, **opts):
        out = Path(settings.ML_MODELS_DIR)
        if opts["only"] in (None, "fraud"):
            folder = Path(opts["fraud_dir"] or settings.FRAUD_SAMPLE_DIR)
            files = sample_data.find_kaggle_files(folder)
            missing = {"beneficiary", "inpatient", "outpatient", "labels"} - set(files)
            if missing:
                raise CommandError(f"{folder} is missing: {', '.join(sorted(missing))}")
            self.stdout.write(f"Building provider features from {folder} ...")
            frames = {k: pd.read_csv(v, low_memory=False) for k, v in files.items()}
            feats = fraud_features.build_provider_features(frames["beneficiary"], frames["inpatient"],
                                                           frames["outpatient"])
            feats = fraud_features.attach_labels(feats, frames["labels"])
            source = "sample" if folder.resolve() == Path(settings.FRAUD_SAMPLE_DIR).resolve() else str(folder)
            b = engine.train_fraud_model(feats, out, source=source)
            m = b["metrics"]
            self.stdout.write(self.style.SUCCESS(
                f"Fraud model: {b['model_name']}  ROC-AUC={m['roc_auc']}  precision={m['precision']}  "
                f"recall={m['recall']}  ({b['n_train_providers']} providers)"))
        if opts["only"] in (None, "readmission"):
            csv_path = Path(opts["readmission_csv"] or settings.READMISSION_DATA)
            b = engine.train_readmission_model(csv_path, out)
            m = b["metrics"]
            self.stdout.write(self.style.SUCCESS(
                f"Readmission model: ROC-AUC={m['roc_auc']}  recall={m['recall']}  ({b['n_rows']} encounters)"))
