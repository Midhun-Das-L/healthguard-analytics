import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="EncounterScore",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("encounter_id", models.BigIntegerField(unique=True)),
                ("patient_nbr", models.BigIntegerField(db_index=True)),
                ("age", models.PositiveIntegerField(default=0)),
                ("gender", models.CharField(blank=True, max_length=10)),
                ("race", models.CharField(blank=True, max_length=30)),
                ("on_insulin", models.BooleanField(default=False)),
                ("medication_changed", models.BooleanField(default=False)),
                ("readmission_probability", models.FloatField(db_index=True)),
                ("risk_level", models.CharField(db_index=True, max_length=10)),
                ("actual_readmitted", models.BooleanField(blank=True, null=True)),
                ("features", models.JSONField(default=dict)),
                ("follow_up_scheduled", models.BooleanField(default=False)),
            ],
            options={"ordering": ["-readmission_probability"]},
        ),
        migrations.CreateModel(
            name="ScoringBatch",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=200)),
                ("source", models.CharField(choices=[("sample", "Sample data"), ("upload", "Uploaded files")],
                                            default="upload", max_length=20)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("model_name", models.CharField(blank=True, max_length=100)),
                ("n_providers", models.PositiveIntegerField(default=0)),
                ("n_claims", models.PositiveIntegerField(default=0)),
                ("n_beneficiaries", models.PositiveIntegerField(default=0)),
                ("total_reimbursed", models.FloatField(default=0)),
                ("has_labels", models.BooleanField(default=False)),
                ("created_by", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL,
                                                 to=settings.AUTH_USER_MODEL)),
            ],
            options={"ordering": ["-created_at"], "verbose_name_plural": "scoring batches"},
        ),
        migrations.CreateModel(
            name="ProviderScore",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("provider_id", models.CharField(db_index=True, max_length=50)),
                ("fraud_probability", models.FloatField(db_index=True)),
                ("risk_level", models.CharField(choices=[("High", "High"), ("Medium", "Medium"), ("Low", "Low")],
                                                db_index=True, max_length=10)),
                ("actual_fraud", models.BooleanField(blank=True, null=True)),
                ("n_beneficiaries", models.PositiveIntegerField(default=0)),
                ("n_claims", models.PositiveIntegerField(default=0)),
                ("inpatient_claims", models.PositiveIntegerField(default=0)),
                ("total_reimbursed", models.FloatField(default=0)),
                ("avg_patient_age", models.FloatField(default=0)),
                ("features", models.JSONField(default=dict)),
                ("review_status", models.CharField(
                    choices=[("new", "Not reviewed"), ("investigating", "Investigating"),
                             ("confirmed", "Fraud confirmed"), ("cleared", "Cleared")],
                    db_index=True, default="new", max_length=20)),
                ("notes", models.TextField(blank=True)),
                ("reviewed_at", models.DateTimeField(blank=True, null=True)),
                ("batch", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="providers",
                                            to="home.scoringbatch")),
                ("reviewed_by", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL,
                                                  related_name="provider_reviews", to=settings.AUTH_USER_MODEL)),
            ],
            options={"ordering": ["-fraud_probability"]},
        ),
        migrations.AddConstraint(
            model_name="providerscore",
            constraint=models.UniqueConstraint(fields=("batch", "provider_id"), name="uniq_provider_per_batch"),
        ),
    ]
