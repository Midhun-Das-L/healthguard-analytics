from django.conf import settings
from django.db import models


class ScoringBatch(models.Model):
    """One run of the fraud model over a set of claim files."""

    SOURCE_SAMPLE = "sample"
    SOURCE_UPLOAD = "upload"
    SOURCE_CHOICES = [(SOURCE_SAMPLE, "Sample data"), (SOURCE_UPLOAD, "Uploaded files")]

    name = models.CharField(max_length=200)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=SOURCE_UPLOAD)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    model_name = models.CharField(max_length=100, blank=True)
    n_providers = models.PositiveIntegerField(default=0)
    n_claims = models.PositiveIntegerField(default=0)
    n_beneficiaries = models.PositiveIntegerField(default=0)
    total_reimbursed = models.FloatField(default=0)
    has_labels = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "scoring batches"

    def __str__(self):
        return f"{self.name} ({self.created_at:%Y-%m-%d %H:%M})"


class ProviderScore(models.Model):
    """Fraud risk for one provider inside a batch, plus the investigator's review."""

    STATUS_NEW = "new"
    STATUS_CHOICES = [
        ("new", "Not reviewed"),
        ("investigating", "Investigating"),
        ("confirmed", "Fraud confirmed"),
        ("cleared", "Cleared"),
    ]
    RISK_CHOICES = [("High", "High"), ("Medium", "Medium"), ("Low", "Low")]

    batch = models.ForeignKey(ScoringBatch, on_delete=models.CASCADE, related_name="providers")
    provider_id = models.CharField(max_length=50, db_index=True)
    fraud_probability = models.FloatField(db_index=True)
    risk_level = models.CharField(max_length=10, choices=RISK_CHOICES, db_index=True)
    actual_fraud = models.BooleanField(null=True, blank=True)
    n_beneficiaries = models.PositiveIntegerField(default=0)
    n_claims = models.PositiveIntegerField(default=0)
    inpatient_claims = models.PositiveIntegerField(default=0)
    total_reimbursed = models.FloatField(default=0)
    avg_patient_age = models.FloatField(default=0)
    features = models.JSONField(default=dict)
    review_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_NEW, db_index=True)
    notes = models.TextField(blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewed_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                                    related_name="provider_reviews")

    class Meta:
        ordering = ["-fraud_probability"]
        constraints = [models.UniqueConstraint(fields=["batch", "provider_id"], name="uniq_provider_per_batch")]

    def __str__(self):
        return f"{self.provider_id} ({self.fraud_probability:.0%})"

    @property
    def inpatient_share(self):
        return self.inpatient_claims / self.n_claims if self.n_claims else 0

    @property
    def reimbursed_per_claim(self):
        return self.total_reimbursed / self.n_claims if self.n_claims else 0


class EncounterScore(models.Model):
    """Readmission risk for one diabetic inpatient encounter."""

    encounter_id = models.BigIntegerField(unique=True)
    patient_nbr = models.BigIntegerField(db_index=True)
    age = models.PositiveIntegerField(default=0)
    gender = models.CharField(max_length=10, blank=True)
    race = models.CharField(max_length=30, blank=True)
    on_insulin = models.BooleanField(default=False)
    medication_changed = models.BooleanField(default=False)
    readmission_probability = models.FloatField(db_index=True)
    risk_level = models.CharField(max_length=10, db_index=True)
    actual_readmitted = models.BooleanField(null=True, blank=True)
    features = models.JSONField(default=dict)
    follow_up_scheduled = models.BooleanField(default=False)

    class Meta:
        ordering = ["-readmission_probability"]

    def __str__(self):
        return f"Encounter {self.encounter_id} ({self.readmission_probability:.0%})"
