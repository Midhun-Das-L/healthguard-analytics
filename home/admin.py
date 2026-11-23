from django.contrib import admin

from .models import EncounterScore, ProviderScore, ScoringBatch


@admin.register(ScoringBatch)
class ScoringBatchAdmin(admin.ModelAdmin):
    list_display = ("name", "source", "model_name", "n_providers", "n_claims", "created_at", "created_by")
    list_filter = ("source",)


@admin.register(ProviderScore)
class ProviderScoreAdmin(admin.ModelAdmin):
    list_display = ("provider_id", "batch", "fraud_probability", "risk_level", "review_status", "actual_fraud")
    list_filter = ("risk_level", "review_status", "batch")
    search_fields = ("provider_id",)
    exclude = ("features",)


@admin.register(EncounterScore)
class EncounterScoreAdmin(admin.ModelAdmin):
    list_display = ("encounter_id", "patient_nbr", "age", "readmission_probability", "risk_level",
                    "actual_readmitted", "follow_up_scheduled")
    list_filter = ("risk_level", "follow_up_scheduled", "gender")
    search_fields = ("encounter_id", "patient_nbr")
    exclude = ("features",)
