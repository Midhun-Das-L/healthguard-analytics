from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("api/dashboard/", views.api_dashboard, name="api_dashboard"),
    path("api/readmission/", views.api_readmission, name="api_readmission"),

    path("fraud/upload/", views.fraud_upload, name="fraud_upload"),
    path("fraud/batches/<int:batch_id>/delete/", views.batch_delete, name="batch_delete"),
    path("fraud/providers/", views.provider_list, name="provider_list"),
    path("fraud/providers/export/", views.provider_export, name="provider_export"),
    path("fraud/providers/<int:pk>/", views.provider_detail, name="provider_detail"),
    path("fraud/samples/<str:which>/<str:filename>", views.sample_download, name="sample_download"),

    path("readmission/", views.readmission, name="readmission"),
    path("readmission/encounters/<int:encounter_id>/", views.encounter_detail, name="encounter_detail"),
    path("readmission/encounters/<int:encounter_id>/follow-up/", views.encounter_follow_up,
         name="encounter_follow_up"),
    path("readmission/score/", views.readmission_score_file, name="readmission_score_file"),

    path("models/", views.model_performance, name="model_performance"),
]
