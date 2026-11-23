import csv
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from . import services
from .forms import FraudUploadForm, ReadmissionUploadForm, ReviewForm
from .ml import engine
from .ml.fraud_features import FEATURE_COLUMNS, FEATURE_LABELS, FeatureError
from .ml.sample_data import find_kaggle_files
from .models import EncounterScore, ProviderScore, ScoringBatch


def _threshold(request):
    try:
        t = float(request.GET.get("threshold", settings.FRAUD_FLAG_THRESHOLD))
    except ValueError:
        t = settings.FRAUD_FLAG_THRESHOLD
    return min(max(t, 0.05), 0.95)


def _selected_batch(request):
    batch_id = request.GET.get("batch")
    if batch_id and batch_id.isdigit():
        return ScoringBatch.objects.filter(pk=batch_id).first() or services.latest_batch()
    return services.latest_batch()


def _setup_needed(request):
    return render(request, "pages/setup_needed.html", status=200)


# ------------------------------------------------------------------- dashboard
@login_required
def dashboard(request):
    if services.fraud_bundle(required=False) is None or not ScoringBatch.objects.exists():
        return _setup_needed(request)
    batch = _selected_batch(request)
    return render(request, "pages/dashboard.html", {
        "batches": ScoringBatch.objects.all()[:50],
        "batch": batch,
        "threshold": _threshold(request),
        "fraud_info": engine.public_info(services.fraud_bundle()),
        "has_encounters": EncounterScore.objects.exists(),
        "segment": "dashboard",
    })


@login_required
def api_dashboard(request):
    batch = _selected_batch(request)
    if batch is None:
        return JsonResponse({"error": "No scoring batches yet."}, status=404)
    data = services.fraud_dashboard(batch, _threshold(request))
    data["readmission"] = services.readmission_dashboard() if EncounterScore.objects.exists() else None
    return JsonResponse(data)


@login_required
def api_readmission(request):
    return JsonResponse(services.readmission_dashboard(services.filter_encounters(request.GET)))


# ----------------------------------------------------------------------- fraud
@login_required
def fraud_upload(request):
    if services.fraud_bundle(required=False) is None:
        return _setup_needed(request)
    form = FraudUploadForm(request.POST or None, request.FILES or None)
    if request.method == "POST" and form.is_valid():
        try:
            frames = {k: services.read_csv(form.cleaned_data[k]) for k in ("beneficiary", "inpatient", "outpatient")}
            labels = services.read_csv(form.cleaned_data["labels"]) if form.cleaned_data.get("labels") else None
            name = form.cleaned_data.get("name") or f"Upload {timezone.localtime():%Y-%m-%d %H:%M}"
            batch = services.create_batch(name, frames["beneficiary"], frames["inpatient"], frames["outpatient"],
                                          labels, user=request.user)
        except (FeatureError, ValueError, KeyError) as exc:
            messages.error(request, f"Could not score these files: {exc}")
        else:
            flagged = batch.providers.filter(fraud_probability__gte=settings.FRAUD_FLAG_THRESHOLD).count()
            messages.success(request, f"Scored {batch.n_providers} providers from {batch.n_claims:,} claims - "
                                      f"{flagged} flagged for review.")
            return redirect(f"/?batch={batch.id}")
    demo = {k: v.name for k, v in find_kaggle_files(settings.FRAUD_DEMO_UPLOAD_DIR).items()}
    return render(request, "pages/fraud_upload.html", {
        "form": form, "batches": ScoringBatch.objects.all()[:20], "demo_files": demo, "segment": "upload",
    })


@login_required
@require_POST
def batch_delete(request, batch_id):
    batch = get_object_or_404(ScoringBatch, pk=batch_id)
    if ScoringBatch.objects.count() <= 1:
        messages.error(request, "Keep at least one batch so the dashboard has data.")
    else:
        name = batch.name
        batch.delete()
        messages.success(request, f"Deleted batch '{name}'.")
    return redirect("fraud_upload")


@login_required
def provider_list(request):
    batch = _selected_batch(request)
    if batch is None:
        return _setup_needed(request)
    qs = services.filter_providers(batch, request.GET)
    page = Paginator(qs, 25).get_page(request.GET.get("page"))
    params = request.GET.copy()
    params.pop("page", None)
    return render(request, "pages/providers.html", {
        "batch": batch, "batches": ScoringBatch.objects.all()[:50], "page": page, "total": qs.count(),
        "querystring": params.urlencode(), "statuses": ProviderScore.STATUS_CHOICES, "f": request.GET,
        "threshold": settings.FRAUD_FLAG_THRESHOLD, "segment": "providers",
    })


@login_required
def provider_export(request):
    batch = _selected_batch(request)
    if batch is None:
        raise Http404
    qs = services.filter_providers(batch, request.GET)
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = f'attachment; filename="provider_risk_batch{batch.id}.csv"'
    w = csv.writer(resp)
    w.writerow(["Provider", "FraudProbability", "RiskLevel", "ReviewStatus", "Notes", "ActualFraud",
                *FEATURE_COLUMNS])
    for p in qs.iterator():
        w.writerow([p.provider_id, round(p.fraud_probability, 4), p.risk_level, p.get_review_status_display(),
                    p.notes, "" if p.actual_fraud is None else ("Yes" if p.actual_fraud else "No"),
                    *[p.features.get(c, "") for c in FEATURE_COLUMNS]])
    return resp


@login_required
def provider_detail(request, pk):
    provider = get_object_or_404(ProviderScore.objects.select_related("batch", "reviewed_by"), pk=pk)
    form = ReviewForm(request.POST or None, instance=provider)
    if request.method == "POST" and form.is_valid():
        obj = form.save(commit=False)
        obj.reviewed_at = timezone.now()
        obj.reviewed_by = request.user
        obj.save()
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"ok": True, "status": obj.get_review_status_display()})
        messages.success(request, f"Review saved for {provider.provider_id}.")
        return redirect("provider_detail", pk=pk)
    bundle = services.fraud_bundle()
    explanation = engine.explain_provider(bundle, provider.features)
    features = [{"key": c, "label": FEATURE_LABELS.get(c, c), "value": provider.features.get(c, 0),
                 "honest": round(float(bundle["baseline_honest"].get(c, 0)), 2)} for c in FEATURE_COLUMNS]
    peers = ProviderScore.objects.filter(batch=provider.batch)
    rank = peers.filter(fraud_probability__gt=provider.fraud_probability).count() + 1
    return render(request, "pages/provider_detail.html", {
        "p": provider, "form": form, "explanation": explanation, "features": features,
        "rank": rank, "peer_count": peers.count(), "segment": "providers",
    })


SAMPLE_DIRS = {"demo": "FRAUD_DEMO_UPLOAD_DIR", "train": "FRAUD_SAMPLE_DIR"}


@login_required
def sample_download(request, which, filename):
    if which not in SAMPLE_DIRS:
        raise Http404
    folder = Path(getattr(settings, SAMPLE_DIRS[which]))
    allowed = {p.name: p for p in find_kaggle_files(folder).values()}
    if filename not in allowed:
        raise Http404
    return FileResponse(open(allowed[filename], "rb"), as_attachment=True, filename=filename)


# ----------------------------------------------------------------- readmission
@login_required
def readmission(request):
    if services.readmission_bundle(required=False) is None or not EncounterScore.objects.exists():
        return _setup_needed(request)
    qs = services.filter_encounters(request.GET)
    page = Paginator(qs, 25).get_page(request.GET.get("page"))
    params = request.GET.copy()
    params.pop("page", None)
    return render(request, "pages/readmission.html", {
        "page": page, "total": qs.count(), "querystring": params.urlencode(), "f": request.GET,
        "info": engine.public_info(services.readmission_bundle()), "upload_form": ReadmissionUploadForm(),
        "segment": "readmission",
    })


@login_required
def encounter_detail(request, encounter_id):
    enc = get_object_or_404(EncounterScore, encounter_id=encounter_id)
    bundle = services.readmission_bundle()
    factors = engine.explain_encounter(bundle, enc.features)
    others = EncounterScore.objects.filter(patient_nbr=enc.patient_nbr).exclude(pk=enc.pk)[:10]
    return render(request, "pages/encounter_detail.html", {
        "e": enc, "factors": factors, "others": others, "segment": "readmission",
    })


@login_required
@require_POST
def encounter_follow_up(request, encounter_id):
    enc = get_object_or_404(EncounterScore, encounter_id=encounter_id)
    enc.follow_up_scheduled = request.POST.get("value", "1") == "1"
    enc.save(update_fields=["follow_up_scheduled"])
    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        return JsonResponse({"ok": True, "follow_up": enc.follow_up_scheduled})
    return redirect(request.POST.get("next") or "readmission")


@login_required
@require_POST
def readmission_score_file(request):
    form = ReadmissionUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        messages.error(request, "Please choose a CSV file.")
        return redirect("readmission")
    try:
        df = services.read_csv(form.cleaned_data["file"])
        scored = engine.score_encounters(services.readmission_bundle(), df)
    except (FeatureError, ValueError) as exc:
        messages.error(request, f"Could not score file: {exc}")
        return redirect("readmission")
    keep = [c for c in ("encounter_id", "patient_nbr") if c in scored.columns]
    out = scored[keep + ["readmission_probability", "risk_level"]].sort_values(
        "readmission_probability", ascending=False)
    out["readmission_probability"] = out["readmission_probability"].round(4)
    resp = HttpResponse(out.to_csv(index=False), content_type="text/csv")
    resp["Content-Disposition"] = 'attachment; filename="readmission_scores.csv"'
    return resp


# --------------------------------------------------------------------- models
@login_required
def model_performance(request):
    return render(request, "pages/models.html", {
        "fraud": engine.public_info(services.fraud_bundle(required=False)),
        "readm": engine.public_info(services.readmission_bundle(required=False)),
        "segment": "models",
    })
