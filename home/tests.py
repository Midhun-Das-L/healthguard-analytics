"""End-to-end tests. Run with:  python manage.py test home"""
import io
import shutil
import tempfile
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse

from home import services
from home.ml import sample_data
from home.models import EncounterScore, ProviderScore, ScoringBatch

TMP = Path(tempfile.mkdtemp(prefix="hg-tests-"))


@override_settings(ML_MODELS_DIR=TMP / "models", FRAUD_SAMPLE_DIR=TMP / "sample", FRAUD_DEMO_UPLOAD_DIR=TMP / "demo")
class HealthGuardTests(TestCase):
    @classmethod
    def setUpClass(cls):
        # Sample files must exist BEFORE super().setUpClass(), which runs setUpTestData (model training).
        sample_data.write_kaggle_style(TMP / "sample", "Train", n_providers=200, n_beneficiaries=2000, seed=3)
        sample_data.write_kaggle_style(TMP / "demo", "Test", n_providers=40, n_beneficiaries=800, seed=4,
                                       provider_start=70001, bene_start=50001, claim_start=700000)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)
        super().tearDownClass()

    @classmethod
    def setUpTestData(cls):
        # settings overrides are active here; trains once per class
        call_command("train_models", stdout=io.StringIO())
        services.create_sample_batch()
        services.seed_encounters()
        cls.user = get_user_model().objects.create_user("tester", password="pass12345!")

    def setUp(self):
        self.client.force_login(self.user)

    def test_pages_render(self):
        batch = ScoringBatch.objects.first()
        provider = ProviderScore.objects.first()
        enc = EncounterScore.objects.first()
        for url in [reverse("dashboard"), reverse("provider_list") + f"?batch={batch.id}&risk=High",
                    reverse("provider_detail", args=[provider.pk]), reverse("fraud_upload"),
                    reverse("readmission") + "?risk=High", reverse("encounter_detail", args=[enc.encounter_id]),
                    reverse("model_performance")]:
            with self.subTest(url=url):
                self.assertEqual(self.client.get(url).status_code, 200)

    def test_login_required(self):
        self.client.logout()
        self.assertEqual(self.client.get(reverse("dashboard")).status_code, 302)

    def test_dashboard_api(self):
        data = self.client.get(reverse("api_dashboard") + "?threshold=0.3").json()
        self.assertEqual(data["kpis"]["providers"], ScoringBatch.objects.first().n_providers)
        self.assertEqual(sum(data["histogram"]["counts"]), data["kpis"]["providers"])
        self.assertIsNotNone(data["evaluation"])
        self.assertIsNotNone(data["readmission"])
        low = self.client.get(reverse("api_dashboard") + "?threshold=0.2").json()["kpis"]["flagged"]
        high = self.client.get(reverse("api_dashboard") + "?threshold=0.8").json()["kpis"]["flagged"]
        self.assertGreaterEqual(low, high)

    def test_upload_scores_new_batch(self):
        files = sample_data.find_kaggle_files(TMP / "demo")
        post = {"name": "Demo upload"}
        for key in ("beneficiary", "inpatient", "outpatient", "labels"):
            post[key] = SimpleUploadedFile(files[key].name, files[key].read_bytes(), content_type="text/csv")
        resp = self.client.post(reverse("fraud_upload"), post)
        self.assertEqual(resp.status_code, 302)
        batch = ScoringBatch.objects.get(name="Demo upload")
        self.assertTrue(batch.has_labels)
        self.assertEqual(batch.providers.count(), batch.n_providers)

    def test_upload_rejects_bad_file(self):
        bad = SimpleUploadedFile("x.csv", b"a,b\n1,2\n", content_type="text/csv")
        resp = self.client.post(reverse("fraud_upload"), {"beneficiary": bad, "inpatient": bad, "outpatient": bad})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Could not score")

    def test_review_and_export(self):
        p = ProviderScore.objects.first()
        resp = self.client.post(reverse("provider_detail", args=[p.pk]),
                                {"review_status": "investigating", "notes": "Duplicate codes"},
                                HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertTrue(resp.json()["ok"])
        p.refresh_from_db()
        self.assertEqual(p.review_status, "investigating")
        csv_resp = self.client.get(reverse("provider_export") + f"?batch={p.batch_id}")
        self.assertIn(p.provider_id, csv_resp.content.decode())

    def test_follow_up_and_readmission_scoring(self):
        enc = EncounterScore.objects.first()
        self.client.post(reverse("encounter_follow_up", args=[enc.encounter_id]), {"value": "1"},
                         HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        enc.refresh_from_db()
        self.assertTrue(enc.follow_up_scheduled)
        data = Path(settings.READMISSION_DATA).read_bytes()[:20000]
        data = data[: data.rfind(b"\n") + 1]
        resp = self.client.post(reverse("readmission_score_file"),
                                {"file": SimpleUploadedFile("enc.csv", data, content_type="text/csv")})
        self.assertEqual(resp["Content-Type"], "text/csv")
        self.assertIn("readmission_probability", resp.content.decode().splitlines()[0])
