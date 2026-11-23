from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import BaseCommand, call_command

from home import services
from home.ml import sample_data
from home.models import ScoringBatch


class Command(BaseCommand):
    help = ("One-step setup: generate sample claims (if missing), train both models, score the sample data "
            "into the database and create a login user.")

    def add_arguments(self, parser):
        parser.add_argument("--regenerate", action="store_true", help="Recreate the synthetic sample CSVs.")
        parser.add_argument("--username", default="admin")
        parser.add_argument("--password", default="admin12345")
        parser.add_argument("--skip-user", action="store_true")

    def handle(self, *args, **opts):
        call_command("migrate", interactive=False, verbosity=0)
        self.stdout.write("Database migrated.")

        if opts["regenerate"] or not sample_data.find_kaggle_files(settings.FRAUD_SAMPLE_DIR):
            sample_data.write_kaggle_style(settings.FRAUD_SAMPLE_DIR, "Train")
            self.stdout.write("Generated sample training claims.")
        if opts["regenerate"] or not sample_data.find_kaggle_files(settings.FRAUD_DEMO_UPLOAD_DIR):
            sample_data.write_kaggle_style(settings.FRAUD_DEMO_UPLOAD_DIR, "Test", n_providers=150,
                                           n_beneficiaries=3000, seed=7, provider_start=60001,
                                           bene_start=31001, claim_start=900000)
            self.stdout.write("Generated demo upload files.")

        call_command("train_models", stdout=self.stdout)

        ScoringBatch.objects.filter(source=ScoringBatch.SOURCE_SAMPLE).delete()
        batch = services.create_sample_batch()
        self.stdout.write(f"Scored sample batch: {batch.n_providers} providers.")
        n = services.seed_encounters()
        self.stdout.write(f"Scored {n} readmission encounters.")

        if not opts["skip_user"]:
            User = get_user_model()
            if not User.objects.filter(username=opts["username"]).exists():
                User.objects.create_superuser(opts["username"], "", opts["password"])
                self.stdout.write(self.style.WARNING(
                    f"Created superuser '{opts['username']}' / '{opts['password']}' - change this password."))
        self.stdout.write(self.style.SUCCESS("Done. Run: python manage.py runserver  ->  http://127.0.0.1:8000"))
