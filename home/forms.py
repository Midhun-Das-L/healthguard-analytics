from django import forms

from .models import ProviderScore

CSV_ATTRS = {"class": "form-control", "accept": ".csv,text/csv"}


class FraudUploadForm(forms.Form):
    name = forms.CharField(max_length=200, required=False,
                           widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "e.g. Q3 2025 claims"}))
    beneficiary = forms.FileField(label="Beneficiary data", widget=forms.ClearableFileInput(attrs=CSV_ATTRS))
    inpatient = forms.FileField(label="Inpatient claims", widget=forms.ClearableFileInput(attrs=CSV_ATTRS))
    outpatient = forms.FileField(label="Outpatient claims", widget=forms.ClearableFileInput(attrs=CSV_ATTRS))
    labels = forms.FileField(label="Known outcomes (optional)", required=False,
                             help_text="Provider, PotentialFraud (Yes/No) - used to measure accuracy.",
                             widget=forms.ClearableFileInput(attrs=CSV_ATTRS))

    def clean(self):
        data = super().clean()
        for key in ("beneficiary", "inpatient", "outpatient", "labels"):
            f = data.get(key)
            if f and not f.name.lower().endswith(".csv"):
                self.add_error(key, "Please upload a .csv file.")
        return data


class ReadmissionUploadForm(forms.Form):
    file = forms.FileField(label="Encounters CSV", widget=forms.ClearableFileInput(attrs=CSV_ATTRS))


class ReviewForm(forms.ModelForm):
    class Meta:
        model = ProviderScore
        fields = ["review_status", "notes"]
        widgets = {
            "review_status": forms.Select(attrs={"class": "form-select"}),
            "notes": forms.Textarea(attrs={"class": "form-control", "rows": 3,
                                           "placeholder": "Investigation notes..."}),
        }
