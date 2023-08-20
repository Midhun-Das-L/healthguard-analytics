from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('fraud_detection',views.fraud_detection_page,name='fraud_detection'),
    path('re_add',views.re_add_prediction,name='readdmission'),
]
