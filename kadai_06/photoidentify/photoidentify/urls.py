from django.contrib import admin
from django.urls import path
from predictor.views import predict

urlpatterns = [
    path('', predict, name='predict'),
]
