from django.contrib import admin
from django.urls import path
from classifierApp import views
from django.conf import settings

urlpatterns = [
    path('', views.index, name='index'),
    path('admin/', admin.site.urls),
]
