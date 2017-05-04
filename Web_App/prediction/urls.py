from django.conf.urls import url
from . import views

#Include URL patterns
urlpatterns = [
    # ex: /prediction/
    url(r'^', views.content_to_predict, name='content_to_predict'),
    # ex: /prediction/5/
    url(r'^show_prediction_result/', views.show_prediction_result, name='show_prediction_result')
]