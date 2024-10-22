from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    # path('input.do/', views.input),
    path('result.do/', views.result),
    path('page1.do/', views.page1),
    path('page2.do/', views.page2),
    path('page3.do/', views.page3),
]

