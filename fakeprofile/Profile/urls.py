from django.urls import path

from . import views

urlpatterns = [path("Home.html", views.index, name="index"),
	       path("AdminLogin.html", views.Admin, name="Admin"),
	       path("AdminLogin", views.AdminLogin, name="AdminLogin"),
	       path("GenerateModel", views.GenerateModel, name="GenerateModel"),
	       path("ViewTrain", views.ViewTrain, name="ViewTrain"),
	       path("UserPannel.html", views.User, name="User"),
	       path("UserCheck", views.UserCheck, name="UserCheck"),
]
