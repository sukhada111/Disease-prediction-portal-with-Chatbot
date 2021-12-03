"""prediction_portal URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from home import views
from register import views as v
from medicine_info import views as view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home,name='home'),
    path('medicine_info/',view.medicine_info, name='medicine_info'),
    path('register/',v.registration_view,name='register'),
    path('login/',v.login_view,name='login'),
     path('logout/',v.logout_view,name='logout')


]
