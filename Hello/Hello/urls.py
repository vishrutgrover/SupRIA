"""
URL configuration for Hello project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from home import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", include("home.urls")),
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('services/', views.services, name='services'),
    path('contacts/', views.contacts, name='contacts'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('chatbot_response/', views.chatbot_response, name='chatbot_response'),
    path('get_conversations/', views.get_conversations, name='get_conversations'),
    path('get_conversation/<int:conversation_id>/', views.get_conversation, name='get_conversation'),
    path('delete_conversation/<int:conversation_id>/', views.delete_conversation, name='delete_conversation'),
] + static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
