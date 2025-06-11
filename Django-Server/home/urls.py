from django.contrib import admin
from django.urls import path
# added here manually
from home import views

urlpatterns = [
    path("", views.index, name = 'home'),
    path("about", views.about, name = 'about'),
    path("services", views.services, name = 'about'),
    path("contacts", views.contacts, name = 'contacts'),
    path("chatbot", views.chatbot, name = 'chatbot'),
    path('chatbot_response/', views.chatbot_response, name='chatbot_response'),
    path('profile/', views.view_profile, name='view_profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile')
]
