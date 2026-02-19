from django.urls import path
from . import views

urlpatterns = [
    # Main query endpoint
    path('ask/', views.ask, name='ask'),
    
    # Batch processing
    path('batch/', views.batch_query, name='batch_query'),
    
    # Chapter information
    path('chapter/<int:chapter_number>/', views.chapter_info, name='chapter_info'),
    path('chapters/', views.list_chapters, name='list_chapters'),
    
    # System health
    path('health/', views.system_health, name='health'),
]