from django.db import models
from django.utils import timezone

# Create your models here.

class ChatConversation(models.Model):
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']  # Most recent conversations first

    def __str__(self):
        return f"{self.title} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class ChatMessage(models.Model):
    conversation = models.ForeignKey(ChatConversation, on_delete=models.CASCADE, related_name='messages')
    message = models.TextField()
    is_user = models.BooleanField(default=True)  # True for user messages, False for bot responses
    timestamp = models.DateTimeField(default=timezone.now)
    sequence = models.IntegerField(default=0)  # Add sequence field for explicit ordering

    class Meta:
        ordering = ['sequence', 'timestamp']  # Order by sequence first, then timestamp

    def __str__(self):
        return f"{'User' if self.is_user else 'Bot'}: {self.message[:50]}..."
