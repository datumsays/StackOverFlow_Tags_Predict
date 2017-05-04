from django.db import models

# Create your models here.
class Posts(models.Model):
    content = models.CharField(max_length=50000)
    tag = models.CharField(max_length=1000)
