from django.db import models
import datetime
import os

def imagepath(request, filename):
    old_filename = filename 
    curTime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    filename = "%s%s", (curTime, old_filename)
    return os.path.join('images/', filename)

class Prediction(models.Model):
    image = models.FileField(upload_to='images/', null=True, blank=True)
    result = models.TextField(null=True, blank=True)


