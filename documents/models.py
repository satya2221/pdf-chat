from django.db import models
from core.models import BaseModel

DOC_STATUS_PENDING = 'pending'
DOC_STATUS_PROCESSING = 'processing'
DOC_STATUS_COMPLETE = 'complete'

DOC_STATUS_CHOICES =(
    (DOC_STATUS_PENDING, 'Pending'), # 1 adalah Value, 2 label dalam admin panel
    (DOC_STATUS_PROCESSING, 'Processing'),
    (DOC_STATUS_COMPLETE, 'Complete')
)

class Document(BaseModel):
    name = models.CharField(max_length=200, blank=True, null=True)
    file = models.FileField(upload_to='documents/') #akan masuk folder documents dalam folder media

    raw_text = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    status = models.CharField(max_length=200, choices=DOC_STATUS_CHOICES ,default=DOC_STATUS_PENDING) # pending, process, done

