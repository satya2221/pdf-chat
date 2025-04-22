from django.shortcuts import render, redirect
from django.views.generic import View
from .models import Document
from core.ai.mistral import mistral
from .tasks import process_document

# Create your views here.
class DocumentUploadView(View):
    def get(self, request):
        return render(request, "documents/index.html")
    
    def post(self, request):
        file = request.FILES.get("file")

        try:
            document = Document.objects.create(file=file, name=file.name)

            process_document(document)

        except Exception as e:
            print(e)

        return redirect("documents")