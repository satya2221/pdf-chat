from django.shortcuts import render, redirect
from django.views.generic import View
from .models import Document

# Create your views here.
class DocumentUploadView(View):
    def get(self, request):
        return render(request, "documents/index.html")
    
    def post(self, request):
        file = request.FILES.get("file")

        try:
            Document.objects.create(file=file, name=file.name)
        except Exception as e:
            print(e)

        return redirect("documents")