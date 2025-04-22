from django.shortcuts import render, redirect
from django.views.generic import View
from .models import Document
from core.ai.mistral import mistral

# Create your views here.
class DocumentUploadView(View):
    def get(self, request):
        return render(request, "documents/index.html")
    
    def post(self, request):
        file = request.FILES.get("file")

        try:
            document = Document.objects.create(file=file, name=file.name)

            uploaded_pdf = mistral.files.upload(
                file={
                    "file_name": document.name,
                    "content": open(f"media/documents/{document.name}", "rb")
                },
                purpose="ocr"
            )

            signed_url = mistral.files.get_signed_url(file_id=uploaded_pdf.id)
            print(signed_url)

            ocr_result = mistral.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type":"document_url",
                    "document_url": signed_url.url
                }
            )
            print(ocr_result)

        except Exception as e:
            print(e)

        return redirect("documents")