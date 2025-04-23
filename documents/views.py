from django.shortcuts import render, redirect
from django.views.generic import View
from .models import Document
from core.ai.mistral import mistral
from core.ai.chromadb import chroma, openai_ef
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

class QueryView(View):
    def get(self, request):
        return render(request, "documents/query.html")
    
    def post(self, request):
        query = request.POST.get("query")

        collection = chroma.get_collection(name="6808c3cbd386bc09f6a81807", embedding_function=openai_ef)
        data = collection.query(
            query_texts=[query],
            n_results=4
        )
        print(data)