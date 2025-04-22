from huey.contrib.djhuey import task
from .models import Document
from core.ai.mistral import mistral

@task()
def process_document(document: Document):
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