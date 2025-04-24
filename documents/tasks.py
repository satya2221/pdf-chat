from huey.contrib.djhuey import task
from .models import Document, DOC_STATUS_COMPLETE
from .methods import process_ocr, process_summarizing, process_split, process_vector
from core.methods import send_notification
from core.ai.mistral import mistral



import json

@task()
def process_document(document: Document):

    # Process OCR
    send_notification("notification", "Processing document")
    ocr_content = process_ocr(document)

    # Process Summarizing
    send_notification("notification", "Summarizing document")
    process_summarizing(document, ocr_content)

    # process splitting
    send_notification("notification", "Creating document")
    splitted_documents =  process_split(ocr_content)

    # Process input to Vector DB
    process_vector(document, splitted_documents)

    send_notification("notification", "DONE!!!")