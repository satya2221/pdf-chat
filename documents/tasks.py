from huey.contrib.djhuey import task
from .models import Document, DOC_STATUS_COMPLETE
from core.methods import send_notification
from core.ai.mistral import mistral
from core.ai.prompt_manager import PromptManager
from core.ai.chromadb import chroma, openai_ef
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import json

@task()
def process_document(document: Document):
    send_notification("notification", "Processing document")
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

    content = ""

    for page in ocr_result.model_dump().get("pages", []):
        content += page["markdown"]

    send_notification("notification", "Summarizing document")
    pm = PromptManager()
    pm.add_message("system", "Please Summarize this following text. Extract the key points too")
    pm.add_message("user", f"Content:{content}")


    summarized_content = pm.generate()

    document.raw_text = content
    document.summary = summarized_content
    document.status = DOC_STATUS_COMPLETE
    document.save()

    send_notification("notification", "Creating document")

    splitter = SemanticChunker(OpenAIEmbeddings())
    documents = splitter.create_documents([content])

    collection = chroma.create_collection(name=str(document.id), embedding_function=openai_ef)
    collection.add(
        documents=[doc.model_dump().get("page_content")  for doc in documents],
        ids=[str(i) for i in range(len(documents))]
    )
    collection = chroma.get_collection(str(document.id))
    send_notification("notification", "DONE!!!")