from .models import Document, DOC_STATUS_COMPLETE
from core.ai.mistral import mistral
from core.ai.prompt_manager import PromptManager
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from typing import List
from core.ai.chromadb import chroma, openai_ef

def process_ocr(document: Document):
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
    
    return content

def process_summarizing(document: Document, content):
    pm = PromptManager()
    pm.add_message("system", "Please Summarize this following text. Extract the key points too")
    pm.add_message("user", f"Content:{content}")


    summarized_content = pm.generate()

    document.raw_text = content
    document.summary = summarized_content
    document.status = DOC_STATUS_COMPLETE
    document.save()

    
def process_split(content):
    splitter = SemanticChunker(OpenAIEmbeddings())
    documents = splitter.create_documents([content])

    return documents

def process_vector(document: Document, documents:List[str]):
    collection = chroma.create_collection(name=str(document.id), embedding_function=openai_ef)
    collection.add(
        documents=[doc.model_dump().get("page_content")  for doc in documents],
        ids=[str(i) for i in range(len(documents))]
    )
    collection = chroma.get_collection(str(document.id))
    print(collection.count())