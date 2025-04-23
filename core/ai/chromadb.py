from chromadb import HttpClient
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os

chroma = HttpClient(port=8010)

openai_ef = OpenAIEmbeddingFunction(model_name="text-embedding-3-small")