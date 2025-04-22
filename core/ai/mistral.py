from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

mistral = Mistral(api_key=MISTRAL_API_KEY)