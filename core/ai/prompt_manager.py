from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()

class PromptManager:
    def __init__(self, messages = [], model="gpt-4o-mini"):
        self.model = model
        self.messages = messages

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def set_messages(self, messages):
        self.messages = messages

    def generate(self):
        response = openai_client.chat.completions.create(
            model = self.model,
            messages= self.messages
        )

        content = response.choices[0].message.content
        return content

    def generate_structure(self, schema):
        response = openai_client.beta.chat.completions.parse(
            model = self.model,
            messages= self.messages,
            response_format=schema
        )

        content = response.choices[0].message.parsed
        data = schema.model_dump(content)

        return data
