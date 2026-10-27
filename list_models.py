
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"timeout": 30000},
)

try:
    print("Models supporting content generation:\n")

    for model in client.models.list():
        supported = getattr(
            model, "supported_actions", []
        ) or []

        if "generateContent" in supported:
            print(model.name)

except Exception as error:
    print("Could not list models:")
    print(type(error).__name__)
    print(error)
