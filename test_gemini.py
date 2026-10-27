import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = "gemini-flash-lite-latest"
if not api_key:
    print("ERROR: GEMINI_API_KEY is missing from .env")
else:
    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 30000},
    )

    print("Testing Gemini API...")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: Gemini is working!",
        )
        print("SUCCESS:")
        print(response.text)
    except Exception as error:
        print("FAILED:")
        print(type(error).__name__)
        print(error)