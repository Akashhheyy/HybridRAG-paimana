
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)


def generate_answer(question, retrieved_projects):
    context = "\n\n".join(
        str(item["project"])
        for item in retrieved_projects
    )

    prompt = f"""
You are the PAIMANA infrastructure project assistant.

Answer using only the project records below.
Do not invent facts, figures, or risk assessments.
If information is missing, clearly say so.
Identify the project records used as evidence.

PROJECT RECORDS:
{context}

USER QUESTION:
{question}

Give a clear, simple answer.
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text or "Gemini returned an empty answer."

        except ServerError:
            if attempt == 2:
                raise
            print("Gemini is busy. Retrying in 3 seconds...")
            time.sleep(3)
