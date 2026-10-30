
import os
import time
import re

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv(
    "GEMINI_MODEL",
    "gemini-flash-lite-latest"
)

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Check your .env file."
    )

client = genai.Client(api_key=api_key)


def generate_answer(question, retrieved_projects):
    # Prepare retrieved project records
    context = "\n\n".join(
        str(item["project"])
        for item in retrieved_projects
    )

    # Collect only IDs present in retrieved records
    valid_ids = {
        str(item["project"].get("project_id"))
        for item in retrieved_projects
        if item["project"].get("project_id")
    }

    prompt = f"""
You are the PAIMANA infrastructure project assistant.

STRICT RULES:
1. Answer only using the supplied project records.
2. Never invent project names, IDs, dates, figures, or risk levels.
3. Cite every factual project-specific claim using its exact project ID.
4. Use citations in this format: [PAI001].
5. Only cite IDs that appear in the supplied records.
6. If information is missing, clearly state that it is unavailable.
7. Do not assume that a revised completion date automatically
   proves a delay. Compare it with the planned completion date.
8. Explain the evidence briefly and clearly.

VALID PROJECT IDs:
{", ".join(sorted(valid_ids)) if valid_ids else "No project IDs available"}

PROJECT RECORDS:
{context}

USER QUESTION:
{question}

Write a clear, simple answer with project IDs as citations.
"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            answer = response.text or "Gemini returned an empty answer."

            # Validate citations against retrieved project IDs
            cited_ids = re.findall(r"\[(.*?)\]", answer)

            invalid_ids = [
                project_id
                for project_id in cited_ids
                if project_id not in valid_ids
            ]

            if invalid_ids:
                return (
                    "The generated answer contained invalid project "
                    "citations. Please retry the question."
                )

            return answer

        except ServerError:
            if attempt == 2:
                raise

            print("Gemini is busy. Retrying in 3 seconds...")
            time.sleep(3)
