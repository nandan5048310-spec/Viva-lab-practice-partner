import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load settings from our private .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")


def test_connection():
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY is missing from .env")
        return

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=120000),
        )

        response = client.models.generate_content(
            model=MODEL,
            contents="Explain what a capacitor is in one simple sentence.",
        )

        print("SUCCESS! Gemma 4 responded:")
        print(response.text or "The model returned no text.")

    except Exception as error:
        print("API connection failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")


if __name__ == "__main__":
    test_connection()


import json


def generate_mcqs(
    study_text: str,
    count: int = 5,
    image_bytes: bytes | None = None
) -> dict:
    """Generate MCQs using Gemma 4."""


    if not isinstance(study_text, str):
        return {
            "success": False,
            "data": None,
            "error": "Study material must be text."
        }

    if not study_text.strip() and image_bytes is None:
        return {
            "success": False,
            "data": None,
            "error": "Please provide study text or an image."
        }


    if type(count) is not int or not 1 <= count <= 20:
        return {
            "success": False,
            "data": None,
            "error": "Question count must be between 1 and 20."
        }


    
    if image_bytes is not None:
        try:
            image_part = _prepare_image(image_bytes)
        except ValueError as error:
            return {
                "success": False,
                "data": None,
                "error": str(error)
            }

    else:
        image_part = None


    if not API_KEY:
        return {
            "success": False,
            "data": None,
            "error": "Gemini API key is missing."
        }

    prompt = f"""
You are a friendly and accurate lab viva quiz teacher.

Generate exactly {count} multiple-choice questions
using ONLY the study material below.

STUDY MATERIAL:
{study_text[:20000]}

Return ONLY a valid JSON object in this exact format:

{{
  "questions": [
    {{
      "id": "q1",
      "question": "Question text",
      "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option"
      }},
      "correct_option": "A",
      "explanation": "Simple explanation"
    }}
  ]
}}

Rules:
- Generate exactly {count} questions.
- Every question must have four options: A, B, C, D.
- Only one option should be correct.
- Explanations must be beginner-friendly.
- Do not invent facts not supported by the material.
- Return JSON only, without Markdown formatting.
"""

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=120000)
        )


        contents = [prompt]

        if image_part is not None:
            contents.append(image_part)

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )


        raw_text = (response.text or "").strip()

        # Remove Markdown fences if the model includes them.
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        data = json.loads(raw_text)

        if not isinstance(data, dict):
            raise ValueError("Model returned an invalid JSON object.")

        questions = data.get("questions")

        if not isinstance(questions, list) or len(questions) != count:
            raise ValueError("Model returned an incorrect number of questions.")

        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                raise ValueError("Invalid question format.")

            required = {
                "id", "question", "options",
                "correct_option", "explanation"
            }

            if set(question.keys()) != required:
                raise ValueError("Question fields do not match the contract.")

            if question["id"] != f"q{index}":
                raise ValueError("Invalid question ID.")

            if not isinstance(question["options"], dict):
                raise ValueError("Invalid answer options.")

            if set(question["options"].keys()) != {"A", "B", "C", "D"}:
                raise ValueError("Each question needs four options.")

            if question["correct_option"] not in ("A", "B", "C", "D"):
                raise ValueError("Invalid correct option.")

            for field in ("question", "explanation"):
                if not isinstance(question[field], str) or not question[field].strip():
                    raise ValueError(f"Invalid {field}.")

            for option in question["options"].values():
                if not isinstance(option, str) or not option.strip():
                    raise ValueError("Empty answer option.")

        return {
            "success": True,
            "data": data,
            "error": None
        }


    except Exception as error:
        error_code = getattr(error, "code", None)
        print("MCQ API error type:", type(error).__name__)
        print("MCQ API error code:", error_code)

        return {
            "success": False,
            "data": None,
            "error": f"MCQ generation failed: {type(error).__name__}"
        }

def chat_with_ai(
    message: str,
    study_text: str,
    history: list[dict],
    image_bytes: bytes | None = None
) -> dict:
    """Friendly AI tutor powered by Gemma 4."""

    if not isinstance(message, str) or not message.strip():
        return {
            "success": False,
            "data": None,
            "error": "Please enter a message."
        }

    if not isinstance(study_text, str):
        return {
            "success": False,
            "data": None,
            "error": "Study material must be text."
        }

    if not isinstance(history, list):
        return {
            "success": False,
            "data": None,
            "error": "Chat history must be a list."
        }

    if image_bytes is not None:
        return {
            "success": False,
            "data": None,
            "error": "Image support will be added later."
        }

    if not API_KEY:
        return {
            "success": False,
            "data": None,
            "error": "Gemini API key is missing."
        }

    # Validate and keep only the latest 10 messages.
    recent_history = history[-10:]

    for item in recent_history:
        if (
            not isinstance(item, dict)
            or item.get("role") not in ("user", "assistant")
            or not isinstance(item.get("content"), str)
        ):
            return {
                "success": False,
                "data": None,
                "error": "Invalid chat history format."
            }

    context = study_text[:15000].strip()

    instructions = """
You are a friendly AI Lab Viva Practice Partner.

Explain concepts in simple, accurate language.
Use relatable everyday examples where helpful.
Be supportive without being excessively informal.
Prioritize the student's study material.
If information is not in the material, clearly say
you are using general knowledge.
Never invent facts or pretend the notes contain
information they do not contain.
Keep answers concise unless the student requests detail.
"""

    conversation = [
        f"Previous {item['role']}: {item['content'][:2000]}"
        for item in recent_history
    ]

    prompt = f"""
{instructions}

STUDENT STUDY MATERIAL:
{context if context else "No study material provided."}

RECENT CHAT:
{chr(10).join(conversation)}

CURRENT STUDENT QUESTION:
{message[:4000]}

Respond directly and helpfully.
"""

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=120000)
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.6
            )
        )

        reply = (response.text or "").strip()

        if not reply:
            raise ValueError("Gemma returned an empty reply.")

        return {
            "success": True,
            "data": {"reply": reply},
            "error": None
        }

    except Exception as error:
        return {
            "success": False,
            "data": None,
            "error": f"Chat failed: {type(error).__name__}"
        }
    
def generate_written_questions(
    study_text: str,
    count: int = 3,
    image_bytes: bytes | None = None
) -> dict:
    """Generate written viva questions using Gemma 4."""

    if not isinstance(study_text, str) or not study_text.strip():
        return {
            "success": False,
            "data": None,
            "error": "Please provide study material."
        }

    if type(count) is not int or not 1 <= count <= 10:
        return {
            "success": False,
            "data": None,
            "error": "Question count must be between 1 and 10."
        }

    if image_bytes is not None:
        return {
            "success": False,
            "data": None,
            "error": "Image support will be added later."
        }

    if not API_KEY:
        return {
            "success": False,
            "data": None,
            "error": "Gemini API key is missing."
        }

    prompt = f"""
You are a friendly and accurate lab viva examiner.

Generate exactly {count} written-answer questions
from the study material below.

STUDY MATERIAL:
{study_text[:20000]}

Return ONLY a valid JSON object with this structure:

{{
  "questions": [
    {{
      "id": "w1",
      "question": "A clear viva question",
      "reference_answer": "An accurate beginner-friendly answer"
    }}
  ]
}}

Rules:
- Generate exactly {count} questions.
- Use IDs w1, w2, w3, etc. in order.
- Base answers on the study material only.
- Make questions suitable for students.
- Do not invent facts.
- Return JSON only, without Markdown.
"""

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=120000)
        )


        contents = [prompt]

        if image_part is not None:
            contents.append(image_part)

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.3
            )
        )


        raw_text = (response.text or "").strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        data = json.loads(raw_text)

        if not isinstance(data, dict):
            raise ValueError("Invalid JSON response.")

        questions = data.get("questions")

        if not isinstance(questions, list) or len(questions) != count:
            raise ValueError("Incorrect question count.")

        for index, question in enumerate(questions, start=1):
            if not isinstance(question, dict):
                raise ValueError("Invalid question format.")

            if set(question.keys()) != {
                "id", "question", "reference_answer"
            }:
                raise ValueError("Question fields do not match contract.")

            if question["id"] != f"w{index}":
                raise ValueError("Invalid question ID.")

            for key in ("question", "reference_answer"):
                if (
                    not isinstance(question[key], str)
                    or not question[key].strip()
                ):
                    raise ValueError(f"Invalid {key}.")

        return {
            "success": True,
            "data": data,
            "error": None
        }


    except Exception as error:
        error_code = getattr(error, "code", None)

        print("Written question API error type:", type(error).__name__)
        print("Written question API error code:", error_code)

        return {
            "success": False,
            "data": None,
            "error": f"Written question generation failed: {type(error).__name__}"
        }

def evaluate_answer(
    question: str,
    student_answer: str,
    reference_answer: str,
    study_text: str
) -> dict:
    """Evaluate a student's written viva answer."""

    inputs = {
        "question": question,
        "student_answer": student_answer,
        "reference_answer": reference_answer,
        "study_text": study_text
    }

    for name, value in inputs.items():
        if not isinstance(value, str):
            return {
                "success": False,
                "data": None,
                "error": f"{name} must be text."
            }

    for name in ("question", "student_answer", "reference_answer"):
        if not inputs[name].strip():
            return {
                "success": False,
                "data": None,
                "error": f"Please provide {name.replace('_', ' ')}."
            }

    if not API_KEY:
        return {
            "success": False,
            "data": None,
            "error": "Gemini API key is missing."
        }

    prompt = f"""
You are a friendly, fair lab viva examiner.

Evaluate the student's answer for scientific accuracy
and understanding rather than exact wording.

QUESTION:
{question[:3000]}

REFERENCE ANSWER:
{reference_answer[:5000]}

STUDY MATERIAL:
{study_text[:12000]}

STUDENT ANSWER:
{student_answer[:5000]}

Return ONLY a valid JSON object in this format:

{{
  "verdict": "partially_correct",
  "feedback": "Friendly and specific feedback",
  "ideal_answer": "Clear beginner-friendly correct answer"
}}

Rules:
- verdict must be exactly one of:
  correct, partially_correct, incorrect
- Use correct when the essential meaning is accurate.
- Use partially_correct for incomplete or partly accurate answers.
- Use incorrect when the main idea is wrong.
- Explain missing or incorrect concepts respectfully.
- Never invent mistakes.
- Use the reference answer and study material as primary evidence.
- Return JSON only, without Markdown.
"""

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=120000)
        )

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2
            )
        )

        raw_text = (response.text or "").strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        data = json.loads(raw_text)

        if not isinstance(data, dict):
            raise ValueError("Invalid evaluation JSON.")

        if set(data.keys()) != {
            "verdict", "feedback", "ideal_answer"
        }:
            raise ValueError("Evaluation fields do not match contract.")

        if data["verdict"] not in {
            "correct", "partially_correct", "incorrect"
        }:
            raise ValueError("Invalid evaluation verdict.")

        for key in ("feedback", "ideal_answer"):
            if (
                not isinstance(data[key], str)
                or not data[key].strip()
            ):
                raise ValueError(f"Invalid {key}.")

        return {
            "success": True,
            "data": data,
            "error": None
        }

    except Exception as error:
        return {
            "success": False,
            "data": None,
            "error": f"Answer evaluation failed: {type(error).__name__}"
        }
    
def generate_revision(
    study_text: str,
    image_bytes: bytes | None = None
) -> dict:
    """Generate revision flashcards and notes with Gemma 4."""

    if not isinstance(study_text, str) or not study_text.strip():
        return {
            "success": False,
            "data": None,
            "error": "Please provide study material."
        }

    if image_bytes is not None:
        return {
            "success": False,
            "data": None,
            "error": "Image support will be added later."
        }

    if not API_KEY:
        return {
            "success": False,
            "data": None,
            "error": "Gemini API key is missing."
        }

    prompt = f"""
You are a helpful revision assistant for students.

Create exactly 5 flashcards and 5 short revision notes
based on the study material below.

STUDY MATERIAL:
{study_text[:20000]}

Return ONLY valid JSON in this exact format:

{{
  "flashcards": [
    {{
      "front": "Question or concept",
      "back": "Short clear answer"
    }}
  ],
  "notes": [
    "Short revision point",
    "Another short revision point"
  ]
}}

Rules:
- Generate exactly 5 flashcards.
- Generate exactly 5 short notes.
- Make content beginner-friendly and accurate.
- Use the study material as the primary reference.
- Do not invent unsupported facts.
- Keep flashcard answers concise.
- Return JSON only, without Markdown.
"""

    try:
        client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=120000)
        )


        for attempt in range(3):
            try:
                print(f"Revision API attempt {attempt + 1}/3")

                response = client.models.generate_content(
                    model=MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3
                    )
                )

                break

            except Exception as api_error:
                error_code = getattr(api_error, "code", None)

                if error_code not in (500, 502, 503, 504):
                    raise

                if attempt == 2:
                    raise

                print(
                    f"Temporary server error {error_code}. Retrying..."
                )
                time.sleep(2 * (attempt + 1))


        raw_text = (response.text or "").strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        data = json.loads(raw_text)

        if not isinstance(data, dict):
            raise ValueError("Invalid revision JSON.")

        if set(data.keys()) != {"flashcards", "notes"}:
            raise ValueError("Revision fields do not match contract.")

        flashcards = data["flashcards"]
        notes = data["notes"]

        if not isinstance(flashcards, list) or len(flashcards) != 5:
            raise ValueError("Expected exactly 5 flashcards.")

        if not isinstance(notes, list) or len(notes) != 5:
            raise ValueError("Expected exactly 5 revision notes.")

        for card in flashcards:
            if not isinstance(card, dict):
                raise ValueError("Invalid flashcard.")

            if set(card.keys()) != {"front", "back"}:
                raise ValueError("Invalid flashcard fields.")

            for key in ("front", "back"):
                if not isinstance(card[key], str) or not card[key].strip():
                    raise ValueError("Flashcard contains empty text.")

        for note in notes:
            if not isinstance(note, str) or not note.strip():
                raise ValueError("Invalid revision note.")

        return {
            "success": True,
            "data": data,
            "error": None
        }

    
    except Exception as error:
        error_code = getattr(error, "code", None)

        print("Revision API error type:", type(error).__name__)
        print("Revision API error code:", error_code)

        return {
            "success": False,
            "data": None,
            "error": f"Revision generation failed: {type(error).__name__}"
        }
    
def _prepare_image(image_bytes: bytes):
    """Validate raw image bytes and prepare them for Gemma 4."""

    if not isinstance(image_bytes, bytes):
        raise ValueError("Image must be provided as raw bytes.")

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("Image must be smaller than 10 MB.")

    # Identify image format from its file signature.
    if image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"

    elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"

    elif (
        image_bytes.startswith(b"RIFF")
        and image_bytes[8:12] == b"WEBP"
    ):
        mime_type = "image/webp"

    else:
        raise ValueError(
            "Unsupported image. Please upload JPEG, PNG, or WebP."
        )

    return types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )




