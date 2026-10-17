"""AI backend for the Lab Viva Practice Partner.

Only google-genai and python-dotenv are required. The five public function
signatures and their {success, data, error} return contract are kept stable.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")


def _success(data: dict) -> dict:
    return {"success": True, "data": data, "error": None}


def _failure(message: str) -> dict:
    return {"success": False, "data": None, "error": message}


def _error_message(task: str, error: Exception) -> str:
    """Display useful errors without exposing secrets or long tracebacks."""
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(error, "status_code", None)

    if code in (500, 502, 503, 504):
        return f"{task}: Gemma 4's server is temporarily unavailable (HTTP {code}). Please try again."
    if code == 429:
        return f"{task}: API usage limit reached (HTTP 429). Please try again later."
    if code in (401, 403):
        return f"{task}: API access denied (HTTP {code}). Check your API key and model access."
    if code == 400:
        return f"{task}: The AI service rejected this request (HTTP 400). Check the input."
    if isinstance(error, json.JSONDecodeError):
        return f"{task}: The model returned invalid JSON. Please retry."
    if isinstance(error, (ValueError, TypeError)):
        return f"{task}: {error}"
    return f"{task}: {type(error).__name__}. Please try again."


def _require_api_key() -> str | None:
    return None if API_KEY else "GEMINI_API_KEY is missing. Add it to your private .env file."


def _prepare_image(client, image_bytes: bytes):
    """Upload raw image bytes via the Files API; never depend on a user file path."""
    if not isinstance(image_bytes, bytes):
        raise ValueError("Image must be raw bytes.")
    if not image_bytes:
        raise ValueError("Image is empty.")
    if len(image_bytes) > 10 * 1024 * 1024:
        raise ValueError("Image must be 10 MB or smaller.")

    if image_bytes.startswith(b"\xff\xd8\xff"):
        suffix = ".jpg"
    elif image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        suffix = ".png"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        suffix = ".webp"
    else:
        raise ValueError("Unsupported image format. Use PNG, JPEG, or WebP.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        return client.files.upload(file=temp_path)
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def _ask_gemma(prompt: str, temperature: float, image_bytes: bytes | None = None) -> str:
    """Single place for model calls, optional images and temporary server retries."""
    client = genai.Client(
        api_key=API_KEY,
        http_options=types.HttpOptions(timeout=120000),
    )
    uploaded_file = None
    try:
        contents = [prompt]
        if image_bytes is not None:
            uploaded_file = _prepare_image(client, image_bytes)
            contents = [uploaded_file, prompt]

        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=temperature),
                )
                text = (response.text or "").strip()
                if not text:
                    raise ValueError("The model returned an empty response. Please retry.")
                return text
            except Exception as error:
                code = getattr(error, "code", None)
                if attempt == 1 or code not in (500, 502, 503, 504):
                    raise
                time.sleep(1.5)
    finally:
        # Uploaded files are only needed for this one request. Cleanup is best-effort.
        if uploaded_file is not None and getattr(uploaded_file, "name", None):
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass


def _parse_json(raw_text: str) -> dict:
    """Handle a JSON answer with optional Markdown fences or brief extra text."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return valid JSON. Please retry.") from None
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            raise ValueError("The model did not return valid JSON. Please retry.") from None
    if not isinstance(obj, dict):
        raise ValueError("The model did not return a JSON object.")
    return obj


def _nonempty_text(value, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {description} in the model response.")
    return value.strip()


def _material_ok(study_text: str, image_bytes: bytes | None) -> str | None:
    if not isinstance(study_text, str):
        return "Study material must be text."
    if not study_text.strip() and image_bytes is None:
        return "Please provide study text or an image."
    return None


def _material_prompt(study_text: str) -> str:
    return study_text[:20000].strip() or "No typed study text. Use the attached study image."


def generate_mcqs(
    study_text: str, count: int = 5, image_bytes: bytes | None = None
) -> dict:
    """Generate multiple-choice questions from study text and/or an image."""
    problem = _material_ok(study_text, image_bytes)
    if problem:
        return _failure(problem)
    if type(count) is not int or not 1 <= count <= 20:
        return _failure("Question count must be between 1 and 20.")
    if _require_api_key():
        return _failure(_require_api_key())

    prompt = f"""
You are a friendly, accurate lab viva quiz teacher.
Generate exactly {count} multiple-choice questions based ONLY on the STUDY MATERIAL
and, if supplied, the attached image. Read visible text in the image carefully.

STUDY MATERIAL:
{_material_prompt(study_text)}

Return ONLY a JSON object with this structure:
{{"questions":[{{"id":"q1","question":"Question text","options":{{"A":"Option 1","B":"Option 2","C":"Option 3","D":"Option 4"}},"correct_option":"A","explanation":"Beginner-friendly explanation"}}]}}
Rules: Exactly {count} questions. Four options A/B/C/D and one correct answer each.
Use q1, q2, ... in order. Do not invent unsupported facts. No Markdown.
"""
    try:
        obj = _parse_json(_ask_gemma(prompt, 0.3, image_bytes))
        questions = obj.get("questions")
        if not isinstance(questions, list) or len(questions) != count:
            raise ValueError(f"Expected exactly {count} MCQs.")
        result = []
        for index, q in enumerate(questions, start=1):
            if not isinstance(q, dict):
                raise ValueError("Invalid MCQ format.")
            options = q.get("options")
            if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
                raise ValueError("Each MCQ needs options A, B, C, and D.")
            answer = _nonempty_text(q.get("correct_option"), "correct_option").upper()
            if answer not in options:
                raise ValueError("Invalid correct_option in MCQ.")
            result.append({
                "id": f"q{index}",
                "question": _nonempty_text(q.get("question"), "question"),
                "options": {k: _nonempty_text(options[k], f"option {k}") for k in "ABCD"},
                "correct_option": answer,
                "explanation": _nonempty_text(q.get("explanation"), "explanation"),
            })
        return _success({"questions": result})
    except Exception as error:
        return _failure(_error_message("MCQ generation failed", error))


def generate_written_questions(
    study_text: str, count: int = 3, image_bytes: bytes | None = None
) -> dict:
    """Generate written-answer viva questions with reference answers."""
    problem = _material_ok(study_text, image_bytes)
    if problem:
        return _failure(problem)
    if type(count) is not int or not 1 <= count <= 10:
        return _failure("Question count must be between 1 and 10.")
    if _require_api_key():
        return _failure(_require_api_key())

    prompt = f"""
You are a friendly, accurate lab viva examiner.
Generate exactly {count} written-answer questions based ONLY on this study material
and any attached study image.

STUDY MATERIAL:
{_material_prompt(study_text)}

Return ONLY valid JSON:
{{"questions":[{{"id":"w1","question":"Clear viva question","reference_answer":"Accurate beginner-friendly answer"}}]}}
Rules: Exactly {count} questions. Use w1, w2, ... in order.
Do not invent facts. No Markdown.
"""
    try:
        obj = _parse_json(_ask_gemma(prompt, 0.3, image_bytes))
        questions = obj.get("questions")
        if not isinstance(questions, list) or len(questions) != count:
            raise ValueError(f"Expected exactly {count} written questions.")
        result = []
        for index, q in enumerate(questions, start=1):
            if not isinstance(q, dict):
                raise ValueError("Invalid written-question format.")
            result.append({
                "id": f"w{index}",
                "question": _nonempty_text(q.get("question"), "question"),
                "reference_answer": _nonempty_text(q.get("reference_answer"), "reference_answer"),
            })
        return _success({"questions": result})
    except Exception as error:
        return _failure(_error_message("Written-question generation failed", error))


def evaluate_answer(
    question: str, student_answer: str, reference_answer: str, study_text: str
) -> dict:
    """Evaluate written answers by meaning, not exact wording."""
    values = {
        "question": question,
        "student_answer": student_answer,
        "reference_answer": reference_answer,
        "study_text": study_text,
    }
    for name, value in values.items():
        if not isinstance(value, str):
            return _failure(f"{name.replace('_', ' ').capitalize()} must be text.")
    for name in ("question", "student_answer", "reference_answer"):
        if not values[name].strip():
            return _failure(f"Please provide {name.replace('_', ' ')}.")
    if _require_api_key():
        return _failure(_require_api_key())

    prompt = f"""
You are a friendly and fair lab viva examiner. Evaluate by scientific accuracy
and understanding, not by exact wording.

QUESTION: {question[:3000]}
REFERENCE ANSWER: {reference_answer[:5000]}
STUDY MATERIAL: {study_text[:12000]}
STUDENT ANSWER: {student_answer[:5000]}

Return ONLY valid JSON:
{{"verdict":"partially_correct","feedback":"Specific supportive feedback","ideal_answer":"Clear correct answer"}}
The verdict MUST be one of: correct, partially_correct, incorrect.
Do not invent mistakes or unsupported facts. No Markdown.
"""
    try:
        obj = _parse_json(_ask_gemma(prompt, 0.2))
        verdict = _nonempty_text(obj.get("verdict"), "verdict").lower().replace(" ", "_")
        if verdict not in {"correct", "partially_correct", "incorrect"}:
            raise ValueError("Invalid answer-evaluation verdict.")
        return _success({
            "verdict": verdict,
            "feedback": _nonempty_text(obj.get("feedback"), "feedback"),
            "ideal_answer": _nonempty_text(obj.get("ideal_answer"), "ideal_answer"),
        })
    except Exception as error:
        return _failure(_error_message("Answer evaluation failed", error))


def chat_with_ai(
    message: str, study_text: str, history: list[dict], image_bytes: bytes | None = None
) -> dict:
    """Friendly study chatbot that accepts conversation history."""
    if not isinstance(message, str) or (not message.strip() and image_bytes is None):
        return _failure("Please enter a message or attach an image.")
    if not isinstance(study_text, str):
        return _failure("Study material must be text.")
    if not isinstance(history, list):
        return _failure("Chat history must be a list.")
    if _require_api_key():
        return _failure(_require_api_key())

    recent_history = history[-10:]
    for item in recent_history:
        if (not isinstance(item, dict)
                or item.get("role") not in ("user", "assistant")
                or not isinstance(item.get("content"), str)):
            return _failure("Invalid chat history format. Use {'role': 'user'|'assistant', 'content': '...'}.")

    history_text = "\n".join(
        f"Previous {item['role']}: {item['content'][:2000]}" for item in recent_history
    )
    prompt = f"""
You are a friendly AI Lab Viva Practice Partner.
Explain accurately in simple terms with relatable everyday examples.
Be supportive, concise, and honest about what the notes contain.
If material is missing, say when you are using general knowledge.

STUDENT STUDY MATERIAL:
{study_text[:15000] or 'No typed study notes.'}

RECENT CHAT:
{history_text or 'No earlier messages.'}

CURRENT STUDENT QUESTION:
{message[:4000] or 'Explain the attached image.'}

If there is an attached image, examine it before answering.
Respond directly as a friendly tutor.
"""
    try:
        reply = _ask_gemma(prompt, 0.6, image_bytes)
        return _success({"reply": _nonempty_text(reply, "reply")})
    except Exception as error:
        return _failure(_error_message("Chat failed", error))


def generate_revision(study_text: str, image_bytes: bytes | None = None) -> dict:
    """Generate five flashcards and five concise revision notes."""
    problem = _material_ok(study_text, image_bytes)
    if problem:
        return _failure(problem)
    if _require_api_key():
        return _failure(_require_api_key())

    prompt = f"""
You are a helpful revision assistant for students.
Create exactly 5 flashcards and exactly 5 short revision notes from the
study material and any attached image.

STUDY MATERIAL:
{_material_prompt(study_text)}

Return ONLY valid JSON:
{{"flashcards":[{{"front":"Question or concept","back":"Short answer"}}],"notes":["Short revision point"]}}
Rules: Exactly five flashcards with front and back; exactly five short string notes.
Be beginner-friendly and accurate. Do not invent unsupported facts. No Markdown.
"""
    try:
        obj = _parse_json(_ask_gemma(prompt, 0.3, image_bytes))
        flashcards, notes = obj.get("flashcards"), obj.get("notes")
        if not isinstance(flashcards, list) or len(flashcards) != 5:
            raise ValueError("Expected exactly five flashcards.")
        if not isinstance(notes, list) or len(notes) != 5:
            raise ValueError("Expected exactly five revision notes.")
        cards = []
        for card in flashcards:
            if not isinstance(card, dict):
                raise ValueError("Invalid flashcard format.")
            cards.append({
                "front": _nonempty_text(card.get("front"), "flashcard front"),
                "back": _nonempty_text(card.get("back"), "flashcard back"),
            })
        clean_notes = [_nonempty_text(note, "revision note") for note in notes]
        return _success({"flashcards": cards, "notes": clean_notes})
    except Exception as error:
        return _failure(_error_message("Revision generation failed", error))


def test_connection():
    """Quick optional text-only connectivity check (does not run on import)."""
    if _require_api_key():
        print("ERROR: GEMINI_API_KEY is missing from .env")
        return
    try:
        print("SUCCESS! Gemma 4 responded:")
        print(_ask_gemma("Explain what a capacitor is in one simple sentence.", 0.3))
    except Exception as error:
        print(_error_message("API connection failed", error))


if __name__ == "__main__":
    test_connection()
