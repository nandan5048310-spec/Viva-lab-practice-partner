
"""Voice utilities for the AI Lab Viva Practice Partner."""

import asyncio
import io
import re
import threading

import edge_tts
import speech_recognition as sr


VOICES = {
    "Neerja": "en-IN-NeerjaNeural",
    "Prabhat": "en-IN-PrabhatNeural",
}


def _clean_for_speech(text: str) -> str:
    """Remove Markdown formatting before speaking."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"[`#*_]", "", text)
    return " ".join(text.split())


async def _generate_speech(text: str, voice: str) -> bytes:
    communicator = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate="-8%",
        pitch="+0Hz",
    )

    audio = bytearray()

    async for chunk in communicator.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    if not audio:
        raise ValueError("The voice service returned no audio.")

    return bytes(audio)


def _run_async(coroutine):
    """Run speech generation without conflicting with an existing event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    result = {}
    errors = {}

    def runner():
        try:
            result["value"] = asyncio.run(coroutine)
        except Exception as error:
            errors["error"] = error

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if "error" in errors:
        raise errors["error"]

    return result["value"]


def text_to_speech(
    text: str,
    voice: str = "en-IN-NeerjaNeural",
) -> dict:
    """Return MP3 audio bytes in the shared success/data/error format."""
    if not isinstance(text, str) or not text.strip():
        return {
            "success": False,
            "data": None,
            "error": "Please provide text to speak.",
        }

    if voice not in VOICES.values():
        return {
            "success": False,
            "data": None,
            "error": "Please select a supported Indian English voice.",
        }

    try:
        cleaned = _clean_for_speech(text)

        audio_bytes = _run_async(_generate_speech(cleaned, voice))

        return {
            "success": True,
            "data": {
                "audio_bytes": audio_bytes,
                "format": "audio/mp3",
            },
            "error": None,
        }

    except Exception:
        return {
            "success": False,
            "data": None,
            "error": "Voice generation failed. You can still read the text.",
        }


def speech_to_text(audio_bytes: bytes) -> dict:
    """Transcribe recorded WAV audio using Indian English recognition."""
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        return {
            "success": False,
            "data": None,
            "error": "Please record an answer first.",
        }

    try:
        recognizer = sr.Recognizer()

        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)

        transcript = recognizer.recognize_google(audio, language="en-IN")

        if not transcript.strip():
            raise sr.UnknownValueError()

        return {
            "success": True,
            "data": {"transcript": transcript},
            "error": None,
        }

    except sr.UnknownValueError:
        message = "Speech was unclear. Please try again or type your answer."
    except sr.RequestError:
        message = "Speech recognition is unavailable. Please type your answer."
    except (ValueError, EOFError, OSError):
        message = "Invalid recording. Please record again."
    except Exception:
        message = "Could not transcribe the recording. Please type your answer."

    return {
        "success": False,
        "data": None,
        "error": message,
    }
