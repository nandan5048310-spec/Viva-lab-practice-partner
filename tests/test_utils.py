
import pytest

from utils import extract_text


def test_extract_text_from_txt():
    content = b"Experiment: To measure the pH of a solution."
    result = extract_text("notes.txt", content)

    assert "Experiment" in result
    assert "pH" in result


def test_extract_text_strips_blank_lines():
    content = b"\n  Viva practice  \n\n  Chemistry  \n"
    result = extract_text("notes.txt", content)

    assert result == "Viva practice\nChemistry"


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text("notes.doc", b"Some content")


def test_extract_text_rejects_empty_file():
    with pytest.raises(ValueError, match="empty"):
        extract_text("notes.txt", b"")


def test_extract_text_rejects_invalid_txt_encoding():
    with pytest.raises(ValueError, match="Could not read"):
        extract_text("notes.txt", b"\xff\xfe\xfa")

