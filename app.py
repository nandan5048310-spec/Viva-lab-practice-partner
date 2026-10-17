
import streamlit as st
from utils import extract_text

st.set_page_config(
    page_title="AI Lab Viva Practice Partner",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 AI Lab Viva Practice Partner")
st.write("Your friendly study buddy for lab viva preparation.")

# -------------------------------
# Study Material
# -------------------------------

st.header("📚 Your Study Material")

if "study_material" not in st.session_state:
    st.session_state.study_material = ""

if "uploaded_study_text" not in st.session_state:
    st.session_state.uploaded_study_text = ""

study_text = st.text_area(
    "Paste your notes or practical procedure here:",
    value=st.session_state.study_material,
    placeholder="Paste your lab notes, textbook content, or practical procedure...",
    height=200,
    key="study_text_input",
)

st.session_state.study_material = study_text

if study_text.strip():
    st.success(f"Study material entered: {len(study_text)} characters.")

st.subheader("📎 Or Upload Study Material")

uploaded_file = st.file_uploader(
    "Choose a PDF, DOCX, TXT, or image file",
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
    key="study_material_upload",
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = file_name.rsplit(".", 1)[-1].lower()
    file_bytes = uploaded_file.getvalue()

    st.success(f"Uploaded: {file_name}")

    if file_extension in ["png", "jpg", "jpeg"]:
        st.subheader("🖼️ Uploaded Image")
        st.image(file_bytes, caption=file_name)

        st.info(
            "The image is uploaded and displayed. "
            "AI image understanding will be connected separately."
        )

    else:
        try:
            material_text = extract_text(file_name, file_bytes)

            if material_text and material_text.strip():
                st.subheader("📖 Extracted Study Material")

                st.text_area(
                    "Review your extracted text:",
                    value=material_text,
                    height=250,
                    key="extracted_material_preview",
                )

                if st.button("Use Uploaded Text as Study Material"):
                    st.session_state.uploaded_study_text = material_text
                    st.session_state.study_material = material_text
                    st.session_state.pop("study_text_input", None)
                    st.rerun()

            else:
                st.warning(
                    "No readable text was found. "
                    "The PDF may contain scanned pages or images."
                )

        except Exception as error:
            st.error(f"Could not read this file: {error}")

# -------------------------------
# MCQ Quiz
# -------------------------------

st.divider()
st.header("📝 MCQ Quiz")

sample_question = {
    "question": "Which Python function displays text on the screen?",
    "options": {
        "A": "input()",
        "B": "print()",
        "C": "type()",
        "D": "len()",
    },
    "correct_option": "B",
    "explanation": (
        "The print() function displays text or other values on the screen."
    ),
}

if "mcq_submitted" not in st.session_state:
    st.session_state.mcq_submitted = False

if "mcq_score" not in st.session_state:
    st.session_state.mcq_score = 0

st.write(sample_question["question"])

selected_option = st.radio(
    "Choose your answer:",
    options=list(sample_question["options"].keys()),
    format_func=lambda key: (
        f"{key}. {sample_question['options'][key]}"
    ),
    key="sample_mcq_answer",
)

if st.button("Submit MCQ Answer", key="submit_sample_mcq"):
    st.session_state.mcq_submitted = True
    st.session_state.mcq_score = int(
        selected_option == sample_question["correct_option"]
    )

if st.session_state.mcq_submitted:
    if selected_option == sample_question["correct_option"]:
        st.success("Correct! " + sample_question["explanation"])
    else:
        st.error(
            f"Correct answer: {sample_question['correct_option']}. "
            + sample_question["explanation"]
        )

    st.write(f"Score: {st.session_state.mcq_score} / 1")

if st.button("Try MCQ Again"):
    st.session_state.mcq_submitted = False
    st.session_state.mcq_score = 0
    st.rerun()

# -------------------------------
# Written Answer Practice
# -------------------------------

st.divider()
st.header("✍️ Written Answer Practice")

st.write("Practice answering lab viva questions in your own words.")

written_question = (
    "What is the purpose of the print() function in Python?"
)

st.subheader("Question")
st.write(written_question)

written_answer = st.text_area(
    "Type your answer here:",
    placeholder="Explain your answer in your own words...",
    height=150,
    key="written_answer",
)

if st.button("Submit Written Answer"):
    if written_answer.strip():
        if "print" in written_answer.lower():
            st.success("Good start! Your answer mentions print().")
            st.write(
                "**Sample answer:** The print() function displays "
                "text or other values on the screen."
            )
        else:
            st.info(
                "Keep trying! Explain which Python function displays "
                "text or values on the screen."
            )
    else:
        st.warning("Please write an answer before submitting.")

# -------------------------------
# Friendly Study Chat Placeholder
# -------------------------------

st.divider()
st.header("💬 Study Chat")

st.write(
    "The friendly AI tutor will answer questions about your study "
    "material once the AI backend is connected."
)

chat_message = st.text_input(
    "Ask a study question:",
    placeholder="For example: Explain this concept simply...",
    key="chat_message",
)

if st.button("Send Question"):
    if chat_message.strip():
        st.info(
            "Your question is ready, but the AI chat backend "
            "has not been connected yet."
        )
    else:
        st.warning("Please enter a question first.")

# -------------------------------
# Revision Notes Placeholder
# -------------------------------

st.divider()
st.header("⚡ Quick Revision")

st.write(
    "AI-generated revision notes will be available after "
    "the backend is connected."
)
