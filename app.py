

import streamlit as st

st.set_page_config(
    page_title="AI Lab Viva Practice Partner",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 AI Lab Viva Practice Partner")
st.write("Your friendly study buddy for lab viva preparation.")

st.info("Paste your study material below to get started.")

study_text = st.text_area(
    "Your study material",
    placeholder="Paste your lab notes, textbook content, or practical procedure here...",
    height=250,
)

if study_text.strip():
    st.success(f"Study material entered: {len(study_text)} characters.")
else:
    st.caption("Your study material will appear here when you enter some text.")

st.divider()
st.header("📝 MCQ Quiz")

if "mcq_score" not in st.session_state:
    st.session_state.mcq_score = 0

if "mcq_submitted" not in st.session_state:
    st.session_state.mcq_submitted = False


sample_question = {
    "question": "Which Python function displays text on the screen?",
    "options": {
        "A": "input()",
        "B": "print()",
        "C": "type()",
        "D": "len()",
    },
    "correct_option": "B",
    "explanation": "The print() function displays text or other values on the screen.",
}

st.write(sample_question["question"])

selected_option = st.radio(
    "Choose your answer:",
    options=list(sample_question["options"].keys()),
    format_func=lambda key: f"{key}. {sample_question['options'][key]}",
    key="sample_mcq_answer",
)


if st.button("Submit Answer", key="submit_sample_mcq"):
    st.session_state.mcq_submitted = True
    st.session_state.mcq_score = int(
        selected_option == sample_question["correct_option"]
    )

if st.session_state.mcq_submitted:
    if selected_option == sample_question["correct_option"]:
        st.success("Correct! " + sample_question["explanation"])
    else:
        st.error(
            f"Not quite. The correct answer is "
            f"{sample_question['correct_option']}. "
            f"{sample_question['explanation']}"
        )

    st.write(f"Score: {st.session_state.mcq_score} / 1")


# -------------------------------
# Written Answer Practice
# -------------------------------

st.divider()
st.header("✍️ Written Answer Practice")

st.write(
    "Practice answering lab viva questions in your own words."
)

written_question = (
    "What is the purpose of the print() function in Python?"
)

st.subheader("Question")
st.write(written_question)

written_answer = st.text_area(
    "Type your answer here:",
    placeholder="Write your explanation in your own words...",
    height=150,
    key="written_answer"
)

if st.button("Submit Written Answer"):
    if written_answer.strip():
        if "print" in written_answer.lower():
            st.success("Good start! Your answer mentions print().")
            st.write(
                "**Sample answer:** "
                "The print() function displays text or other values "
                "on the screen."
            )
        else:
            st.info(
                "Keep trying! Explain which Python function displays "
                "text or values on the screen."
            )
    else:
        st.warning("Please write an answer before submitting.")


# -------------------------------
# Study Material Upload
# -------------------------------

st.divider()
st.header("📚 Upload Study Material")

st.write(
    "Upload your notes so you can use them for lab viva preparation."
)

uploaded_file = st.file_uploader(
    "Choose a study material file",
    type=["txt", "pdf", "docx", "png", "jpg", "jpeg"],
    key="study_material_upload"
)

if uploaded_file is not None:
    file_name = uploaded_file.name
    file_extension = file_name.rsplit(".", 1)[-1].lower()

    st.success(f"Uploaded: {file_name}")
    type=["txt", "pdf", "docx", "png", "jpg", "jpeg"],

    try:
        if file_extension in ["png", "jpg", "jpeg"]:
            st.subheader("🖼️ Uploaded Image")
            st.image(uploaded_file.getvalue(), caption=file_name)
            st.success(
                "Image uploaded successfully. "
                "It can be sent to the AI backend in a later step."
            )

        elif file_extension == "txt":
            material_text = uploaded_file.getvalue().decode("utf-8")

        elif file_extension == "pdf":
            from pypdf import PdfReader
            import io

            pdf_reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
            material_text = "\n".join(
                page.extract_text() or ""
                for page in pdf_reader.pages
            )

        elif file_extension == "docx":
            from docx import Document
            import io

            document = Document(io.BytesIO(uploaded_file.getvalue()))
            material_text = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        st.subheader("📖 Extracted Study Material")

        if material_text.strip():
            st.text_area(
                "Review your extracted text:",
                value=material_text,
                height=250,
                key="extracted_material_preview"
            )
            st.info(
                "Your material is ready for the next step: "
                "AI explanations and quiz generation."
            )
        else:
            st.warning(
                "No readable text was found. "
                "The PDF may contain scanned pages or images."
            )

    except Exception as error:
        st.error(f"Could not read this file: {error}")

