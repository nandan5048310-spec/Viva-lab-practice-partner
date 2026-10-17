

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

st.subheader("📎 Or upload study material")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    st.success(f"Selected file: {uploaded_file.name}")
    st.caption(f"File size: {uploaded_file.size:,} bytes")

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
