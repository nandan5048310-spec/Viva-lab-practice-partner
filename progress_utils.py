
"""Track student attempts and calculate topic performance."""

SCORES = {
    "correct": 1.0,
    "partially_correct": 0.5,
    "incorrect": 0.0,
}


def record_attempt(attempts, question_id, topic,
                   question_type, verdict, attempt_id,
                   document_id):
    """Record a submission once per document and attempt ID."""

    if verdict not in SCORES:
        raise ValueError("Invalid verdict.")

    if question_type not in {"mcq", "written", "viva"}:
        raise ValueError("Invalid question type.")

    if not document_id or not attempt_id or not question_id:
        raise ValueError("Document, attempt and question IDs are required.")

    # Avoid duplicate records when Streamlit reruns.
    if any(
        a["document_id"] == document_id
        and a["attempt_id"] == attempt_id
        for a in attempts
    ):
        return False

    attempts.append({
        "document_id": document_id,
        "attempt_id": attempt_id,
        "question_id": question_id,
        "topic": topic.strip() if isinstance(topic, str)
                 and topic.strip() else "Unknown",
        "question_type": question_type,
        "verdict": verdict,
        "score": SCORES[verdict],
    })

    return True


def calculate_progress(attempts, document_id):
    """Calculate progress for one study document."""

    selected = [
        a for a in attempts
        if a["document_id"] == document_id
    ]

    summary = {
        "total": len(selected),
        "correct": 0,
        "partially_correct": 0,
        "incorrect": 0,
        "overall_percentage": 0.0,
        "topics": [],
    }

    if not selected:
        return summary

    topic_groups = {}

    for attempt in selected:
        summary[attempt["verdict"]] += 1

        topic = attempt["topic"]
        topic_groups.setdefault(topic, []).append(
            attempt["score"]
        )

    summary["overall_percentage"] = round(
        100 * sum(a["score"] for a in selected) / len(selected),
        1,
    )

    for topic, scores in topic_groups.items():
        percentage = round(
            100 * sum(scores) / len(scores), 1
        )

        if len(scores) < 3:
            status = "Limited Evidence"
        elif percentage >= 80:
            status = "Strong"
        elif percentage >= 50:
            status = "Needs Practice"
        else:
            status = "Weak"

        summary["topics"].append({
            "topic": topic,
            "attempted": len(scores),
            "percentage": percentage,
            "status": status,
        })

    summary["topics"].sort(
        key=lambda item: (item["percentage"], item["topic"])
    )

    return summary
