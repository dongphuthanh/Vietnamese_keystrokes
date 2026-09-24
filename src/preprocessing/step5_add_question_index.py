import os
def add_session_and_question_index(file_path: str) -> str:
    """
    Reads a JSON keystroke log file, identifies unique questions, assigns session and question_index,
    and saves the updated data to a new file.

    :param file_path: Path to the input JSON file.
    :param output_path: Path where the updated JSON file will be saved.
    :return: The output file path.
    """
    import json
    from pathlib import Path

    # Load the JSON file
    with Path(file_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract unique questions in the order they appear
    unique_questions = []
    seen = set()
    for entry in data["keystrokes"]:
        q = entry["question"]
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    # Build mapping: question text -> (session number, question index)
    question_info = {}
    for idx, question in enumerate(unique_questions):
        session_num = idx // 6 + 1
        question_num = idx % 6 + 1
        question_info[question] = {
            "session": session_num,
            "question_index": f"{session_num}.{question_num}"
        }

    # Apply mapping to all entries
    for entry in data["keystrokes"]:
        q = entry["question"]
        entry["session"] = question_info[q]["session"]
        entry["question_index"] = question_info[q]["question_index"]

    # Save the updated file
    with Path((file_path)).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

folder_path = "keystroke_sessions_only"  # thư mục gốc chứa nhiều folder con

# os.walk duyệt tất cả thư mục con
for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)
            add_session_and_question_index(file_path)