import os
import json
import shutil

def merge_json_files_in_folder(folder_path):
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not json_files:
        return None

    merged_keystrokes = []

    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, dict) and "keystrokes" in data and isinstance(data["keystrokes"], list):
                    merged_keystrokes.extend(data["keystrokes"])
                else:
                    print(f"⚠️ Skipping {file_path}: missing or invalid 'keystrokes' key")
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"⚠️ Skipping invalid or unreadable file: {file_path}")

    if not merged_keystrokes:
        return None

    return {"keystrokes": merged_keystrokes}

def traverse_and_merge(root_dir):
    for user_folder in os.listdir(root_dir):
        user_path = os.path.join(root_dir, user_folder)
        if not os.path.isdir(user_path):
            continue

        for subfolder in os.listdir(user_path):
            sub_path = os.path.join(user_path, subfolder)
            if not os.path.isdir(sub_path):
                continue

            merged_data = merge_json_files_in_folder(sub_path)
            if merged_data is not None:
                output_path = os.path.join(user_path, f"{subfolder}.json")
                with open(output_path, "w", encoding="utf-8") as out_file:
                    json.dump(merged_data, out_file, ensure_ascii=False, indent=2)
                print(f"✅ Created {output_path}")
                shutil.rmtree(sub_path)

# Example usage
root_directory = "keystroke_sessions_only"
traverse_and_merge(root_directory)