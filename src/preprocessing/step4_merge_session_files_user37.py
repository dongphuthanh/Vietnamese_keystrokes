import json
import os

folder_path = "user_37"

input_files = [
    "keystrokes_session1.json",
    "keystrokes_session2.json",
    "keystrokes_session3.json"
]

merged_data = {"keystrokes": []}

for file_name in input_files:
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, "r",encoding="utf8") as f:
        data = json.load(f)
        merged_data["keystrokes"].extend(data.get("keystrokes", []))

# Output file (saved inside the same folder)
output_path = os.path.join(folder_path, "first_time.json")

with open(output_path, "w",encoding="utf8") as f:
    json.dump(merged_data, f, indent=2)

print("Merged JSON saved as:", output_path)