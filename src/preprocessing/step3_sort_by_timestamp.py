import json
import os



def resort(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)



    data["keystrokes"]=sorted(data["keystrokes"], key=lambda x: x["timestamp"], reverse=False)


    # Open the file in write mode ('w') and use json.dump() to write the data
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(data, f,ensure_ascii=False, indent=2)

def check_time(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    for i in range(len(data["keystrokes"])-1):
        if data["keystrokes"][i]["timestamp"]>data["keystrokes"][i+1]["timestamp"]:
            print(file_path+str(i))

folder_path = "keystroke_sessions_only"  # thư mục gốc chứa nhiều folder con

# os.walk duyệt tất cả thư mục con
for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)
            resort(file_path)