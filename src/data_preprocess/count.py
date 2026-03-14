import json
import os
def search2(data):
        if isinstance(data, dict):
            return any(search2(v) for v in data.values())
        elif isinstance(data, list):
            return any(search2(item) for item in data)
        elif isinstance(data, str):
            return "·" in data
        return False


def count(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)


    down=0
    up=0
    for i in range(len(data["keystrokes"])):
        
        if data["keystrokes"][i]["event"]=="keydown":
            down+=1
        if data["keystrokes"][i]["event"]=="keyup":
            up+=1
    print(file_path+" Down: "+str(down)+" Up: "+str(up))
    print(search2(data))
    print(down-up)



folder_path = "../../dataset/viet_preprocessed" 
 # thư mục gốc chứa nhiều folder con
#count("preptest.json")
# os.walk duyệt tất cả thư mục con

for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)
            count(file_path)

