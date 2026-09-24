import json
import os
from telex_map import * # type: ignore


def next_unidentified(data,i):
    j=i+2
    while j<i+20:
        if j==len(data["keystrokes"]):
            break
        if data["keystrokes"][j]["key"]=="Unidentified":
            return j
        else:
            j+=1
    return i+10

def real_or_fake(data):
    new={"keystrokes":[]}
    a=["Backspace","Unidentified","·"]
    data["keystrokes"][0]["actual"]="real"
    data["keystrokes"][len(data["keystrokes"])-1]["actual"]="real"


    for i in range(1,len(data["keystrokes"])-1):
        if data["keystrokes"][i]["code"]=="" and len(data["keystrokes"][i]["key"])==1:
            data["keystrokes"][i]["actual"]="indicate"
        elif data["keystrokes"][i]["code"]=="" or data["keystrokes"][i]["code"]=="IntlYen" or data["keystrokes"][i]["code"]=="NonConvert":
            data["keystrokes"][i]["actual"]="fake"
        elif data["keystrokes"][i]["key"]=="Backspace":
            if data["keystrokes"][i]["event"]=="keydown" and data["keystrokes"][i+1]["timestamp"]-data["keystrokes"][i]["timestamp"]<30:
                if data["keystrokes"][i-1]["key"]!="Backspace":
                    data["keystrokes"][i]["actual"]="half"
                    for j in range(i+1,next_unidentified(data,i)):
                        j=min(j,len(data["keystrokes"])-1)
                        if data["keystrokes"][j]["key"]=="Backspace":
                            data["keystrokes"][j]["actual"]="fake"
                else:
                    data["keystrokes"][i]["actual"]="fake"
            elif data["keystrokes"][i]["event"]=="keyup":
                if data["keystrokes"][i]["timestamp"]-data["keystrokes"][i-1]["timestamp"]<15 or data["keystrokes"][i-1]["actual"]!="real":
                    data["keystrokes"][i]["actual"]="fake"
                else:
                    data["keystrokes"][i]["actual"]="real"
            else:
                data["keystrokes"][i]["actual"]="real"
        else:
            data["keystrokes"][i]["actual"]="real"
            

    for i in range(len(data["keystrokes"])):
        if data["keystrokes"][i]["actual"]!="fake":
            new["keystrokes"].append(data["keystrokes"][i])
    return new

def next_indicator(data,i):
    j=i+1
    while j<i+10:
        if j==len(data["keystrokes"]):
            break
        if data["keystrokes"][j]["actual"]=="indicate":
            return j
        else:
            j+=1
    return
def fix_dot(data):
    #already_match=[]
    #z=["w","j","r","s","x","o","a","e","f","d","D"]
    temp=None
    new={"keystrokes":[]}
    for i in range(len(data["keystrokes"])):
        if data["keystrokes"][i]["actual"]=="half":
            j=next_indicator(data,i)
            if j is None:
                continue
            remap=char_to_telex(data["keystrokes"][j]["key"],temp)
            data["keystrokes"][i]["key"]=remap
            data["keystrokes"][i]["code"]="Key"+remap.upper()
            temp=data["keystrokes"][j]["key"]
            
    for i in range(len(data["keystrokes"])):
        if data["keystrokes"][i]["actual"]!="indicate":
            new["keystrokes"].append(data["keystrokes"][i])

    return new

def search(data):
        if isinstance(data, dict):
            return any(search(v) for v in data.values())
        elif isinstance(data, list):
            return any(search(item) for item in data)
        elif isinstance(data, str):
            return "Unidentified" in data
        return False
def search2(data):
        if isinstance(data, dict):
            return any(search2(v) for v in data.values())
        elif isinstance(data, list):
            return any(search2(item) for item in data)
        elif isinstance(data, str):
            return "·" in data
        return False
def search3(data):
        if isinstance(data, dict):
            return any(search3(v) for v in data.values())
        elif isinstance(data, list):
            return any(search3(item) for item in data)
        elif isinstance(data, str):
            return "Process" in data
        return False
def fix_windows(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    if not (search(data) or search2(data)):
            return
    data=real_or_fake(data)
    data=fix_dot(data)
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(data, f,ensure_ascii=False, indent=2)
folder_path="user_37"
def fix_backspace(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    new={"keystrokes":[]}
    for i in range(len(data["keystrokes"])):
        if data["keystrokes"][i]["key"]=="Backspace" and data["keystrokes"][i]["event"]=="keydown":
            if data["keystrokes"][i-1]["key"]=="Backspace" and data["keystrokes"][i-1]["event"]=="keydown":
                pass
            else:
                new["keystrokes"].append(data["keystrokes"][i])
        else:
            new["keystrokes"].append(data["keystrokes"][i])

    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(new, f,ensure_ascii=False, indent=2)
def fix_delete(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    new={"keystrokes":[]}
    for i in range(len(data["keystrokes"])):
        if data["keystrokes"][i]["key"]=="Delete" and data["keystrokes"][i]["event"]=="keydown":
            if data["keystrokes"][i-1]["key"]=="Delete" and data["keystrokes"][i-1]["event"]=="keydown":
                pass
            else:
                new["keystrokes"].append(data["keystrokes"][i])
        else:
            new["keystrokes"].append(data["keystrokes"][i])

    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(new, f,ensure_ascii=False, indent=2)
def fix_shift(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    new={"keystrokes":[]}
    for i in range(len(data["keystrokes"])):
        if data["keystrokes"][i]["key"]=="Shift" and data["keystrokes"][i]["event"]=="keydown":
            if data["keystrokes"][i-1]["key"]=="Shift" and data["keystrokes"][i-1]["event"]=="keydown":
                pass
            else:
                new["keystrokes"].append(data["keystrokes"][i])
        else:
            new["keystrokes"].append(data["keystrokes"][i])
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(new, f,ensure_ascii=False, indent=2)
def fix_arrow(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    new={"keystrokes":[]}
    for i in range(len(data["keystrokes"])):
        if "Arrow" in data["keystrokes"][i]["code"]:
            pass
        else:
            new["keystrokes"].append(data["keystrokes"][i])
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(new, f,ensure_ascii=False, indent=2)
def fix_control(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    new={"keystrokes":[]}
    for i in range(len(data["keystrokes"])):
        if "Control" in data["keystrokes"][i]["code"]:
            pass
        else:
            new["keystrokes"].append(data["keystrokes"][i])
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(new, f,ensure_ascii=False, indent=2)
def fix_duplicate(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    if not search3(data):
        return
    new={"keystrokes":[]}
    new["keystrokes"].append(data["keystrokes"][0])
    for i in range(1,len(data["keystrokes"])):
        if data["keystrokes"][i]["code"]==data["keystrokes"][i-1]["code"] and data["keystrokes"][i]["timestamp"]-data["keystrokes"][i]["timestamp"]<10 and data["keystrokes"][i]["event"]==data["keystrokes"][i-1]["event"]:
            pass
        else:
            new["keystrokes"].append(data["keystrokes"][i])
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(new, f,ensure_ascii=False, indent=2)
"""
fix_windows("15session1.json")
fix_backspace("preptest.json")
fix_shift("preptest.json")
fix_arrow("preptest.json")
"""


for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)
            print(file_path)
            fix_windows(file_path)
            fix_backspace(file_path)
            fix_delete(file_path)
            fix_shift(file_path)
            fix_arrow(file_path)
            fix_control(file_path)
            fix_duplicate(file_path)



