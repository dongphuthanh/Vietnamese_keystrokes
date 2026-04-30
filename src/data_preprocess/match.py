import json
import os


x={"a","b"}
def match(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
    x=[]
    i=0
    j=1
    while i<len(data["keystrokes"])-1:
        if j>=len(data["keystrokes"]):
            i+=1
            j=i+1
            continue
        if j>i+30:
            i+=1
            j=i+1
            continue
        if data["keystrokes"][i]==None:
            i+=1
            j=i+1
            continue
        if data["keystrokes"][i]["event"]=="keyup":
            i+=1
            j=i+1
            continue
        elif data["keystrokes"][i]["code"]==data["keystrokes"][j]["code"] and data["keystrokes"][j]["event"]=="keyup" and i!=j:
            x.append(data["keystrokes"][i])
            x.append(data["keystrokes"][j])
            i+=1
            j=i+1
            continue
        elif data["keystrokes"][i]["code"]==data["keystrokes"][j]["code"] and data["keystrokes"][j]["event"]=="keydown" and i!=j:
            data["keystrokes"][i]=None
            j+=1
        else:
            j+=1
        
    output={"keystrokes":x}
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(output, f,ensure_ascii=False, indent=2)

def match_upgrade(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        # Load the JSON data from the file
        data = json.load(f)
        
    x = []
    i = 0
    j = 1
    
    while i < len(data["keystrokes"]) - 1:
        if j >= len(data["keystrokes"]):
            i += 1
            j = i + 1
            continue
            
        if j > i + 20:
            i += 1
            j = i + 1
            continue
            
        if data["keystrokes"][i] is None:
            i += 1
            j = i + 1
            continue
            
        if data["keystrokes"][i]["event"] == "keyup":
            i += 1
            j = i + 1
            continue

        # --- THE UPGRADE: Safely check if the keys are synthetic ---
        is_i_synthetic = data["keystrokes"][i].get("actual") == "synthetic"
        is_j_synthetic = data["keystrokes"][j].get("actual") == "synthetic"
        
        # Check if the keys are the same AND belong to the same stream
        keys_match = (data["keystrokes"][i]["code"] == data["keystrokes"][j]["code"])
        streams_match = (is_i_synthetic == is_j_synthetic)

        # Only process the pairing if both the key code and the stream match
        if keys_match and streams_match and i != j:
            if data["keystrokes"][j]["event"] == "keyup":
                # Perfect match found
                x.append(data["keystrokes"][i])
                x.append(data["keystrokes"][j])
                i += 1
                j = i + 1
                continue
            elif data["keystrokes"][j]["event"] == "keydown":
                # Conflict: The same stream pressed this key down again before releasing it
                data["keystrokes"][i] = None
                j += 1
            else:
                j += 1
        else:
            # If the keys don't match, or they belong to different streams, skip ahead
            j += 1
            
    output = {"keystrokes": x}
    with open(file_path, 'w', encoding="utf8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

def find_anomaly(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        data = json.load(f)
    i=0
    j=1
    while i<len(data["keystrokes"])-1:
        if data["keystrokes"][i]["key"]=="Shift":
            if int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"])>10000:
                pass
                #print("Anomalies found: "+file_path)
                #print(data["keystrokes"][i])
                #print(data["keystrokes"][j])
                #print(int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"]))
                #print("=============================")
        elif data["keystrokes"][i]["code"]=="Control" or data["keystrokes"][i]["code"]=="Backspace":
            if int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"])>5000:
                pass
                #print("Anomalies found: "+file_path)
                #print(data["keystrokes"][i])
                #print(data["keystrokes"][j])
                #print(int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"]))
                #print("=============================")
        elif int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"])>5000:
            print("Anomalies found: "+file_path)
            print(data["keystrokes"][i])
            print(data["keystrokes"][j])
            print(int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"]))
            print("=============================")
        i+=1
        j+=1
def fix_anomaly(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        data = json.load(f)
    i=len(data["keystrokes"])-4
    j=i+2
    while i>=0:
        if int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"])>2000:
            data["keystrokes"].pop(j+1)
            data["keystrokes"].pop(j)
            #data["keystrokes"].pop(i+1)
           # data["keystrokes"].pop(i)
        j-=2
        i-=2
    with open(file_path, 'w',encoding="utf8") as f:
        json.dump(data, f,ensure_ascii=False, indent=2)
def check_time(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        data = json.load(f)
    i=0
    j=2
    while i<len(data["keystrokes"])-2:
        if int(data["keystrokes"][i]["timestamp"])>int(data["keystrokes"][j]["timestamp"]):
            print("Anomalies found: "+file_path)
            print(data["keystrokes"][i])
            print(data["keystrokes"][j])
        if int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"])>200000:
            print("Anomalies found: "+file_path)
            print(data["keystrokes"][i])
            print(data["keystrokes"][j])
            print(int(data["keystrokes"][j]["timestamp"])-int(data["keystrokes"][i]["timestamp"]))
            print("================================")
        i+=2
        j+=2

def check_pos(file_path):
    with open(file_path, 'r',encoding="utf8") as f:
        data = json.load(f)
    i=0
    j=2
    num_anomalies=0
    backspace=0
    while i<len(data["keystrokes"])-2:
        if data["keystrokes"][i]["standardized"]=="Shift" or data["keystrokes"][i]["standardized"]=="CapsLock" or data["keystrokes"][i]["standardized"]=="Control":
            i+=2
            j=i+2
            continue
        else: 
            if j-i>16:
                print("Anomalies found: "+file_path)
                print(data["keystrokes"][i])
                print(data["keystrokes"][i+1])
                print(backspace)
                num_anomalies+=1
                print("================================")
                i+=2
                j=i+2
                backspace=0
                continue
            elif j>len(data["keystrokes"])-1:
                i+=2
                j=i+2
                backspace=0
                continue
            elif data["keystrokes"][j]["timestamp"]>data["keystrokes"][i+1]["timestamp"]:
                i+=2
                j=i+2
                backspace=0
                continue
            else:
                if data["keystrokes"][j]["code"]=="Backspace":
                    backspace+=1
                j+=2
    return num_anomalies
folder_path = "../../dataset/Korean" 



for root, _, files in os.walk(folder_path):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)
            find_anomaly(file_path)


