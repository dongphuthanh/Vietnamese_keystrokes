import json
import string
from collections import defaultdict, Counter
import numpy as np
import os
import pickle


def summarize_list(lst):
    if len(lst) == 0:
        return {k: 0.0 for k in ["mean", "std", "s/m", "range", "max", "median"]}
    arr = np.array(lst, dtype=float)
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    arr = arr[(arr >= Q1 - 2 * IQR) & (arr <= Q3 + 2 * IQR)]
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "s/m": float(np.std(arr) / np.mean(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def extract_key_times_flattened(filepath, top_bigrams):
    valid_keys = set(string.ascii_lowercase) | {" ", "backspace", "shift"}
    try:
        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f" Skipping bad JSON: {filepath} ({e})")
        return []

    flattened_sessions = []

    for session_id in [1, 2, 3]:
        result = {k: [] for k in valid_keys}
        # result={}
        result.update({bigram: [] for bigram in top_bigrams})

        # Filter events for current session
        session_events = [k for k in data.get("keystrokes", []) if k.get("session") == session_id]

        # Keep only valid events
        events = []

        for k in session_events:
            key = str(k.get("key", "")).lower()
            type_ = k.get("event", "").lower()
            ts = k.get("timestamp")
            if key in valid_keys and type_ in {"keydown", "keyup"} and ts is not None:
                events.append({"key": key, "event": type_, "timestamp": ts})

        last_keydown = {}
        prev_keydown_event = None

        for ev in events:
            key, type_, ts = ev["key"], ev["event"], ev["timestamp"]

            # Hold time: keydown -> keyup
            if type_ == "keydown":
                last_keydown[key] = ts
                # Interval time: previous keydown -> current keydown
                if prev_keydown_event:
                    prev_key = prev_keydown_event["key"]
                    prev_ts = prev_keydown_event["timestamp"]
                    bigram_str = f"{prev_key}->{key}"
                    interval_time = ts - prev_ts
                    if interval_time >= 0 and bigram_str in top_bigrams:
                        result[bigram_str].append(interval_time)
                prev_keydown_event = {"key": key, "timestamp": ts}

            elif type_ == "keyup" and key in last_keydown:
                hold_time = ts - last_keydown[key]
                if hold_time >= 0:
                    result[key].append(hold_time)

        # Flatten lists into summary stats
        flat_features = {}
        for k, v in result.items():
            stats = summarize_list(v)
            for stat_name, value in stats.items():
                flat_features[f"{k}_{stat_name}"] = value
        flattened_sessions.append(flat_features)

    return flattened_sessions


def build_key_bigram(filepath, valid_keys):
    bigrams = defaultdict(Counter)
    try:
        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Skipping bad JSON: {filepath} ({e})")
        return bigrams
    keystrokes = data.get("keystrokes", [])
    events = [k for k in keystrokes if k.get("key") and k.get("event") and k.get("timestamp") is not None]
    for i in range(len(events) - 1):
        curr, next_ = events[i], events[i + 1]
        key1, type1 = str(curr["key"]).lower(), curr["event"].lower()
        key2, type2 = str(next_["key"]).lower(), next_["event"].lower()
        if type1 == "keyup" and type2 == "keydown" and key1 in valid_keys and key2 in valid_keys:
            bigrams[key1][key2] += 1
    return bigrams


def merge_bigrams(bigrams_total, bigrams_new):
    for k1, sub in bigrams_new.items():
        for k2, count in sub.items():
            bigrams_total[k1][k2] += count


def get_top_bigrams_from_folder(folder_path, top_n=50):
    valid_keys = set(string.ascii_lowercase) | {" "}
    bigrams_total = defaultdict(Counter)
    # Walk through folder and subfolders
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".json"):
                filepath = os.path.join(root, filename)
                bigrams_file = build_key_bigram(filepath, valid_keys)
                merge_bigrams(bigrams_total, bigrams_file)
    # Flatten and get top N
    flat_counts = Counter()
    for k1, sub in bigrams_total.items():
        for k2, count in sub.items():
            flat_counts[f"{k1}->{k2}"] = count
    top_bigrams = [k for k, _ in flat_counts.most_common(top_n)]
    return top_bigrams


def extract_key_times_flattened(filepath, top_bigrams):
    valid_keys = set(string.ascii_lowercase) | {" ", "backspace", "shift"}
    try:
        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Skipping bad JSON: {filepath} ({e})")
        return []

    flattened_sessions = []

    for session_id in [1, 2, 3]:
        result = {k: [] for k in valid_keys}
        result.update({bigram: [] for bigram in top_bigrams})

        session_events = [k for k in data.get("keystrokes", []) if k.get("session") == session_id]

        # --- ĐOẠN NÀY QUAN TRỌNG: Lấy question_index cho cả Attack ---
        q_index = 0
        if len(session_events) > 0:
            q_index = session_events[0].get("question_index", 0)
        # -------------------------------------------------------------

        events = []
        for k in session_events:
            key = str(k.get("key", "")).lower()
            type_ = k.get("event", "").lower()
            ts = k.get("timestamp")
            if key in valid_keys and type_ in {"keydown", "keyup"} and ts is not None:
                events.append({"key": key, "event": type_, "timestamp": ts})

        last_keydown = {}
        prev_keydown_event = None

        for ev in events:
            key, type_, ts = ev["key"], ev["event"], ev["timestamp"]

            if type_ == "keydown":
                last_keydown[key] = ts
                if prev_keydown_event:
                    prev_key = prev_keydown_event["key"]
                    prev_ts = prev_keydown_event["timestamp"]
                    bigram_str = f"{prev_key}->{key}"
                    interval_time = ts - prev_ts
                    if interval_time >= 0 and bigram_str in top_bigrams:
                        result[bigram_str].append(interval_time)
                prev_keydown_event = {"key": key, "timestamp": ts}

            elif type_ == "keyup" and key in last_keydown:
                hold_time = ts - last_keydown[key]
                if hold_time >= 0:
                    result[key].append(hold_time)

        flat_features = {}
        for k, v in result.items():
            stats = summarize_list(v)
            for stat_name, value in stats.items():
                flat_features[f"{k}_{stat_name}"] = value
                
        # --- GÁN QUESTION_INDEX VÀO KẾT QUẢ ---
        flat_features["question_index"] = q_index
        flattened_sessions.append(flat_features)

    return flattened_sessions


def extract_attack_features(folder_path, pickle_path, top_bigrams):
    all_features = []

    for user_folder in sorted(os.listdir(folder_path)):
        user_path = os.path.join(folder_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        user_id = user_folder

        for filename in ["first_time_attack.json", "second_time_attack.json"]:
            filepath = os.path.join(user_path, filename)
            if not os.path.exists(filepath):
                print(f"  Missing: {filepath}")
                continue

            session_features = extract_key_times_flattened(filepath, top_bigrams)

            session_label_map = {1: "bonafide", 2: "paraphrase", 3: "transcribe"}

            for i, feat in enumerate(session_features):
                feat["user_id"] = user_id
                feat["label"] = session_label_map[i + 1]
                feat["file"] = filename
                feat["session"] = i + 1

            all_features.extend(session_features)

    with open(pickle_path, "wb") as f:
        pickle.dump(all_features, f)
    print(f"Saved {len(all_features)} attack rows")

def extract_features_from_folder(folder_path, pickle_path, top_bigrams):
    all_features = []

    for user_folder in sorted(os.listdir(folder_path)):
        user_path = os.path.join(folder_path, user_folder)
        if not os.path.isdir(user_path):
            continue

        user_id = user_folder

        # File của normal thường là first_time.json và second_time.json
        for filename in ["first_time.json", "second_time.json"]:
            filepath = os.path.join(user_path, filename)
            if not os.path.exists(filepath):
                print(f"  Missing: {filepath}")
                continue

            session_features = extract_key_times_flattened(filepath, top_bigrams)

            for i, feat in enumerate(session_features):
                feat["user_id"] = user_id
                feat["label"] = "normal"  # Gán nhãn bình thường
                feat["file"] = filename
                feat["session"] = i + 1

            all_features.extend(session_features)

    with open(pickle_path, "wb") as f:
        pickle.dump(all_features, f)
    print(f"Saved {len(all_features)} rows")


# ==========================================
# --- CHẠY CẢ 2: NORMAL VÀ ATTACK ---
# ==========================================

# 1. Khai báo đường dẫn thư mục đầu vào (Input)
normal_folder = r"C:\Users\cao minh\Downloads\preproccessed-20260223T225650Z-1-001\preproccessed"

# LƯU Ý: Giống hệt vụ giải nén lúc nãy, nếu bên trong thư mục Attack-2026... có một thư mục con tên là Attack, 
# thì bạn phải thêm "\Attack" vào đuôi giống như dòng dưới đây nhé!
attack_folder = r"C:\Users\cao minh\Downloads\Attack-20260404T023818Z-1-001\Attack" 

# 2. Khai báo đường dẫn file đầu ra (Output)
normal_pickle = r"C:\Users\cao minh\Vietnamese_keystrokes\src\Minh_code\full.pkl"
attack_pickle = r"C:\Users\cao minh\Vietnamese_keystrokes\src\Minh_code\attack.pkl"

# 3. Thực thi
print("Đang tính top bigrams...")
top_bigrams = get_top_bigrams_from_folder(normal_folder, top_n=200)
print("Xong top bigrams!")

print("\nĐang xử lý Normal data...")
# (Hãy chắc chắn hàm extract_features_from_folder đã được định nghĩa ở trên nhé)
extract_features_from_folder(normal_folder, normal_pickle, top_bigrams)

print("\nĐang xử lý Attack data...")
extract_attack_features(attack_folder, attack_pickle, top_bigrams)

print("\n🎉 HOÀN TẤT TOÀN BỘ!")