#!/usr/bin/env python3
"""
prep.py - Trích xuất features từ JSON keystroke data
Cấu trúc folder (mỗi user):
  ├── first_time.json
  ├── second_time.json           (có thể không có)
  ├── first_time_attack.json     (có thể không có)
  └── second_time_attack.json    (có thể không có)

full.pkl   → tất cả file KHÔNG có "_attack" (first_time + second_time)
attack.pkl → tất cả file CÓ "_attack"       (first_time_attack + second_time_attack)

1 row = 1 (user, session, question_index) — giữ cognitive level riêng biệt
"""

import json
import string
from collections import defaultdict, Counter
import numpy as np
import os
import pickle


# ==========================================
# HELPERS
# ==========================================
def summarize_list(lst):
    if len(lst) == 0:
        return {k: 0.0 for k in ["mean", "std", "s/m", "range", "max", "median"]}
    arr = np.array(lst, dtype=float)
    arr = arr[(arr >= 50) & (arr <= 2000)]
    if len(arr) == 0:
        return {k: 0.0 for k in ["mean", "std", "s/m", "range", "max", "median"]}
    Q1, Q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    IQR = Q3 - Q1
    arr = arr[(arr >= Q1 - 1.5 * IQR) & (arr <= Q3 + 1.5 * IQR)]
    if len(arr) == 0:
        return {k: 0.0 for k in ["mean", "std", "s/m", "range", "max", "median"]}
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr)),
        "s/m":    float(np.std(arr) / np.mean(arr)) if np.mean(arr) != 0 else 0.0,
        "range":  float(np.max(arr) - np.min(arr)),
        "max":    float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def build_key_bigram(filepath, valid_keys):
    bigrams = defaultdict(Counter)
    try:
        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return bigrams
    ks = data.get("keystrokes", [])
    for i in range(len(ks) - 1):
        k1 = str(ks[i].get("key", "")).lower()
        k2 = str(ks[i+1].get("key", "")).lower()
        if (ks[i].get("event","").lower() == "keyup" and
                ks[i+1].get("event","").lower() == "keydown" and
                k1 in valid_keys and k2 in valid_keys):
            bigrams[k1][k2] += 1
    return bigrams


def get_top_bigrams_from_folder(folder_path, top_n=200):
    valid_keys = set(string.ascii_lowercase) | {" "}
    total = defaultdict(Counter)
    for root, _, files in os.walk(folder_path):
        for fn in files:
            if fn.endswith(".json"):
                for k1, sub in build_key_bigram(os.path.join(root, fn), valid_keys).items():
                    for k2, cnt in sub.items():
                        total[k1][k2] += cnt
    flat = Counter({f"{k1}->{k2}": cnt for k1, sub in total.items() for k2, cnt in sub.items()})
    return [k for k, _ in flat.most_common(top_n)]


def extract_features_per_question(filepath, top_bigrams):
    """1 row per (session, question_index)"""
    valid_keys = set(string.ascii_lowercase) | {" ", "backspace", "shift"}
    try:
        with open(filepath, "r", encoding="utf8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Skipping bad JSON: {filepath} ({e})")
        return []

    # Group theo (session, question_index)
    groups = defaultdict(list)
    for k in data.get("keystrokes", []):
        sess = k.get("session")
        qi   = k.get("question_index")
        key  = str(k.get("key", "")).lower()
        evt  = k.get("event", "").lower()
        ts   = k.get("timestamp")
        if sess and qi and key in valid_keys and evt in {"keydown", "keyup"} and ts is not None:
            groups[(sess, str(qi))].append({"key": key, "event": evt, "timestamp": ts})

    results = []
    for (sess, qi), events in sorted(groups.items()):
        acc = {k: [] for k in valid_keys}
        acc.update({b: [] for b in top_bigrams})

        last_down = {}
        prev_down = None
        for ev in events:
            key, evt, ts = ev["key"], ev["event"], ev["timestamp"]
            if evt == "keydown":
                last_down[key] = ts
                if prev_down:
                    bg = f"{prev_down['key']}->{key}"
                    dt = ts - prev_down["timestamp"]
                    if dt >= 0 and bg in top_bigrams:
                        acc[bg].append(dt)
                prev_down = {"key": key, "timestamp": ts}
            elif evt == "keyup" and key in last_down:
                ht = ts - last_down[key]
                if ht >= 0:
                    acc[key].append(ht)

        flat = {}
        for k, v in acc.items():
            for stat, val in summarize_list(v).items():
                flat[f"{k}_{stat}"] = val
        flat["session"]        = sess
        flat["question_index"] = qi
        results.append(flat)

    return results


# ==========================================
# EXTRACT NORMAL (file không có _attack)
# ==========================================
def extract_normal(folder_path, pickle_path, top_bigrams):
    session_label_map = {1: "bonafide", 2: "paraphrase", 3: "transcribe"}
    all_features = []

    for user_folder in sorted(os.listdir(folder_path)):
        user_path = os.path.join(folder_path, user_folder)
        if not os.path.isdir(user_path):
            continue
        user_id = user_folder

        # Lấy tất cả .json KHÔNG có "_attack"
        normal_files = sorted([
            fn for fn in os.listdir(user_path)
            if fn.endswith(".json") and "_attack" not in fn
        ])

        if not normal_files:
            print(f"  [NORMAL] {user_id}: không có file normal")
            continue

        for filename in normal_files:
            filepath = os.path.join(user_path, filename)
            rows = extract_features_per_question(filepath, top_bigrams)
            for feat in rows:
                feat["user_id"] = user_id
                feat["file"]    = filename
                feat["label"]   = session_label_map.get(feat["session"], "unknown")
            all_features.extend(rows)
            print(f"  [NORMAL] {user_id}/{filename}: {len(rows)} rows | "
                  f"qi: {[f['question_index'] for f in rows]}")

    with open(pickle_path, "wb") as f:
        pickle.dump(all_features, f)
    print(f"\n✅ full.pkl: {len(all_features)} rows → {pickle_path}")


# ==========================================
# EXTRACT ATTACK (file có _attack)
# ==========================================
def extract_attack(folder_path, pickle_path, top_bigrams):
    all_features = []

    for user_folder in sorted(os.listdir(folder_path)):
        user_path = os.path.join(folder_path, user_folder)
        if not os.path.isdir(user_path):
            continue
        user_id = user_folder

        # Lấy tất cả .json CÓ "_attack"
        attack_files = sorted([
            fn for fn in os.listdir(user_path)
            if fn.endswith(".json") and "_attack" in fn
        ])

        if not attack_files:
            print(f"  [ATTACK] {user_id}: không có file attack")
            continue

        for filename in attack_files:
            filepath = os.path.join(user_path, filename)
            rows = extract_features_per_question(filepath, top_bigrams)
            for feat in rows:
                feat["user_id"] = user_id
                feat["file"]    = filename
            all_features.extend(rows)
            print(f"  [ATTACK] {user_id}/{filename}: {len(rows)} rows | "
                  f"qi: {[f['question_index'] for f in rows]}")

    with open(pickle_path, "wb") as f:
        pickle.dump(all_features, f)
    print(f"\n✅ attack.pkl: {len(all_features)} rows → {pickle_path}")


# ==========================================
# CHẠY
# ==========================================
if __name__ == "__main__":
    FOLDER_PATH = r"C:\Users\cao minh\Downloads\Attack-20260409T014455Z-3-001\Attack"
    NORMAL_PKL  = r"C:\Users\cao minh\Vietnamese_keystrokes\src\Minh_code\full.pkl"
    ATTACK_PKL  = r"C:\Users\cao minh\Vietnamese_keystrokes\src\Minh_code\attack.pkl"

    print("="*60)
    print("BƯỚC 1: Tính top bigrams...")
    print("="*60)
    top_bigrams = get_top_bigrams_from_folder(FOLDER_PATH, top_n=200)
    print(f"  Computed {len(top_bigrams)} bigrams.")

    print("\n" + "="*60)
    print("BƯỚC 2: Extract NORMAL (file không có _attack)")
    print("="*60)
    extract_normal(FOLDER_PATH, NORMAL_PKL, top_bigrams)

    print("\n" + "="*60)
    print("BƯỚC 3: Extract ATTACK (file có _attack)")
    print("="*60)
    extract_attack(FOLDER_PATH, ATTACK_PKL, top_bigrams)

    # Kiểm tra
    print("\n" + "="*60)
    print("KIỂM TRA")
    print("="*60)
    import pandas as pd
    for path, name in [(NORMAL_PKL, "full.pkl"), (ATTACK_PKL, "attack.pkl")]:
        with open(path, "rb") as f:
            df = pd.DataFrame(pickle.load(f))
        print(f"\n{name}: {df.shape}")
        print(f"  question_index: {sorted(df['question_index'].unique())}")
        print(f"  session:        {sorted(df['session'].unique())}")
        if 'label' in df.columns:
            print(f"  label:          {sorted(df['label'].unique())}")