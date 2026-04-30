"""
Step 1: Extract KIT and KHT features from raw JSON files → save to PKL.

KHT (Key Hold Time)  = time between keydown and keyup for the same key
KIT (Key Interval Time) = time between consecutive keydown events (bigram)

Cognitive levels are grouped into 3 pairs before feature extraction:
  Group 1: levels {1, 4}   Group 2: levels {2, 5}   Group 3: levels {3, 6}
Features are computed over all keystrokes from both levels in the group
within the same session → more stable statistics, lower variance.

Each file produces 9 rows: 3 sessions × 3 groups.

Outputs:
  pkl/full.pkl    – normal users
  pkl/attack.pkl  – attack users
"""

import json
import pickle
import string
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    NORMAL_FOLDER, ATTACK_FOLDER,
    NORMAL_PKL, ATTACK_PKL,
    PKL_DIR, TOP_BIGRAMS_N,
)

# ============================================================
# VALID KEYS
# ============================================================
VALID_KEYS = set(string.ascii_lowercase) | {" ", "backspace", "shift"}
BIGRAM_KEYS = set(string.ascii_lowercase) | {" "}

# Group cognitive levels into 3 pairs
LEVEL_GROUPS = {1: [1, 4], 2: [2, 5], 3: [3, 6]}

# ============================================================
# SUMMARIZE A LIST → 6 STATISTICS  (with IQR outlier removal)
# ============================================================
TIMING_MIN_MS = 50.0
TIMING_MAX_MS = 5000.0

def summarize(values: list) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "s/m": 0.0, "range": 0.0, "max": 0.0, "median": 0.0}
    arr = np.array(values, dtype=float)
    arr = arr[(arr >= TIMING_MIN_MS) & (arr <= TIMING_MAX_MS)]
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "s/m": 0.0, "range": 0.0, "max": 0.0, "median": 0.0}
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    arr = arr[(arr >= q1 - 2 * iqr) & (arr <= q3 + 2 * iqr)]
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "s/m": 0.0, "range": 0.0, "max": 0.0, "median": 0.0}
    m = float(np.mean(arr))
    s = float(np.std(arr))
    return {
        "mean":   m,
        "std":    s,
        "s/m":    s / m if m != 0 else 0.0,
        "range":  float(np.max(arr) - np.min(arr)),
        "max":    float(np.max(arr)),
        "median": float(np.median(arr)),
    }


# ============================================================
# PASS 1 – COUNT BIGRAMS ACROSS ALL JSON FILES IN A FOLDER
# ============================================================
def count_bigrams_in_folder(folder: Path, top_n: int) -> list:
    total: Counter = Counter()
    for json_file in folder.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        keystrokes = data.get("keystrokes", [])
        prev_key, prev_type = None, None
        for ev in keystrokes:
            key  = str(ev.get("key", "")).lower()
            typ  = ev.get("event", "").lower()
            if key not in BIGRAM_KEYS or typ not in {"keydown", "keyup"}:
                continue
            if prev_type == "keyup" and typ == "keydown" and prev_key in BIGRAM_KEYS:
                total[f"{prev_key}->{key}"] += 1
            prev_key, prev_type = key, typ
    return [b for b, _ in total.most_common(top_n)]


# ============================================================
# COLLECT RAW KHT/KIT/RUKD VALUES FROM A SINGLE QUESTION
# (does not cross question boundaries)
# ============================================================
def _collect_question_values(events, top_bigrams, kht, kit, rukd):
    """Accumulate raw timing values from one question's events into kht/kit/rukd."""
    last_down: dict = {}
    prev_down_key: Optional[str] = None
    prev_down_ts:  Optional[float] = None
    prev_up_key:   Optional[str] = None
    prev_up_ts:    Optional[float] = None

    for key, typ, ts in events:
        if typ == "keydown":
            last_down[key] = ts
            if prev_down_key is not None:
                bigram = f"{prev_down_key}->{key}"
                interval = ts - prev_down_ts
                if interval >= 0 and bigram in kit:
                    kit[bigram].append(interval)
            if prev_up_key is not None and prev_up_key in BIGRAM_KEYS and key in BIGRAM_KEYS:
                bigram = f"{prev_up_key}->{key}"
                flight = ts - prev_up_ts
                if flight >= 0 and bigram in rukd:
                    rukd[bigram].append(flight)
            prev_down_key = key
            prev_down_ts  = ts
        elif typ == "keyup" and key in last_down:
            hold = ts - last_down[key]
            if hold >= 0:
                kht[key].append(hold)
            if key in BIGRAM_KEYS:
                prev_up_key = key
                prev_up_ts  = ts


# ============================================================
# EXTRACT FEATURES FROM ONE JSON FILE
# Returns up to 9 rows: one per (session_id, group_id).
# Features are computed over combined keystrokes from both levels
# in the group, within the same session.
# ============================================================
def extract_features_from_json(filepath: Path, top_bigrams: list) -> list:
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [skip] bad JSON: {filepath} ({e})")
        return []

    keystrokes = data.get("keystrokes", [])

    # Build index: (session_id, level) → list of (key, typ, ts)
    question_events: dict = {}
    for ev in keystrokes:
        qi = ev.get("question_index")
        if qi in (None, 0, "0"):
            continue
        try:
            parts = str(qi).split(".")
            sess  = int(parts[0])
            level = int(parts[1])
        except Exception:
            continue
        key = str(ev.get("key", "")).lower()
        typ = ev.get("event", "").lower()
        ts  = ev.get("timestamp")
        if key in VALID_KEYS and typ in {"keydown", "keyup"} and ts is not None:
            question_events.setdefault((sess, level), []).append((key, typ, ts))

    rows_out = []

    for session_id in [1, 2, 3]:
        for group_id, levels in LEVEL_GROUPS.items():
            kht  = {k: [] for k in VALID_KEYS}
            kit  = {b: [] for b in top_bigrams}
            rukd = {b: [] for b in top_bigrams}

            found_any = False
            for level in levels:
                q_events = question_events.get((session_id, level), [])
                if q_events:
                    found_any = True
                    _collect_question_values(q_events, top_bigrams, kht, kit, rukd)

            if not found_any:
                continue

            flat: dict = {}
            for key, vals in kht.items():
                for stat, val in summarize(vals).items():
                    flat[f"{key}_{stat}"] = val
            for bigram, vals in kit.items():
                for stat, val in summarize(vals).items():
                    flat[f"{bigram}_{stat}"] = val
            for bigram, vals in rukd.items():
                for stat, val in summarize(vals).items():
                    flat[f"{bigram}_rkd_{stat}"] = val

            flat["session"]  = session_id
            flat["group_id"] = group_id
            rows_out.append(flat)

    return rows_out


# ============================================================
# PROCESS ONE FOLDER OF USERS
# ============================================================
def process_folder(
    folder: Path,
    json_filenames: list,
    top_bigrams: list,
    source_label: str,
) -> list:
    rows = []
    for user_dir in sorted(folder.iterdir()):
        if not user_dir.is_dir():
            continue
        user_id = user_dir.name
        for fname in json_filenames:
            fpath = user_dir / fname
            if not fpath.exists():
                print(f"  [missing] {fpath}")
                continue
            extracted = extract_features_from_json(fpath, top_bigrams)
            for row in extracted:
                row["user_id"] = user_id
                row["source"]  = source_label
                row["file"]    = fname
            rows.extend(extracted)
    return rows


# ============================================================
# MAIN
# ============================================================
def main():
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    if not NORMAL_FOLDER.exists():
        raise FileNotFoundError(f"Normal folder not found: {NORMAL_FOLDER}")
    if not ATTACK_FOLDER.exists():
        raise FileNotFoundError(f"Attack folder not found: {ATTACK_FOLDER}")

    print(f"Computing top-{TOP_BIGRAMS_N} bigrams from {NORMAL_FOLDER} ...")
    top_bigrams = count_bigrams_in_folder(NORMAL_FOLDER, TOP_BIGRAMS_N)
    print(f"  Done. Example bigrams: {top_bigrams[:5]}")

    print(f"\nExtracting normal features from {NORMAL_FOLDER} ...")
    normal_rows = process_folder(
        NORMAL_FOLDER,
        json_filenames=["first_time.json", "second_time.json"],
        top_bigrams=top_bigrams,
        source_label="normal",
    )
    print(f"  {len(normal_rows)} rows extracted")
    with open(NORMAL_PKL, "wb") as f:
        pickle.dump(normal_rows, f)
    print(f"  Saved to {NORMAL_PKL}")

    print(f"\nExtracting attack features from {ATTACK_FOLDER} ...")
    attack_rows = process_folder(
        ATTACK_FOLDER,
        json_filenames=["first_time_attack.json", "second_time_attack.json"],
        top_bigrams=top_bigrams,
        source_label="attack",
    )
    print(f"  {len(attack_rows)} rows extracted")
    with open(ATTACK_PKL, "wb") as f:
        pickle.dump(attack_rows, f)
    print(f"  Saved to {ATTACK_PKL}")

    print("\n[Step 1] Done.")


if __name__ == "__main__":
    main()
