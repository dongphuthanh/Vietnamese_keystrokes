"""
Step 1: Extract KIT/KHT/RUKD features from raw JSON files → save to PKL.

Two extraction granularities are produced:

  Context-independent (9 rows/file):
    Levels grouped into 3 pairs per session: {1,4}, {2,5}, {3,6}.
    Each row = one (session, group) combination.
    → lower variance statistics (more keystrokes per row).

  User-independent (3 rows/file):
    All 6 levels combined per session.
    Each row = one full session.
    → maximum data aggregation per session.

Outputs:
  pkl/full_context.pkl   pkl/attack_context.pkl   (context-indep)
  pkl/full_user.pkl      pkl/attack_user.pkl       (user-indep)
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
    NORMAL_PKL_CONTEXT, ATTACK_PKL_CONTEXT,
    NORMAL_PKL_USER,   ATTACK_PKL_USER,
    PKL_DIR, TOP_BIGRAMS_N,
)

VALID_KEYS  = set(string.ascii_lowercase) | {" ", "backspace", "shift"}
BIGRAM_KEYS = set(string.ascii_lowercase) | {" "}

# Groupings
LEVEL_GROUPS_CONTEXT = {1: [1, 4], 2: [2, 5], 3: [3, 6]}   # 9 rows/file
LEVEL_GROUPS_USER    = {1: [1, 2, 3, 4, 5, 6]}              # 3 rows/file

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
            key = str(ev.get("key", "")).lower()
            typ = ev.get("event", "").lower()
            if key not in BIGRAM_KEYS or typ not in {"keydown", "keyup"}:
                continue
            if prev_type == "keyup" and typ == "keydown" and prev_key in BIGRAM_KEYS:
                total[f"{prev_key}->{key}"] += 1
            prev_key, prev_type = key, typ
    return [b for b, _ in total.most_common(top_n)]


def _collect_question_values(events, top_bigrams, kht, kit, rukd):
    """Accumulate raw timing values from one question into kht/kit/rukd dicts."""
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


def _flatten(kht, kit, rukd) -> dict:
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
    return flat


def extract_features_from_json(filepath: Path, top_bigrams: list,
                                level_groups: dict) -> list:
    """
    Extract features from one JSON file.

    level_groups: dict mapping group_id → list of cognitive levels.
      LEVEL_GROUPS_CONTEXT = {1:[1,4], 2:[2,5], 3:[3,6]}  → 9 rows/file
      LEVEL_GROUPS_USER    = {1:[1,2,3,4,5,6]}             → 3 rows/file
    """
    try:
        data = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [skip] bad JSON: {filepath} ({e})")
        return []

    keystrokes = data.get("keystrokes", [])

    # Index events by (session_id, level)
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
        for group_id, levels in level_groups.items():
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

            flat = _flatten(kht, kit, rukd)
            flat["session"]  = session_id
            flat["group_id"] = group_id
            rows_out.append(flat)

    return rows_out


def process_folder(folder: Path, json_filenames: list, top_bigrams: list,
                   source_label: str, level_groups: dict) -> list:
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
            extracted = extract_features_from_json(fpath, top_bigrams, level_groups)
            for row in extracted:
                row["user_id"] = user_id
                row["source"]  = source_label
                row["file"]    = fname
            rows.extend(extracted)
    return rows


def _save(rows, path):
    with open(path, "wb") as f:
        pickle.dump(rows, f)
    print(f"    Saved {len(rows)} rows -> {path}")


def main():
    PKL_DIR.mkdir(parents=True, exist_ok=True)

    if not NORMAL_FOLDER.exists():
        raise FileNotFoundError(f"Normal folder not found: {NORMAL_FOLDER}")
    if not ATTACK_FOLDER.exists():
        raise FileNotFoundError(f"Attack folder not found: {ATTACK_FOLDER}")

    print(f"Computing top-{TOP_BIGRAMS_N} bigrams from {NORMAL_FOLDER} ...")
    top_bigrams = count_bigrams_in_folder(NORMAL_FOLDER, TOP_BIGRAMS_N)
    print(f"  Done. Example: {top_bigrams[:5]}")

    normal_files = ["first_time.json", "second_time.json"]
    attack_files = ["first_time_attack.json", "second_time_attack.json"]

    # ---- Context-independent PKLs (grouped by level pairs) ----
    print("\n[Context-indep] Extracting normal ...")
    normal_ctx = process_folder(NORMAL_FOLDER, normal_files, top_bigrams, "normal", LEVEL_GROUPS_CONTEXT)
    print("[Context-indep] Extracting attack ...")
    attack_ctx = process_folder(ATTACK_FOLDER, attack_files, top_bigrams, "attack", LEVEL_GROUPS_CONTEXT)
    _save(normal_ctx, NORMAL_PKL_CONTEXT)
    _save(attack_ctx, ATTACK_PKL_CONTEXT)

    # ---- User-independent PKLs (all levels per session) ----
    print("\n[User-indep] Extracting normal ...")
    normal_usr = process_folder(NORMAL_FOLDER, normal_files, top_bigrams, "normal", LEVEL_GROUPS_USER)
    print("[User-indep] Extracting attack ...")
    attack_usr = process_folder(ATTACK_FOLDER, attack_files, top_bigrams, "attack", LEVEL_GROUPS_USER)
    _save(normal_usr, NORMAL_PKL_USER)
    _save(attack_usr, ATTACK_PKL_USER)

    print("\n[Step 1] Done.")


if __name__ == "__main__":
    main()
