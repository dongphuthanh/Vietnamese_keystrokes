# -*- coding: utf-8 -*-
"""Extract the 107 rhythmic features (pauses, P-/R-/deletion bursts, entropy; Sec. III-B.4).

One sample = one question pair ({1,4}, {2,5}, {3,6}) of one writing session in one file.
Writes users_window_features.csv (B/P/T), attack_window_features.csv (FP/FT) and
merged_data.csv, where section_1..5 = B, P, T, FP, FT.

Originally a Colab notebook:
    https://colab.research.google.com/drive/1_WNF0Gas44FbvlaTwwUVtHwWAG8Pm0UJ

Usage:
    python extract_rhythmic_features.py [--dataset-dir DIR] [--output-dir DIR] [--include-phase2-attacks]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy


def replace_outliers_iqr_np(X, method='median'):
    """
    Handles outliers. Note: In keystroke dynamics, extreme pauses may
    reflect cognition rather than noise, so use with caution.
    """
    if len(X) < 5:
        return X

    X_clean = X.copy().astype(float)

    for i in range(X_clean.shape[1]):
        col = X_clean[:, i]
        Q1 = np.percentile(col, 25)
        Q3 = np.percentile(col, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        replacement = np.median(col) if method == 'median' else np.mean(col)

        mask = (col < lower) | (col > upper)
        X_clean[mask, i] = replacement

    return X_clean


def pause_entropy(pauses, bin_width=100, base=2):
    if len(pauses) == 0:
        return 0.0

    min_val, max_val = min(pauses), max(pauses)
    if min_val == max_val:
        return 0.0

    bins = np.arange(min_val, max_val + bin_width, bin_width)
    counts, _ = np.histogram(pauses, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    return entropy(probs, base=base)


def extract_keystroke_features(df):
    """
    Pooled feature extraction across multiple questions, but with hard resets
    at question boundaries so no stateful feature spans questions.

    Key behavior:
    - pauses across question boundaries are ignored
    - p-bursts are CLOSED and ADDED when question changes
    - edit bursts are reset at question changes
    - keydown/keyup pairs only count if they belong to the same question
    """

    # If your data is already in true chronological order, this is better:
    # df = df.sort_values(by=["timestamp"]).reset_index(drop=True)

    # If each question block is already conceptually separated, keep this:
    df = df.sort_values(by=["question_index", "timestamp"]).reset_index(drop=True)

    # --- 1. Pair keydowns with keyups ---
    key_units = []
    open_keys = {}
    prev_q_idx = None

    for _, row in df.iterrows():
        q_idx = row['question_index']
        k_id = row['code'] if ('code' in row and pd.notnull(row['code'])) else row['key']

        # Reset open key state at question shift
        if prev_q_idx is not None and q_idx != prev_q_idx:
            open_keys = {}

        if row['event'] == 'keydown':
            open_keys[k_id] = (row['timestamp'], row['key'], q_idx)

        elif row['event'] == 'keyup' and k_id in open_keys:
            down_ts, key_val, down_q_idx = open_keys.pop(k_id)
            if down_q_idx == q_idx:
                key_units.append({
                    'key': key_val,
                    'down': down_ts,
                    'up': row['timestamp'],
                    'hold': row['timestamp'] - down_ts,
                    'q_idx': q_idx
                })

        prev_q_idx = q_idx

    units_df = pd.DataFrame(key_units)
    if units_df.empty:
        return {}

    # --- 2. Calculate inter-key pauses ---
    units_df['pause'] = units_df['down'] - units_df['up'].shift(1)

    # Invalidate pause across question boundaries
    units_df.loc[units_df['q_idx'] != units_df['q_idx'].shift(1), 'pause'] = np.nan

    features = {}

    # --- 3. Stateful extraction ---
    edit_burst_durations = []
    r_burst_durations = []
    p_burst_durations = []

    pause_before_delete = []
    pause_inter_words = []
    pause_inter_sentences = []

    threshold_bins = {f"pause threshold {max(i - 300, 3)}-{i}": [] for i in range(300, 2401, 300)}
    threshold_bins["pause threshold 2400+"] = []

    current_edit_start = None
    current_edit_end = None

    current_q_idx = units_df.iloc[0]['q_idx']
    p_burst_start_time = units_df.iloc[0]['down']

    for i in range(len(units_df)):
        row = units_df.iloc[i]
        pause = row['pause']
        is_backspace = (row['key'] == 'Backspace')

        # --- HARD RESET on question boundary ---
        if row['q_idx'] != current_q_idx:
            prev_row = units_df.iloc[i - 1]

            # CLOSE previous question's p-burst and ADD it
            if prev_row['up'] > p_burst_start_time:
                p_burst_durations.append(prev_row['up'] - p_burst_start_time)

            # Close any edit burst that was active in the previous question
            if current_edit_start is not None:
                edit_burst_durations.append(current_edit_end - current_edit_start)
                current_edit_start = None
                current_edit_end = None

            # Reset for new question
            current_q_idx = row['q_idx']
            p_burst_start_time = row['down']

        # --- Pause-based features ---
        if not np.isnan(pause) and 10 < pause < 10000:
            if is_backspace:
                pause_before_delete.append(pause)
                r_burst_durations.append(pause)

            if row['key'] == ' ':
                pause_inter_words.append(pause)

            prev_key = units_df.iloc[i - 1]['key'] if i > 0 else None
            if prev_key in ['.', '!', '?', 'Enter']:
                pause_inter_sentences.append(pause)

            found_bin = False
            for t in range(300, 2401, 300):
                if max(t - 300, 3) <= pause < t:
                    threshold_bins[f"pause threshold {max(t - 300, 3)}-{t}"].append(pause)
                    found_bin = True
                    break

            if not found_bin and pause >= 2400:
                threshold_bins["pause threshold 2400+"].append(pause)

            # Long pause closes the current typing burst and starts a new one
            if pause > 2000:
                prev_row = units_df.iloc[i - 1]
                if prev_row['up'] > p_burst_start_time:
                    p_burst_durations.append(prev_row['up'] - p_burst_start_time)
                p_burst_start_time = row['down']

        # --- Edit burst logic ---
        if is_backspace:
            if current_edit_start is None:
                current_edit_start = row['down']
            current_edit_end = row['up']
        else:
            if current_edit_start is not None:
                edit_burst_durations.append(current_edit_end - current_edit_start)
                current_edit_start = None
                current_edit_end = None

    # --- Finalize trailing bursts at end of data ---

    # Final edit burst
    if current_edit_start is not None:
        edit_burst_durations.append(current_edit_end - current_edit_start)

    # Final p-burst
    last_row = units_df.iloc[-1]
    if last_row['up'] > p_burst_start_time:
        p_burst_durations.append(last_row['up'] - p_burst_start_time)

    # --- 4. Aggregate stats ---
    def process_and_stats(data_list, prefix, features_dict):
        if not data_list:
            features_dict[f'mean {prefix}'] = 0
            features_dict[f'median {prefix}'] = 0
            features_dict[f'std {prefix}'] = 0
            features_dict[f'total duration {prefix}' if 'duration' not in prefix else f'total {prefix}'] = 0
            features_dict[f'count {prefix}'] = 0
            features_dict[f'Q1 {prefix}'] = 0
            features_dict[f'Q3 {prefix}'] = 0
            return

        arr = np.array(data_list).reshape(-1, 1)
        clean_arr = replace_outliers_iqr_np(arr, method='median').flatten()

        features_dict[f'mean {prefix}'] = np.mean(clean_arr)
        features_dict[f'median {prefix}'] = np.median(clean_arr)
        features_dict[f'std {prefix}'] = np.std(clean_arr)
        features_dict[f'total duration {prefix}' if 'duration' not in prefix else f'total {prefix}'] = np.sum(clean_arr)
        features_dict[f'count {prefix}'] = len(clean_arr)
        features_dict[f'Q1 {prefix}'] = np.percentile(clean_arr, 25)
        features_dict[f'Q3 {prefix}'] = np.percentile(clean_arr, 75)

    process_and_stats(edit_burst_durations, "delete burst duration", features)
    process_and_stats(r_burst_durations, "r-burst duration", features)
    process_and_stats(p_burst_durations, "p-burst", features)
    process_and_stats(pause_before_delete, "pause before delete", features)
    process_and_stats(pause_inter_words, "pause inter words", features)
    process_and_stats(pause_inter_sentences, "pause inter sentences", features)

    for bin_name, vals in threshold_bins.items():
        process_and_stats(vals, bin_name, features)

    # --- Entropy features ---
    all_pauses = units_df['pause'].dropna()
    all_pauses = all_pauses[(all_pauses > 10) & (all_pauses < 100000)]

    if len(all_pauses):
        clean_pauses = replace_outliers_iqr_np(np.array(all_pauses).reshape(-1, 1)).flatten()
    else:
        clean_pauses = np.array([])

    features['pause entropy'] = pause_entropy(clean_pauses)

    if len(p_burst_durations):
        clean_p_bursts = replace_outliers_iqr_np(np.array(p_burst_durations).reshape(-1, 1)).flatten()
    else:
        clean_p_bursts = np.array([])

    features['p-burst entropy'] = pause_entropy(clean_p_bursts)

    return features

def feature_extracting_by_rolling_window(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        data = json.load(file)

    df = pd.DataFrame(data["keystrokes"])

    # 1. Parse the index: "1.2" -> Section "1", Question "2"
    # We convert to string first to ensure splitting works reliably
    df["q_str"] = df["question_index"].astype(str)
    df["section"] = df["q_str"].str.split('.').str[0]
    df["q_num"] = df["q_str"].str.split('.').str[1].astype(int)

    features = {}

    # 2. Identify all unique sections present (e.g., "1", "2")
    unique_sections = sorted(df["section"].unique(), key=lambda x: int(x))

    # 3. Define the specific question pairs you want within each section
    target_pairs = [(1, 4), (2, 5), (3, 6)]

    for sec in unique_sections:
        for q_a, q_b in target_pairs:
            # Filter rows: Must be the same section AND question number must be in the pair
            window_df = df[(df["section"] == sec) & (df["q_num"].isin([q_a, q_b]))]

            if not window_df.empty:
                # Create a clear label: SectionX_Questions_A_B
                window_label = f"S{sec}_Q{q_a}_{q_b}"

                # Extract features for this specific pair within this section
                features[f"Questions_{window_label}"] = extract_keystroke_features(window_df)

    return features


def extract_all_features_rolling(dataset_dir, session_files, output_csv, n_users=45):
    rows = []

    for i in range(1, n_users + 1):
        user_folder = os.path.join(dataset_dir, f"User{i}")

        for session_name, filename in session_files:
            file_path = os.path.join(user_folder, filename)

            if os.path.exists(file_path):
                print(f"Processing User {i} - {session_name}...")

                # Extract features for the (1,4), (2,5), (3,6) pairs per section
                feats = feature_extracting_by_rolling_window(file_path)

                for window_name, window_features in feats.items():
                    row = {
                        "user_id": f"user{i}",
                        "session": session_name,
                        "window": window_name
                    }

                    if isinstance(window_features, dict):
                        row.update(window_features)
                    else:
                        row["value"] = window_features

                    rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No data processed. Check if question indices match the X.Y format.")
        return df

    # Organize columns: metadata first, then alphabetical features
    meta_cols = ["user_id", "session", "window"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + sorted(feature_cols)]

    df.to_csv(output_csv, index=False)
    print(f"Extraction complete. Saved to {output_csv}")
    return df


def merge_normal_and_attack(df_normal, df_attack, output_csv):
    # Label = section: attack section_2 (paraphrase) -> section_4 (FP), section_3 (transcription) -> section_5 (FT)
    df_normal['section'] = 'section_'+df_normal['window'].str.extract(r'S(\d+)')
    df_attack['section'] = 'section_'+df_attack['window'].str.extract(r'S(\d+)')
    mapping = {
        'section_2': 'section_4',
        'section_3': 'section_5'
    }

    df_attack['section'] = df_attack['section'].replace(mapping)

    merged_df = pd.concat([df_normal, df_attack], ignore_index=True)
    merged_df.to_csv(output_csv, index=False)

    print(f"Files merged successfully! Output saved as '{output_csv}'.")
    print("\nAttack section counts (verifying rename):")
    print(merged_df[merged_df['section'].isin(['section_4', 'section_5'])]['section'].value_counts())
    return merged_df


REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "dataset" / "Attack4",
                        help="Folder with User<i>/ normal and *_attack.json files (default: dataset/Attack4).")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--include-phase2-attacks", action="store_true",
                        help="Also extract second_time_attack.json. The reported rhythmic results did NOT include it: "
                             "the original notebook requested 'second_time_.json', which does not exist, so only "
                             "phase-1 attack files were used (FP/FT rows = 45 files x 3 windows = 135 samples).")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    normal_files = [("session1", "first_time.json"), ("session2", "second_time.json")]
    attack_files = [("session1", "first_time_attack.json")]
    if args.include_phase2_attacks:
        attack_files.append(("session2", "second_time_attack.json"))

    df_normal = extract_all_features_rolling(args.dataset_dir, normal_files, args.output_dir / "users_window_features.csv")
    df_attack = extract_all_features_rolling(args.dataset_dir, attack_files, args.output_dir / "attack_window_features.csv")
    merge_normal_and_attack(df_normal, df_attack, args.output_dir / "merged_data.csv")


if __name__ == "__main__":
    main()
