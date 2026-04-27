"""Shared helpers: load PKL, label assignment, cognitive level parsing."""

import pickle
from typing import Optional

import pandas as pd
import numpy as np


def assign_label(row) -> int:
    """
    Labels:
      normal / session 1 → 0 (Bonafide)
      normal / session 2 → 1 (Paraphrase)
      normal / session 3 → 2 (Transcribe)
      attack / session 2 → 3 (Fake Paraphrase)
      attack / session 3 → 4 (Fake Transcribe)
    """
    try:
        sess = int(row["session"])
    except Exception:
        return -1
    src = row.get("source", "")
    if src == "normal":
        return {1: 0, 2: 1, 3: 2}.get(sess, -1)
    if src == "attack":
        return {2: 3, 3: 4}.get(sess, -1)
    return -1


def extract_cognitive_level(qi) -> Optional[int]:
    """
    question_index like '1.3', 2.5, etc.
    Returns the part after the dot (1-6), or None if invalid.
    """
    try:
        s = str(qi).strip()
        if "." not in s:
            return None
        level = int(s.split(".")[-1])
        return level if 1 <= level <= 6 else None
    except Exception:
        return None


def load_pkl_as_df(pkl_path, source_name: str) -> pd.DataFrame:
    """Load a PKL file produced by step1, add source/label/cognitive_level columns."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(0)
    df["source"]          = source_name
    df["label"]           = df.apply(assign_label, axis=1)
    df["cognitive_level"] = df["question_index"].apply(extract_cognitive_level)
    return df


def drop_meta_cols(df: pd.DataFrame, extra: list = None) -> pd.DataFrame:
    """Drop metadata columns that XGB.py should not see as features."""
    always_drop = ["source", "cognitive_level", "question_index", "file"]
    to_drop = always_drop + (extra or [])
    return df.drop(columns=[c for c in to_drop if c in df.columns])
