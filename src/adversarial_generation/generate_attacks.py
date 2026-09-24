# -*- coding: utf-8 -*-
"""Generate behaviorally manipulated (adversarial) keystroke sequences, FP and FT (Sec. III-B.3).

For every participant file, the paraphrase (session 2) and transcription (session 3) keystrokes
are transformed with velocity scaling, revision injection and boundary-aware Weibull pauses,
and written as <name>_attack.json.

Originally a Colab notebook:
    https://colab.research.google.com/drive/1Z4Uo3AqYPcUu3HUGBd06R14WHGRC1Dk3

Note: this is the algorithm described in the paper, but not the exact notebook revision that produced
dataset/Attack4 (created 2026-03-30); re-running it gives similar but not identical attack files.

Usage:
    python generate_attacks.py [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import os
from pathlib import Path

import math
import random
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

# -----------------------------
# 1. DATA STRUCTURES
# -----------------------------

@dataclass
class KeypressUnit:
    code: str
    key: str
    hold_ms: int
    inter_ms: int
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PauseModel:
    """
    Weibull-based pause model with an optional long-tail mixture.

    Why Weibull here?
      - k (shape) > 1 produces a "rounded" unimodal distribution.
      - It avoids the very sharp "spiky" behavior you can get from lognormal
        when sigma is high, and it's easy to tune "peak vs spread".

    Parameters:
      shape_k:   Weibull shape (k). 2.0–3.0 typically yields rounded peaks.
      scale_lam: Weibull scale (λ). Controls overall pause magnitude.
      min_ms/max_ms: hard clamps for realism + safety.
      tail_p:    probability of triggering a "deep think" long-tail event.
      tail_mult: multiplier applied when the tail triggers.
    """
    shape_k: float
    scale_lam: float
    min_ms: int
    max_ms: int
    tail_p: float = 0.08
    tail_mult: float = 2.5


# -----------------------------
# 2. CONFIGURABLE PARAMETERS
# -----------------------------

# --- EDITION ENFORCEMENT ---
FORCE_AT_LEAST_NUM_EDITIONS = 2

# --- MICRO-RHYTHM PERTURBATION ---
# Simulates natural fluctuations in typing speed (velocity drift).
# 1.0 is neutral. 0.8 is 20% faster, 1.2 is 20% slower.
BURST_SPEED_RANGE = (0.75, 1.1)
SPEED_CHANGE_PROB = 0.15

# --- TEMPORAL BIAS ---
# You asked to avoid needing bias scaling (like 1.5).
# Keep this at 1.0. The pause distribution itself will be tuned to produce
# more rounded peaks + wider spread (plus rare long-tail pauses).
LONG_PAUSE_BIAS = 1.0

# --- [NEW] RE-TYPE SPEED FACTORS ---
# Simulates the "fluency" of re-typing something you already wrote.
RETYPE_PAUSE_FACTOR = 0.7     # Multiplier for pauses between keys (0.7 = 30% faster gaps)
RETYPE_HOLD_FACTOR = 0.85     # Multiplier for key hold duration (0.85 = 15% shorter hold)

# --- DELETION & ERROR CORRECTION ---
DELETION_PROB = 0.1           # Random chance (in addition to forced editions)
MIN_WORDS_TO_DELETE = 2       # Words to wipe out per block
MAX_WORDS_TO_DELETE = 5

HESITATION_RANGE = (600, 1400)
RETHINK_RANGE = (900, 2600)

# --- PAUSE CHANCES ---
PAUSE_CHANCES = {"WORD": 0.2, "CLAUSE": 0.45, "SENTENCE": 0.70, "LINE": 0.85}

# --- [NEW] MACRO-PAUSE SKIP CHECK ---
# If an inter-key gap is already "long", don't add ("fix") it by injecting an additional pause.
# This prevents inflating already-large pauses and keeps the transform from over-correcting.
SKIP_MACRO_IF_INTER_MS_GT = 1500


# -----------------------------
# 3. PAUSE MODELING (WEIBULL + TAIL MIX)
# -----------------------------

def make_weibull_model(
    mean_ms: float,
    shape_k: float,
    min_ms: int,
    max_ms: int,
    tail_p: float = 0.08,
    tail_mult: float = 2.5,
) -> PauseModel:
    """
    Builds a PauseModel by specifying the desired mean pause length (ms)
    and a Weibull shape k.

    Weibull mean:
        E[X] = λ * Γ(1 + 1/k)
    => λ = mean / Γ(1 + 1/k)
    """
    scale_lam = mean_ms / math.gamma(1.0 + 1.0 / shape_k)
    return PauseModel(
        shape_k=shape_k,
        scale_lam=scale_lam,
        min_ms=min_ms,
        max_ms=max_ms,
        tail_p=tail_p,
        tail_mult=tail_mult,
    )

def sample_pause(model: PauseModel, rng: random.Random) -> int:
    """
    Samples a "thinking pause" duration.

    Base: Weibull(k, λ) -> rounded peak
    Tail: With probability tail_p, multiply by tail_mult -> occasional deep planning
    Clamp: Enforces min/max for realism and safety.
    """
    pause = rng.weibullvariate(model.shape_k, model.scale_lam)

    # Rare long-tail "deep thinking" events
    if rng.random() < model.tail_p:
        pause *= model.tail_mult

    # Hard clamp
    pause = max(model.min_ms, min(model.max_ms, pause))
    return int(pause)


# --- PAUSE MODELS (TUNED FOR ROUNDED PEAK + WIDER SPREAD) ---
#
# These are tuned so you do NOT need LONG_PAUSE_BIAS=1.5.
# You can shift the "typical" pauses by adjusting mean_ms,
# and shape_k controls how peaked/spready the distribution feels.
PAUSE_MODELS = {
    # boundary:   mean_ms, k,    min, max,   tail_p, tail_mult
    "WORD":     make_weibull_model(750,  2.4, 150, 3000, tail_p=0.4, tail_mult=2.5),
    "CLAUSE":   make_weibull_model(900,  2.3, 250, 4500, tail_p=0.5, tail_mult=2.7),
    "SENTENCE": make_weibull_model(1400, 2.2, 350, 6000, tail_p=0.5, tail_mult=2.8),
    "LINE":     make_weibull_model(1750, 2.1, 450, 9000, tail_p=0.6, tail_mult=3),
}


# -----------------------------
# 4. UTILITIES & CONVERSION
# -----------------------------

def pair_to_units(events: List[Dict]) -> List[KeypressUnit]:
    """
    Converts raw keydown/keyup events into KeypressUnit objects.

    - hold_ms = keyup_ts - keydown_ts
    - inter_ms = next_keydown_ts - prev_keydown_ts
    """
    sorted_evs = sorted(events, key=lambda x: x["timestamp"])
    stacks: Dict[str, List[Dict]] = {}
    temp_units = []

    for e in sorted_evs:
        id_key = e.get("code") or e.get("key")

        if e["event"] == "keydown":
            stacks.setdefault(id_key, []).append(e)

        elif e["event"] == "keyup" and id_key in stacks and stacks[id_key]:
            down_event = stacks[id_key].pop(0)
            meta = {k: v for k, v in down_event.items() if k not in ["timestamp", "event"]}

            temp_units.append({
                "code": e.get("code") or id_key,
                "key": e.get("key", ""),
                "down_t": down_event["timestamp"],
                "up_t": e["timestamp"],
                "meta": meta
            })

    temp_units.sort(key=lambda x: x["down_t"])
    if not temp_units:
        return []

    units: List[KeypressUnit] = []
    prev_down = temp_units[0]["down_t"]
    for u in temp_units:
        hold = max(1, u["up_t"] - u["down_t"])
        inter = u["down_t"] - prev_down
        units.append(KeypressUnit(u["code"], u["key"], hold, inter, u["meta"]))
        prev_down = u["down_t"]

    return units

def units_to_events(units: List[KeypressUnit], start_ts: int) -> List[Dict]:
    """
    Converts KeypressUnits back to keydown/keyup events while preserving metadata.
    """
    out: List[Dict] = []
    curr_t = start_ts

    for u in units:
        curr_t += u.inter_ms

        for ev_type in ["keydown", "keyup"]:
            ev = copy.deepcopy(u.meta)
            ts = curr_t if ev_type == "keydown" else curr_t + u.hold_ms
            ev.update({"event": ev_type, "code": u.code, "key": u.key, "timestamp": int(ts)})
            out.append(ev)

    return sorted(out, key=lambda x: x["timestamp"])


# -----------------------------
# 5. TRANSFORMATION LOGIC
# -----------------------------

def transform_single_question(events: List[Dict], rng: random.Random) -> List[Dict]:
    """
    Main transform pass for a single question block.

    Effects:
      - Micro-rhythm velocity drift (burst-like speed changes)
      - Occasional macro "thinking pauses" at boundaries (Weibull + tail mix)
      - Forced + stochastic edit blocks (delete + re-type) with human-like timings
      - Re-typing is faster (RETYPE_PAUSE_FACTOR / RETYPE_HOLD_FACTOR)
      - [NEW] Skip macro pause injection when inter_ms is already > SKIP_MACRO_IF_INTER_MS_GT
    """
    if not events:
        return []

    original_start_ts = events[0]["timestamp"]
    units = pair_to_units(events)
    if not units:
        return events

    # Pre-calculate Forced Edition Indices (spaces are good edit anchor points)
    candidate_indices = []
    for idx, u in enumerate(units):
        if u.code == "Space" or u.key == " ":
            candidate_indices.append(idx)

    num_to_force = min(len(candidate_indices), FORCE_AT_LEAST_NUM_EDITIONS)
    forced_indices = set(rng.sample(candidate_indices, num_to_force)) if candidate_indices else set()

    final_sequence: List[KeypressUnit] = []
    word_indices = [0]

    # Current micro-rhythm multiplier
    current_velocity = rng.uniform(*BURST_SPEED_RANGE)

    for i, current_unit in enumerate(units):
        # --- PHASE 0: APPLY MICRO-RHYTHM ---
        # Perturb the inter_ms and hold_ms based on the current velocity
        current_unit.inter_ms = int(current_unit.inter_ms * current_velocity)
        current_unit.hold_ms = int(max(1, current_unit.hold_ms * current_velocity))

        final_sequence.append(current_unit)

        # --- Boundary Detection ---
        key, code = current_unit.key, current_unit.code
        boundary = None

        if code == "Space" or key == " ":
            boundary = "WORD"
        elif key in (".", "?", "!"):
            boundary = "SENTENCE"
        elif key in (",", ";", ":"):
            boundary = "CLAUSE"
        elif code == "Enter" or key.lower() in ("enter", "\r", "\n"):
            boundary = "LINE"

        # Change speed for the next "burst" at word boundaries
        if boundary == "WORD" and rng.random() < SPEED_CHANGE_PROB:
            current_velocity = rng.uniform(*BURST_SPEED_RANGE)

        # --- EDIT BLOCKS (DELETION + RE-TYPE) ---
        if boundary in ("WORD", "SENTENCE"):
            word_indices.append(len(final_sequence))
            should_delete = (i in forced_indices) or (rng.random() < DELETION_PROB)

            if should_delete and len(word_indices) > (MIN_WORDS_TO_DELETE + 1):
                num_words = rng.randint(
                    MIN_WORDS_TO_DELETE,
                    min(MAX_WORDS_TO_DELETE, len(word_indices) - 1)
                )

                start_idx = word_indices[-(num_words + 1)]
                block = final_sequence[start_idx:]  # portion to delete/retype

                # Realization phase: small pause before backspacing begins
                hesitation = int(rng.randint(*HESITATION_RANGE) * LONG_PAUSE_BIAS)

                # Backspacing: first backspace includes hesitation, subsequent are rapid
                for b_idx, target_unit in enumerate(reversed(block)):
                    bs_delay = (rng.randint(60, 110) + hesitation) if b_idx == 0 else rng.randint(60, 110)
                    bs_meta = copy.deepcopy(target_unit.meta)
                    bs_meta["actual"] = "synthetic"
                    final_sequence.append(
                        KeypressUnit(
                            "Backspace",
                            "Backspace",
                            rng.randint(40, 70),
                            bs_delay,
                            bs_meta
                        )
                    )

                # Resumption phase: pause before re-typing begins
                rethink = int(rng.randint(*RETHINK_RANGE) * LONG_PAUSE_BIAS)

                # Re-typing: faster + more confident
                for r_idx, u in enumerate(block):
                    new_u = copy.copy(u)
                    if r_idx == 0:
                        new_u.inter_ms = rethink
                    else:
                        new_u.inter_ms = int(max(20, u.inter_ms * RETYPE_PAUSE_FACTOR))
                    new_u.hold_ms = int(max(1, u.hold_ms * RETYPE_HOLD_FACTOR))
                    final_sequence.append(new_u)

                word_indices.append(len(final_sequence))

        # --- MACRO PLANNING PAUSES ---
        # Inject rounded + spread pauses using Weibull + tail mix.
        # No need for LONG_PAUSE_BIAS=1.5; the distribution handles it.
        #
        # [NEW] If the current inter_ms is already "long" (> 1500ms by default),
        # skip injecting another pause so we don't inflate already-large gaps.
        if boundary in PAUSE_CHANCES and rng.random() < PAUSE_CHANCES[boundary]:
            if current_unit.inter_ms <= SKIP_MACRO_IF_INTER_MS_GT:
                model = PAUSE_MODELS[boundary]
                current_unit.inter_ms += sample_pause(model, rng)

    return units_to_events(final_sequence, original_start_ts)

def transform_entire_dataset(events: List[Dict], seed: int = 42) -> List[Dict]:
    """
    Groups events by question_index and transforms each block independently.

    This ensures:
      - Forced edit blocks are fairly distributed per question
      - Timing remains consistent within each question block
    """
    rng = random.Random(seed)
    questions: Dict[Any, List[Dict]] = {}

    for e in events:
        qi = e.get("question_index", "default")
        questions.setdefault(qi, []).append(e)

    transformed: List[Dict] = []
    for qi in sorted(questions.keys()):
        transformed.extend(transform_single_question(questions[qi], rng))

    return transformed


"""
=== Burst Edition (Weibull Pause Models) ===

BURST_SPEED_RANGE:
    Controls micro-rhythm typing speed drift within bursts.

SPEED_CHANGE_PROB:
    Probability of changing typing speed at WORD boundaries.

=== TEMPORAL BIAS ===
LONG_PAUSE_BIAS:
    Kept at 1.0 by default.
    You asked not to rely on 1.5 global scaling; instead the pause distribution itself
    is tuned to produce rounded peaks + more spread + rare long-tail pauses.

=== RE-TYPE SPEED ===
RETYPE_PAUSE_FACTOR:
    Scales gaps between keys during re-typing (lower => faster).
RETYPE_HOLD_FACTOR:
    Scales key hold duration during re-typing (lower => shorter holds).

=== PAUSE CHANCES & MODELS ===
PAUSE_CHANCES:
    Probability (0.0 to 1.0) that a "thinking pause" is injected at boundaries.

PAUSE_MODELS (Weibull + Tail Mix):
    Base distribution: Weibull(k, λ)
      - k > 1 gives a smooth, rounded peak (more "human" than spiky bursts).
      - λ (scale) sets overall magnitude.

    Tail mixture:
      - tail_p: chance of triggering a long-tail event
      - tail_mult: multiplier when tail triggers
    This produces occasional deep planning pauses WITHOUT multiplying all pauses.

=== SKIP MACRO PAUSE WHEN ALREADY LONG ===
SKIP_MACRO_IF_INTER_MS_GT:
    If current_unit.inter_ms is already > this threshold (default 1500ms),
    we do NOT inject an additional macro "thinking pause" at that boundary.
    This avoids over-inflating already-long pauses.

=== DELETION & ERROR CORRECTION ===
FORCE_AT_LEAST_NUM_EDITIONS:
    Guarantees at least N deletion blocks per question (if long enough).

DELETION_PROB:
    Additional stochastic chance of a deletion happening at a boundary.

MIN_WORDS_TO_DELETE / MAX_WORDS_TO_DELETE:
    Controls the size of the deletion window (phrase-level edits).

HESITATION_RANGE:
    Gap before first Backspace (noticing an error).

RETHINK_RANGE:
    Gap after finishing deletion before re-typing (planning rephrase).

=== DATA INTEGRITY & STRUCTURE ===
METADATA PRESERVATION:
    Original metadata is preserved; synthetic Backspaces inherit metadata and mark:
        meta["actual"] = "synthetic"

QUESTION_INDEX GROUPING:
    Transformation resets for each unique 'question_index' block.
"""

REPO_ROOT = Path(__file__).resolve().parents[2]


def transform_keystrokes(file_path, output_path):
  if not os.path.exists(file_path):
    return
  with open(file_path, encoding="utf8") as f:
    data=json.load(f)
  copy_keystrokes = [
    item for item in data["keystrokes"]
    if (
        "question_index" in item
        and isinstance(item["question_index"], str)
        and item["question_index"].split(".")[0] in ["2","3"]
    )
  ]
  temp={}
  temp["keystrokes"]=transform_entire_dataset(copy_keystrokes)
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  with open(output_path, 'w', encoding="utf8") as f:
    json.dump(
        temp,
        f,
        indent=4,          # controls indentation level
        ensure_ascii=False # keeps Unicode characters readable
    )
  return


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "dataset" / "viet_preprocessed",
                        help="Folder with User<i>/first_time.json and second_time.json (default: dataset/viet_preprocessed).")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dataset" / "generated_attacks",
                        help="Where User<i>/<name>_attack.json are written (default: dataset/generated_attacks). "
                             "Kept separate so the attack files used in the paper (dataset/Attack4) are never overwritten.")
    parser.add_argument("--n-users", type=int, default=45)
    args = parser.parse_args()

    for i in range(1, args.n_users + 1):
        print(i)
        for name in ("first_time", "second_time"):
            transform_keystrokes(str(args.input_dir / f"User{i}" / f"{name}.json"),
                                 str(args.output_dir / f"User{i}" / f"{name}_attack.json"))


if __name__ == "__main__":
    main()
