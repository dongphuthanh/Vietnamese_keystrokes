# Detecting LLM-Assisted Vietnamese Writing via Keystrokes under Behavioral Manipulation

Code for the ICTAI 2026 paper. Keystroke logs of Vietnamese participants are classified into five writing modes:

| Label | Mode | Source |
|---|---|---|
| **B** | Bona fide composition | session 1 of each file |
| **P** | Paraphrasing an LLM response | session 2 |
| **T** | Transcribing an LLM response | session 3 |
| **FP** | Paraphrasing with manipulated typing behavior | adversarial version of session 2 |
| **FT** | Transcribing with manipulated typing behavior | adversarial version of session 3 |

Four modeling approaches are evaluated:
- temporal features + XGBoost;
- rhythmic features + XGBoost;
- a 1D-CNN;
- TypeNet (LSTM).

Each is evaluated under two settings:
- **user-independent (UIE):** train and test users are disjoint;
- **context-independent (CIE):** train and test questions are disjoint.

Each setting has four training configurations; testing always covers all five classes:

| Config | Classes seen in training |
|---|---|
| **M2** | B, T |
| **M3** | B, P, T |
| **M4** | B, P, T, FT |
| **M5** | B, P, T, FP, FT |

---

## Repository structure

```
src/
├── preprocessing/            raw keystroke logs -> cleaned, paired, indexed JSON
├── adversarial_generation/   cleaned JSON -> FP/FT (behaviorally manipulated) JSON
├── temporal_xgboost/         temporal features (KHT/KIT/RKDT) + XGBoost      (Temporal rows, Figs. 1-2)
├── rhythmic_features/        107 pause/burst/revision features + XGBoost     (Rhythmic rows, Figs. 1-2)
└── sequential_models/        1D-CNN and TypeNet, UIE and CIE                 (1D-CNN / TypeNet rows, Figs. 1-2)
    ├── optuna_studies/       saved hyperparameter searches used for the paper
    └── figures/              confusion matrices reported in the paper
```

## Setup

Tested with Python 3.13 on Windows with an NVIDIA GPU (CUDA). The sequential models also run on CPU, more slowly.

```bash
pip install numpy pandas scipy scikit-learn xgboost==3.0.5 deap optuna torch matplotlib
```

Versions used: torch 2.9, optuna 4.8, xgboost 3.0.5, scikit-learn 1.7.2, pandas 2.3, numpy 2.3, deap 1.4.

## Dataset

45 participants: all completed phase 1, and 30 also completed phase 2. Two downloads are available:

| Download | Contents | Place in |
|---|---|---|
| **[Attack4](https://drive.google.com/drive/folders/1NdT6VrEq26uF2DigYAlRVg1wT-xFlSHT?usp=sharing)** (used by all models) | Preprocessed normal files **and** the adversarial files used in the paper | `dataset/Attack4/` |
| [Preprocessed only](https://drive.google.com/drive/folders/1ZjGDdipbasBNizL3zXuuy9X_OFKBC2QT?usp=sharing) | Preprocessed normal files (B, P, T) | `dataset/viet_preprocessed/` |

**To run the models, download `Attack4`.** Every modeling pipeline reads it by default, and it is required to reproduce the paper's numbers exactly.

```
dataset/
└── Attack4/
    ├── User1/
    │   ├── first_time.json            phase 1: B, P, T (all 45 participants)
    │   ├── first_time_attack.json     phase 1: FP, FT
    │   ├── second_time.json           phase 2: B, P, T (30 participants)
    │   └── second_time_attack.json    phase 2: FP, FT
    ├── ...
    └── User45/
```

The normal files in `Attack4` are identical to those in the preprocessed-only download.

**File format.** Each file holds `{"keystrokes": [...]}`, one entry per keydown/keyup event:
- `key`, `code`, `event` (`keydown`/`keyup`), `timestamp` (ms);
- `question_index = "<session>.<question>"`: session 1 = B, 2 = P, 3 = T; questions 1-6;
- in the `*_attack.json` files, sessions 2 and 3 are FP and FT.

**Regenerating the adversarial files yourself (optional).** To build an `Attack4`-style folder from the preprocessed-only download:

```bash
cp -r dataset/viet_preprocessed dataset/Attack4_regenerated
python src/adversarial_generation/generate_attacks.py --input-dir dataset/viet_preprocessed --output-dir dataset/Attack4_regenerated
```

> The attack generator is seeded, but it is not the exact notebook revision that produced the adversarial files in `Attack4`. Regenerated FP/FT files follow the same procedure and parameters but are not byte-identical, so results computed on them differ slightly from the paper. To use them, point the pipelines at `Attack4_regenerated`:
> - temporal: `NORMAL_FOLDER` / `ATTACK_FOLDER` in `config.py`;
> - rhythmic: `--dataset-dir`;
> - sequential: `folder_path` in each script.

---

## Pipelines

Run each command from the folder shown, because several scripts use paths relative to their own folder.

### 1. Preprocessing — `src/preprocessing/`

Turns the raw web-logger output into the files in `viet_preprocessed/`. **You only need this for new raw data**; the download is already preprocessed.

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_clean_unikey.py` (+ `telex_map.py`) | Removes virtual keystrokes injected by the Unikey IME on Windows and maps composed Vietnamese characters back to TELEX keys. Also removes auto-repeated Backspace/Delete/Shift keydowns, arrow and Control keys, and duplicate IME events. |
| 2 | `step2_pair_keydown_keyup.py` | Pairs each keydown with its keyup (same `code`, within 10 events) and drops unpaired keydowns. |
| 3 | `step3_sort_by_timestamp.py` | Sorts events by timestamp. |
| 4 | `step4_merge_session_files.py` | Merges each participant's per-part files into one file per phase. |
| 5 | `step5_add_question_index.py` | Numbers questions in order of appearance (6 per session) and writes `session` and `question_index`. |

The scripts process every JSON file under a folder named `keystroke_sessions_only/` in the current directory (set by `folder_path` at the bottom of each script). They are run in order:

```bash
python step1_clean_unikey.py
python step2_pair_keydown_keyup.py
python step3_sort_by_timestamp.py
python step4_merge_session_files.py
python step5_add_question_index.py
```

> **These scripts overwrite the JSON files in place.** Run them on a copy of the raw data.

**Special case, User37.** This participant used a Japanese-layout keyboard, and their raw data came as three separate session files. It goes through a dedicated route over a folder `user_37/`:

```
step1_clean_unikey_user37 -> step2 -> step3 -> step4_merge_session_files_user37 -> step5
```

`check_key_event_counts.py` is a sanity check that compares keydown and keyup counts.

### 2. Adversarial generation — `src/adversarial_generation/`

Implements the threat model of Sec. III-B.3. For each participant file it takes the paraphrase (session 2) and transcription (session 3) keystrokes, and applies three transformations:
- **velocity scaling:** typing speed drifts, resampled at word boundaries;
- **revision injection:** delete 2-5 words and retype them, with realistic hesitation and re-planning pauses, at least twice per question;
- **boundary-aware pauses:** Weibull-distributed planning pauses at word, clause, sentence and line boundaries.

The results are written as `<name>_attack.json`. Each file is seeded (`random.Random(42)`), so runs are repeatable.

```bash
python src/adversarial_generation/generate_attacks.py [--input-dir DIR] [--output-dir DIR] [--n-users 45]
```

Defaults: `--input-dir dataset/viet_preprocessed`, `--output-dir dataset/generated_attacks`.

### 3. Temporal features + XGBoost — `src/temporal_xgboost/`

The repository already includes the exact train/test CSVs of the paper's run, and that run's results:

| Folder | Contents |
|---|---|
| `user_indep_datasets/`, `context_indep_datasets/` | the train/test CSVs of the paper's run |
| `runs_user_indep/`, `runs_context_indep/` | that run's results |

**To reproduce the paper's Temporal results,** skip feature extraction and train on the included CSVs:

```bash
cd src/temporal_xgboost
mv runs_user_indep runs_user_indep_paper          # step 4 skips runs that already have results
mv runs_context_indep runs_context_indep_paper
python run_all.py --start 4                       # train and evaluate on the included CSVs
```

Then compare the new `runs_*/summary.json` with the `*_paper` copies.

**To run the full pipeline from the raw JSON** (e.g. on other data):

```bash
python run_all.py                    # steps 1-5, both settings
python run_all.py --type user        # only user-independent (or --type context)
python run_all.py --gpu              # run XGBoost on the GPU
```

> **Running from step 1 overwrites the included CSVs**, and the regenerated ones give slightly different results (see the column-order note below). Keep a copy if you need the paper's exact numbers.

The data is read from `dataset/Attack4` by default. To use another folder, change `NORMAL_FOLDER` / `ATTACK_FOLDER` in `config.py`.

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `step1_extract.py` | Computes hold time (KHT) per key, and key interval (KIT) and release-to-keydown (RKDT) times for the 50 most frequent bigrams. Each is summarised by mean, std, std/mean, range, max and median, after discarding values outside [50, 5000] ms and a 2×IQR outlier filter. | `pkl/` |
| 2 | `step2_context_indep.py` | CIE split: question pairs {1,4}, {2,5}, {3,6}; each fold holds one pair out. | `context_indep_datasets/` |
| 3 | `step3_user_indep.py` | UIE split: 3-fold `KFold` over users (seed 42). | `user_indep_datasets/` |
| 4 | `step4_run_xgb.py` → `xgb_ga_classifier.py` | Keeps the top 50% of features by mutual information. **M5** tunes XGBoost with a genetic algorithm (DEAP; population 20, 10 generations, 5-fold CV); **M2-M4 reuse that fold's M5 parameters.** | `runs_*/xgb_<M>_fold<k>/results.json` |
| 5 | `step5_collect.py` | Sums the folds into combined confusion matrices, accuracy, F1, FAR and FRR. | `runs_*/summary.json`, `detailed_results.csv` |

Notes:
- Step 4 **skips** any run whose `results.json` already exists; move the `runs_*` folders aside to re-run.
- `xgb_ga_classifier.py` can also be run on its own for a single train/test CSV pair (`--help` for options).
- **Column order.** The feature columns of step 1 come out in a different order on each run (they are built from a Python `set`). The feature values are identical, but XGBoost's result depends on column order. This is why the original CSVs are included and why exact reproduction uses `--start 4`.

### 4. Rhythmic features + XGBoost — `src/rhythmic_features/`

```bash
cd src/rhythmic_features
python extract_rhythmic_features.py [--dataset-dir ../../dataset/Attack4] [--output-dir output] [--include-phase2-attacks]
python classify_user_independent.py
python classify_context_independent.py
```

**Feature extraction (`extract_rhythmic_features.py`).**
- **Features:** 107 per sample.
  - Durations of P-bursts, R-bursts and deletion bursts.
  - Pauses before deletions, between words and between sentences.
  - Nine pause-length bins.
  - Each of those 15 groups summarised by 7 statistics, plus pause entropy and P-burst entropy.
- **Sample unit:** one question pair ({1,4}, {2,5}, {3,6}) of one writing mode in one file.
- **Filtering:** pauses must lie in (10, 10000) ms, and outliers beyond 1.5×IQR are replaced by the median.
- **Output:** `output/users_window_features.csv` (B/P/T), `output/attack_window_features.csv` (FP/FT), and `output/merged_data.csv`, with `section_1`-`section_5` = B, P, T, FP, FT.
- **Phase-2 attacks:** by default only phase-1 attack files are used for FP/FT, as in the paper. Add `--include-phase2-attacks` to also use `second_time_attack.json`.

**Classification (`classify_*_independent.py`).**
- **Model and tuning:** XGBoost on the top 50% of features by mutual information, tuned with a randomized search followed by a local grid search, for every configuration and fold.
- **Folds:** UIE uses 3-fold `KFold` over users; CIE holds out one question pair per fold.
- **Scenario names:** S1-S4 in the output = M2-M5.
- **Output:** `output/xgb_user_kfold/` and `output/xgb_window_based/`, with `<scenario>_percentage_report.csv` and `<scenario>_cm.png` for each configuration.

> The rhythmic results reproduce the paper's figures approximately (typically within a few percentage points). The intermediate feature table of the original run was not preserved.

### 5. Sequential models: 1D-CNN and TypeNet — `src/sequential_models/`

| Script | Model | Setting | Hyperparameter study |
|---|---|---|---|
| `cnn_user_independent.py` | 1D-CNN | UIE | `optuna_studies/cnn_uie.db` |
| `cnn_context_independent.py` | 1D-CNN | CIE | `optuna_studies/cnn_cie.db` |
| `typenet_user_independent.py` | TypeNet | UIE | `optuna_studies/typenet_uie.db` |
| `typenet_context_independent.py` | TypeNet | CIE | `optuna_studies/typenet_cie.db` |

```bash
cd src/sequential_models
python cnn_user_independent.py M3            # configuration M2, M3, M4 or M5
python cnn_context_independent.py M3
python typenet_user_independent.py M3
python typenet_context_independent.py M3
```

If no configuration is given, the TypeNet CIE script runs M5 and the other three run M2.

**What they do.**
- **Windows:** keystroke sequences are cut into windows of 100 keydowns with stride 50. Each keystroke's key is mapped to one of 5 classes (letter, digit, space, backspace, other).
- **1D-CNN input:** log inter-keydown interval + key class. The architecture is a small projection / embedding, then 3 convolutional layers (kernel 3), global max pooling and a 2-layer classifier.
- **TypeNet input:** hold, inter-key, press-press and release-release latencies + key class, fed to 2 stacked LSTMs (128 units).
- **Prediction:** per window; window probabilities are averaged per session (mean pooling), and the class with the highest average is the session's prediction.
- **Evaluation:** 3 outer folds.
  - UIE: `KFold` over users.
  - CIE: one question pair held out.
- **Output:** per-fold and overall accuracy and EER printed to the console (the CIE scripts also print per-attack FAR); the confusion matrices saved to `figures/<model>_<uie|cie>_<M>_confusion_matrix.png`.

**Hyperparameters.** Only the learning rate is tuned, with Optuna (TPE), on an inner split of each outer fold's training data. Tuning is done **once on M5**, and the tuned value is reused for M2-M4. The study names are fixed to `…M5_fold<k>`.

- `OPTUNA_TRIALS = 0` (the default) loads the tuned values from `optuna_studies/` without new trials. This reproduces the paper's **M2-M4** exactly.
- The scripts do not reseed between tuning and final training. So **M5** reproduces exactly only when it runs together with its tuning trials:
  - point `storage=` to a new, empty database file;
  - set `OPTUNA_TRIALS` to 30 (TypeNet CIE: 10);
  - run M5.

A CUDA GPU is recommended. A full M2-M5 run of one script takes minutes (1D-CNN) to tens of minutes (TypeNet). Results are deterministic on the same hardware and software, but may differ slightly on a different GPU or CUDA version.

---

## Reproducibility summary

| Result (Figs. 1-2) | Reproduces |
|---|---|
| 1D-CNN, TypeNet — M2-M4 | exactly, from the saved Optuna studies |
| 1D-CNN, TypeNet — M5 | exactly, when re-run together with the tuning trials (see above) |
| Temporal (XGBoost) | exactly, with `run_all.py --start 4` on the included CSVs |
| Rhythmic (XGBoost) | approximately, within a few percentage points |

"Exactly" assumes the `Attack4` download. With attack files regenerated by `generate_attacks.py`, all FP/FT-related numbers differ slightly.
