# Data collection web app

The browser app used to collect the keystroke data (paper Sec. III-B.1). Participants answer 18 questions in three writing sessions:

| Session | Mode | What the participant does |
|---|---|---|
| 1 | **B**, bona fide | Answers 6 questions on their own |
| 2 | **P**, paraphrase | Asks ChatGPT, pastes its answer, then paraphrases it |
| 3 | **T**, transcription | Asks ChatGPT, pastes its answer, then retypes it |

Every keydown and keyup in the participant's own writing fields is logged with a millisecond timestamp.

The same app and the same questions were used for both study phases. Each participant completed it once per phase, at least two weeks apart.

| File | Contents |
|---|---|
| `index.html` | Page layout: language choice, study introduction, participant information form |
| `script.js` | All logic: questions and instructions (English and Vietnamese), session flow, keystroke logging, file downloads |
| `style.css` | Styling |

It is a static page with no server or database, and nothing is uploaded: all data is saved to the participant's own computer as downloaded files. `index.html` loads the p5.js library, but the app does not use it, so the page also runs without p5.js.

---

## Running it

### Option A: p5.js web editor (how the study was run)

1. Go to **https://editor.p5js.org** and sign in, so the sketch can be saved and shared.
2. Create a new sketch. It starts with `index.html`, `sketch.js` and `style.css`.
3. Replace the contents of `index.html` and `style.css` with the files from this folder.
4. Rename `sketch.js` to **`script.js`** (or create a new file with that name) and paste in `script.js` from this folder. `index.html` loads `script.js`, so the name must match.
5. Save, and press Play to test it in the preview.
6. To send it to participants, use the editor's **Share** menu and give them the **Full Screen** (or Present) link, not the editor link.

### Option B: open it locally

Keep the three files in one folder and open `index.html` in a browser. To serve it instead:

```bash
python -m http.server 8000
```

Then open http://localhost:8000. The same files also work on any static host, such as GitHub Pages.

### Browser notes

- **Browser:** use a desktop browser (Chrome or Edge recommended) on the participant's own laptop, so they type on the keyboard they normally use.
- **Downloads:** the app saves files at the end of each session. The browser may ask to **allow multiple downloads**, and participants must accept, otherwise later files are lost.
- **Don't reload:** reloading or closing the tab mid-session loses that session's keystrokes, because nothing is saved until the session's "Next Session" / "Submit" button is pressed.

---

## What participants see

1. **Language:** English or Vietnamese. This sets the interface and the language of the questions. Answers are written in Vietnamese either way.
2. **Introduction:** study purpose, the three sessions, and the evaluation criteria (grammar, relevance, length).
3. **Participant information:**

   | Field | Required | Stored as |
   |---|---|---|
   | Gender | yes | `gender` |
   | Age (5–90) | yes | `age` |
   | Handedness | yes | `handedness` |
   | Vietnamese proficiency | no | `vietnameseProficiency` |
   | Education level | no | `educationLevel` |

4. **Session 1: Bona fide writing.** One text box per question (90–150 words). Copy and paste are disabled.
5. **Session 2: Paraphrasing ChatGPT.** Two boxes per question:
   - the ChatGPT answer, **pasted in** (paste allowed, keystrokes **not** logged);
   - the participant's paraphrase, **typed** (paste disabled, keystrokes logged).
6. **Session 3: Retyping ChatGPT.** Same two boxes: the pasted ChatGPT answer, then the participant's retyped copy (logged).

A session cannot be finished until every box has text. The word counter is only a guide; the 90–150-word limit is not enforced by the app.

---

## Output files

Each run downloads **five files** to the participant's computer:

| File | Downloaded when | Contents |
|---|---|---|
| `keystrokes_session1.json` | leaving session 1 | keystrokes of session 1 (B) |
| `keystrokes_session2.json` | leaving session 2 | keystrokes of session 2 (P) |
| `keystrokes_session3.json` | on Submit | keystrokes of session 3 (T) |
| `responses_data.json` | on Submit | final text of every answer, pasted ChatGPT answers and word counts |
| `UserInformation.json` | on Submit | the participant information form |

The final screen asks participants to email the files to the research team, and has buttons to download the last two again. The address is set in `showThankYouMessage()` in `script.js`; see the next section.

**Keystroke record** (one per keydown or keyup, in `{"keystrokes": [...]}`):

```json
{ "session": 1, "question": "<question text>", "key": "a", "code": "KeyA",
  "event": "keydown", "timestamp": 1752426382879, "repeat": false }
```

- `key` is the character produced (with Unikey/TELEX this can be a composed Vietnamese letter);
- `code` is the physical key;
- `timestamp` is `Date.now()` in milliseconds;
- `repeat` is true for keyboard auto-repeat.

Only the participant's own writing fields are logged; the pasted ChatGPT fields are not.

---

## From downloads to the dataset

The models use the keystroke files only. `responses_data.json` was used for the response validation described in the paper (lexicon check, similarity to the ChatGPT text, LLM-based quality scoring). That validation code is not part of this repository.

1. Arrange each participant's keystroke files by phase:

   ```
   keystroke_sessions_only/
   └── User<i>/
       ├── first_time/     keystrokes_session1.json  keystrokes_session2.json  keystrokes_session3.json
       └── second_time/    keystrokes_session1.json  keystrokes_session2.json  keystrokes_session3.json
   ```

2. Run the preprocessing pipeline in [`../src/preprocessing/`](../src/preprocessing/), **on a copy**, because it rewrites files in place:

   ```
   step1_clean_unikey -> step2_pair_keydown_keyup -> step3_sort_by_timestamp -> step4_merge_session_files -> step5_add_question_index
   ```

   - Step 4 merges each phase folder into `User<i>/first_time.json` / `second_time.json`.
   - Step 5 numbers the questions (`question_index = "<session>.<question>"`).

   The result has the layout of `dataset/viet_preprocessed/` (see the main [README](../README.md)).

---

## Adapting it for a new study

All of these are in `script.js`:

| To change | Where |
|---|---|
| Questions | `session1Questions`, `session2Questions`, `session3Questions` (one list per language; keep 6 per session, since preprocessing step 5 assumes 6) |
| Session instructions | `sessionInstructions` |
| Where participants send their files | the email address in `showThankYouMessage()` |
| Introduction text, form labels, translations | `translateContent()` (and `index.html` for the English defaults) |

The introduction ends with a note that serious violations of the evaluation criteria could reduce the participation payment. Update or remove it (`violation-note` in `translateContent()`) to match your own study protocol.

**Privacy.** The downloaded files contain everything participants wrote, and the keystroke files let that text be reconstructed. Collect, store and share them only as your ethics approval (IRB) allows.
