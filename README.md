# M6 human evaluation — whisper rating app

A small Streamlit app for rating proactive-assistant "whispers" in conversations. Each
of five annotators rates every whisper in their assigned conversations on two items:

- **uptake (1–5)** — LlamaPIE's five-point rubric: did the user go on to use it?
- **should the system have stayed silent here? (0/1)**

plus an optional note. **This repository is private**: the conversations come from the
LlamaPIE dataset, and ratings are saved into the repo, where they should not be visible
to other annotators before they finish.

## For annotators

1. Open the app link you were given (or run it locally, below).
2. Choose **your** annotator ID (`ann1` … `ann5`) in the sidebar. You can also open
   `…/?annotator=ann3` to skip this step.
3. For each card:
   - Read the conversation on the left. The whisper you are rating is highlighted in
     yellow; the turns after it are marked in blue.
   - **Read past the whisper first.** Levels 4 and 5 depend on what the user says and does
     *next*, so the rating buttons stay locked until you tick
     *"I have read past this whisper"*.
   - Choose an uptake level and a silence answer. Add a note if you want.
   - Press **Next ▶**.
4. You will read each conversation several times, each time with a different
   **version** of the whispers. Rate each version on its own; do not compare versions.

Everything saves the moment you click, so closing the tab or refreshing loses nothing;
the app reopens at the card where you left off. The sidebar shows your progress, a
*Jump to first unrated* button, and **Download my sheet (CSV)**. When you finish,
download your sheet and send it to the study organiser.

Please do not discuss your ratings with the other annotators until everyone is done.

## Run locally

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Ratings are written to `annotations/sheet_annN.csv` (and `annotations/progress_annN.json`
for the resume position) next to `app.py`.

## Deploy on Streamlit Community Cloud

1. https://share.streamlit.io → **Create app** → *Deploy a public app from GitHub* →
   choose this repository, branch `main`, main file **`app.py`**. (Streamlit Cloud can deploy
   a private repo once you grant it access to your GitHub account; the app itself can be
   shared by link.)
2. **Advanced settings → Secrets** — paste:

   ```toml
   [github]
   token  = "github_pat_..."
   repo   = "JubayerONROB/pca-m6-annotate"
   branch = "main"
   ```

   Use a **fine-grained** personal access token scoped to *this repository only*, with
   **Contents: Read and write**. Streamlit Cloud's disk is wiped on every restart; with the
   token, every annotator's work is also committed to `annotations/` in this repo, so a
   restart costs nothing. Without it the app still runs, but saved work lives only until
   the container restarts — annotators should then download their sheet regularly.
3. Deploy and send each annotator the link, e.g. `https://<app>.streamlit.app/?annotator=ann1`.

## What is in here

```
app.py                       the app
requirements.txt
data/sheet_annN.csv          blank sheets, one per annotator (never written by the app)
data/transcripts_annN/       the conversations each annotator reads, one file per version
annotations/                 created at run time: sheet_annN.csv + progress_annN.json
```

The app is **blind by construction**: it reads only `data/`. The blinding key that maps
version letters to systems is not in this repository and the app has no code path that
could open it. It also never shows which conversations are shared between annotators.

## For the study organiser: scoring

The exported `annotations/sheet_annN.csv` has exactly the source sheet's columns and rows;
only `uptake_1to5`, `should_be_silent_0or1` and `note` are filled. To score:

1. Copy the five returned sheets over the blank ones in the thesis repo's
   `results/human_eval/` (the blinding key and `items.json` stay as they are).
2. `python analysis/m6/score_human_eval.py` — it refuses to unblind if the key or
   `items.json` differ from their pre-registered hashes.

The blank sheets shipped in `data/` are byte-identical to the ones pinned in the thesis
repo's `results/human_eval/PREREGISTRATION.json` (pre-registration v2, 2026-09-30):

| sheet | sha256 (first 16) |
|---|---|
| sheet_ann1.csv | `3df3625c807dc129` |
| sheet_ann2.csv | `8492a1483b08c28c` |
| sheet_ann3.csv | `5e4564576dfda21d` |
| sheet_ann4.csv | `791c0f2cbc858050` |
| sheet_ann5.csv | `80f30e9fe51668f1` |
