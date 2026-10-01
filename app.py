"""M6 human-evaluation annotation app.

    streamlit run app.py

One card per whisper. Each card shows the whole conversation -- including everything
said AFTER the whisper -- with the whisper under evaluation highlighted, the five-point
uptake rubric inline, and two ratings: uptake_1to5 and should_be_silent_0or1, plus an
optional note.

BLIND BY CONSTRUCTION. This app reads only data/sheet_annN.csv and
data/transcripts_annN/. It never reads KEY_do_not_open_until_scoring.csv (the key is not
in this repo and there is no code path that opens it), and it never shows which system
produced a version or which conversations are shared across annotators.

DURABLE. Every change is flushed the moment it is made, through a temp-file rename, so
a refresh or crash costs nothing. Work goes to annotations/sheet_annN.csv -- the source
sheet in data/ is never written. On Streamlit Community Cloud, whose disk does not
survive a restart, the same file is also mirrored into this GitHub repo when a token is
configured in the app's secrets (see README).

The exported sheet has exactly the source sheet's columns and rows, with only
uptake_1to5, should_be_silent_0or1 and note filled, so the returned files score with
analysis/m6/score_human_eval.py unchanged.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "annotations"
REMOTE_DIR = "annotations"
RATING_COLS = ("uptake_1to5", "should_be_silent_0or1", "note")

RUBRIC = [
    (1, "Not relevant / not used", "unrelated; the user ignored it and never referenced it"),
    (2, "Relevant but redundant", "made sense, but the user had already covered it"),
    (3, "Relevant but not acted on", "useful and on-topic, but the user did not respond to it"),
    (4, "Relevant, used later", "helpful; the user acted on it only in a later turn"),
    (5, "Highly relevant, used now", "exactly what was needed; used in the very next turn"),
]
SILENCE_LABELS = {0: "0 — speaking here was fine", 1: "1 — should have stayed silent"}


# =============================================================================
# Storage: atomic local write, optional GitHub mirror (same design as pca-annotate)
# =============================================================================


def write_local(path: Path, text: str) -> None:
    """Atomic write: a crash mid-save must never truncate earlier work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


@dataclass
class GitHubStore:
    token: str
    repo: str
    branch: str = "main"
    _sha: dict[str, str] = field(default_factory=dict)

    def _request(self, method: str, path: str, payload: dict | None = None):
        req = urllib.request.Request(
            f"https://api.github.com{path}", method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization": f"Bearer {self.token}",
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28",
                     "User-Agent": "pca-m6-annotate",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)

    def read_text(self, path: str) -> str | None:
        try:
            data = self._request("GET", f"/repos/{self.repo}/contents/{path}?ref={self.branch}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        self._sha[path] = data.get("sha", "")
        return base64.b64decode(data["content"]).decode("utf-8")

    def write_text(self, path: str, text: str, message: str) -> str:
        payload = {"message": message, "branch": self.branch,
                   "content": base64.b64encode(text.encode("utf-8")).decode("ascii")}
        if self._sha.get(path):
            payload["sha"] = self._sha[path]
        try:
            data = self._request("PUT", f"/repos/{self.repo}/contents/{path}", payload)
        except urllib.error.HTTPError as exc:
            if exc.code not in (409, 422):        # stale sha: re-read once and retry
                raise
            self.read_text(path)
            payload.pop("sha", None)
            if self._sha.get(path):
                payload["sha"] = self._sha[path]
            data = self._request("PUT", f"/repos/{self.repo}/contents/{path}", payload)
        self._sha[path] = data["content"]["sha"]
        return data["commit"]["sha"][:7]


def store_from_secrets() -> GitHubStore | None:
    cfg: dict[str, str] = {}
    try:
        section = st.secrets.get("github", None)
        if section:
            cfg = {k: str(v) for k, v in dict(section).items()}
    except Exception:                       # no secrets file at all: local mode
        cfg = {}
    for k in ("token", "repo", "branch"):
        cfg.setdefault(k, os.environ.get(f"GITHUB_{k.upper()}", ""))
    if not cfg.get("token") or not cfg.get("repo"):
        return None
    return GitHubStore(cfg["token"], cfg["repo"], cfg.get("branch") or "main")


# =============================================================================
# Data
# =============================================================================


def annotators() -> list[str]:
    return sorted(p.stem.replace("sheet_", "") for p in DATA.glob("sheet_*.csv"))


def read_csv_text(text: str) -> tuple[list[str], list[dict]]:
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), [dict(r) for r in reader]


def to_csv_text(fields: list[str], rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _identity(rows: list[dict], fields: list[str]) -> list[tuple]:
    """Every non-rating column, row by row: what must never change."""
    keep = [f for f in fields if f not in RATING_COLS]
    return [tuple(r.get(f, "") for f in keep) for r in rows]


def load_work(ann: str, store: GitHubStore | None) -> tuple[list[str], list[dict], str]:
    """Source sheet, overlaid with saved work (local first, then the GitHub mirror)."""
    fields, src = read_csv_text((DATA / f"sheet_{ann}.csv").read_text(encoding="utf-8"))
    for origin, text in (("local", _read_local(OUT / f"sheet_{ann}.csv")),
                         ("github", _read_remote(store, f"{REMOTE_DIR}/sheet_{ann}.csv"))):
        if not text:
            continue
        f2, saved = read_csv_text(text)
        if f2 != fields or _identity(saved, fields) != _identity(src, fields):
            st.error(f"Saved work ({origin}) does not match the source sheet for {ann}; "
                     "it was NOT loaded. Contact the study organiser before rating.")
            st.stop()
        return fields, saved, origin
    return fields, [dict(r) for r in src], "new"


def _read_local(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _read_remote(store: GitHubStore | None, path: str) -> str | None:
    if store is None:
        return None
    try:
        return store.read_text(path)
    except Exception:
        return None


WHISPER_RE = re.compile(r"\n?[ \t]*>>> WHISPER (\d+): (.*?) <<<[ \t]*\n?")


@st.cache_data(show_spinner=False)
def transcript_body(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    sep = "=" * 78
    return text.split(sep, 1)[1].strip("\n") if sep in text else text


def transcript_path(ann: str, row: dict) -> Path:
    d, v, item = int(row["dialogue_order"]), int(row["version_order"]), row["item_id"]
    return DATA / f"transcripts_{ann}" / f"{d:02d}-{v}-{item}.txt"


def render_html(body: str, current: int) -> str:
    """The conversation with whisper `current` highlighted and everything after it
    marked as the part that decides levels 4-5."""
    parts, pos, after = [], 0, False
    for m in WHISPER_RE.finditer(body):
        parts.append(_turns(body[pos:m.start()], after))
        n, text = int(m.group(1)), html.escape(m.group(2).strip())
        if n == current:
            parts.append(f'<div id="cur" class="w cur">&#9658; WHISPER {n} (rate this one): '
                         f'<b>{text}</b></div><div class="nexthdr">&#9660; what happened next '
                         '&mdash; read this before rating</div>')
            after = True
        else:
            parts.append(f'<div class="w other">whisper {n}: {text}</div>')
        pos = m.end()
    parts.append(_turns(body[pos:], after))
    css = """<style>
      body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:15px;line-height:1.5;
           margin:0;padding:8px 12px;color:#222;background:#fff}
      .t{margin:2px 0}.spk{font-weight:600}
      .after .t{border-left:3px solid #4a90d9;padding-left:8px}
      .p{color:#aaa;font-size:12px}
      .w{margin:8px 0;padding:6px 10px;border-radius:6px}
      .cur{background:#fff3b0;border:2px solid #e0a800}
      .other{background:#f1f1f1;color:#777;font-size:13px}
      .nexthdr{color:#4a90d9;font-size:12px;font-weight:600;margin:4px 0}
      @media (prefers-color-scheme: dark){body{background:#0e1117;color:#ddd}
        .cur{background:#5a4b00;border-color:#e0a800}.other{background:#262730;color:#999}}
    </style>"""
    script = ("<script>const c=document.getElementById('cur');"
              "if(c){c.scrollIntoView({block:'center'});}</script>")
    return css + "".join(parts) + script


def _turns(chunk: str, after: bool) -> str:
    out = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line or line == "|SILENCE >":
            continue
        safe = html.escape(line).replace("|SILENCE &gt;", '<span class="p">[pause]</span>')
        m = re.match(r"^(User:|Speaker \d+:)(.*)$", safe)
        safe = f'<span class="spk">{m.group(1)}</span>{m.group(2)}' if m else safe
        out.append(f'<div class="t">{safe}</div>')
    return f'<div class="{"after" if after else "before"}">{"".join(out)}</div>'


# =============================================================================
# App
# =============================================================================


def done(r: dict) -> bool:
    return r["uptake_1to5"].strip() in {"1", "2", "3", "4", "5"} and \
        r["should_be_silent_0or1"].strip() in {"0", "1"}


def save(ann: str, sync: bool = False) -> None:
    """Flush the sheet and progress to disk now; mirror to GitHub when asked."""
    s = st.session_state
    sheet = to_csv_text(s.fields, s.rows)
    prog = json.dumps({"position": s.pos, "read_past": sorted(s.read_past)}, indent=1)
    write_local(OUT / f"sheet_{ann}.csv", sheet)
    write_local(OUT / f"progress_{ann}.json", prog)
    if sync and s.store is not None:
        try:
            sha = s.store.write_text(f"{REMOTE_DIR}/sheet_{ann}.csv", sheet,
                                     f"{ann}: {sum(map(done, s.rows))}/{len(s.rows)} rated")
            s.store.write_text(f"{REMOTE_DIR}/progress_{ann}.json", prog, f"{ann}: progress")
            s.sync = f"synced to GitHub ({sha}) at {time.strftime('%H:%M:%S')}"
        except Exception as exc:  # noqa: BLE001 -- keep working, report honestly
            s.sync = f"GitHub sync FAILED ({type(exc).__name__}); saved locally only"


def load_progress(ann: str, store: GitHubStore | None) -> dict:
    for text in (_read_local(OUT / f"progress_{ann}.json"),
                 _read_remote(store, f"{REMOTE_DIR}/progress_{ann}.json")):
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
    return {}


def start(ann: str) -> None:
    s = st.session_state
    s.store = store_from_secrets()
    s.fields, s.rows, origin = load_work(ann, s.store)
    prog = load_progress(ann, s.store)
    s.read_past = set(prog.get("read_past", [])) | {i for i, r in enumerate(s.rows) if done(r)}
    first_open = next((i for i, r in enumerate(s.rows) if not done(r)), len(s.rows) - 1)
    s.pos = min(int(prog.get("position", first_open)), len(s.rows) - 1)
    s.ann = ann
    s.sync = ("local only -- no GitHub token configured" if s.store is None
              else f"GitHub mirror on (loaded from {origin})")


def set_field(ann: str, i: int, col: str, key: str) -> None:
    v = st.session_state[key]
    st.session_state.rows[i][col] = "" if v is None else str(v)
    save(ann)


def set_read(ann: str, i: int, key: str) -> None:
    (st.session_state.read_past.add if st.session_state[key]
     else st.session_state.read_past.discard)(i)
    save(ann)


def go(ann: str, to: int) -> None:
    s = st.session_state
    s.pos = max(0, min(to, len(s.rows) - 1))
    save(ann, sync=True)


def main() -> None:
    st.set_page_config(page_title="M6 whisper rating", page_icon="📝", layout="wide")
    anns = annotators()
    if not anns:
        st.error("No annotator sheets found in data/.")
        return

    qp = st.query_params.get("annotator")
    with st.sidebar:
        st.markdown("### Who are you?")
        default = anns.index(qp) if qp in anns else None
        ann = st.selectbox("Annotator ID", anns, index=default,
                           placeholder="choose your ID", key="ann_pick")
    if not ann:
        st.title("M6 whisper rating")
        st.info("Choose your annotator ID in the sidebar to begin. Your progress is "
                "saved after every click and restored when you come back.")
        return
    if st.session_state.get("ann") != ann:
        start(ann)
    s = st.session_state
    rows, i = s.rows, s.pos
    r = rows[i]
    total = len(rows)
    n_done = sum(map(done, rows))
    n_conv = max(int(x["dialogue_order"]) for x in rows)

    with st.sidebar:
        st.progress(n_done / total, text=f"{n_done} of {total} whispers rated")
        convs_done = sum(all(done(x) for x in rows if x["dialogue_order"] == d)
                         for d in {x["dialogue_order"] for x in rows})
        st.caption(f"Conversations fully rated: {convs_done} of {n_conv}")
        nxt = next((j for j in range(total) if not done(rows[j])), None)
        if nxt is not None and st.button("Jump to first unrated", use_container_width=True):
            go(ann, nxt)
            st.rerun()
        st.caption(f"Save state: {s.sync}")
        if s.store is not None and st.button("Sync now", use_container_width=True):
            save(ann, sync=True)
            st.rerun()
        st.download_button("Download my sheet (CSV)", to_csv_text(s.fields, rows),
                           file_name=f"sheet_{ann}.csv", mime="text/csv",
                           use_container_width=True)
        with st.expander("Instructions"):
            st.markdown(
                "- You will read each conversation several times, each time with a "
                "different **version** of the assistant's whispers. Rate each version "
                "on its own; do not compare versions.\n"
                "- **Read past the whisper.** Levels 4 and 5 are about whether the user "
                "went on to use it, which you can only judge from what comes next.\n"
                "- *Should have stayed silent*: 1 if the assistant would have done better "
                "to say nothing at that pause.\n"
                "- Rate every whisper. The note is optional.")

    st.markdown(f"**Conversation {r['dialogue_order']} of {n_conv} · Version "
                f"{r['version_order']} of 7 (label {r['version']}) · Whisper "
                f"{r['whisper_index']} of {r['n_whispers']}** · card {i + 1} of {total}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        # Every transcript string is html.escape()d in render_html; the only markup is ours.
        st.iframe(render_html(transcript_body(str(transcript_path(ann, r))),
                              int(r["whisper_index"])), height=560)
    with right:
        with st.container(border=True):
            st.markdown("**Uptake rubric (LlamaPIE D.4.1)**")
            st.markdown("\n".join(f"- **{n}** {name} — {desc}" for n, name, desc in RUBRIC))
        key_rp = f"rp_{ann}_{i}"
        st.session_state.setdefault(key_rp, i in s.read_past)
        st.checkbox("I have read **past** this whisper — what the user said and did next",
                    key=key_rp, on_change=set_read, args=(ann, i, key_rp))
        locked = i not in s.read_past
        if locked:
            st.caption("Read the conversation after the highlighted whisper, then tick the "
                       "box to rate it.")
        up = r["uptake_1to5"].strip()
        key_u = f"u_{ann}_{i}"
        st.radio("Uptake (1–5)", [1, 2, 3, 4, 5], key=key_u, horizontal=True,
                 index=int(up) - 1 if up in {"1", "2", "3", "4", "5"} else None,
                 format_func=lambda n: f"{n} · {RUBRIC[n - 1][1]}", disabled=locked,
                 on_change=set_field, args=(ann, i, "uptake_1to5", key_u))
        si = r["should_be_silent_0or1"].strip()
        key_s = f"s_{ann}_{i}"
        st.radio("Should the system have stayed silent here?", [0, 1], key=key_s,
                 index=int(si) if si in {"0", "1"} else None,
                 format_func=lambda n: SILENCE_LABELS[n], disabled=locked,
                 on_change=set_field, args=(ann, i, "should_be_silent_0or1", key_s))
        key_n = f"n_{ann}_{i}"
        st.session_state.setdefault(key_n, r["note"])
        st.text_input("Note (optional)", key=key_n,
                      on_change=set_field, args=(ann, i, "note", key_n))
        if done(r):
            st.success("Saved.")
        b1, b2 = st.columns(2)
        if b1.button("◀ Previous", disabled=i == 0, use_container_width=True):
            go(ann, i - 1)
            st.rerun()
        if b2.button("Next ▶", disabled=i == total - 1, use_container_width=True,
                     type="primary" if done(r) else "secondary"):
            go(ann, i + 1)
            st.rerun()
        if not done(r):
            st.caption("This whisper is not fully rated yet — you can come back to it.")
        if n_done == total:
            st.success("All whispers rated. Thank you! Download your sheet from the sidebar "
                       "and send it to the study organiser.")


if __name__ == "__main__":
    main()
