"""M6 human-evaluation annotation app.

    streamlit run app.py

One card per VERSION: the whole conversation with every whisper of that version marked in
place, and per-whisper ratings (uptake_1to5, should_be_silent_0or1, optional note) beside
it. Versions are still met one at a time, in the sheet's pre-randomised order, under their
A/B/C labels. The focused whisper is highlighted and the turns after it are marked, since
levels 4-5 depend on what the user does next; a one-time acknowledgement per session
unlocks rating.

Keyboard: 1-5 uptake, 0/9 silent no/yes, <- -> previous/next whisper (crossing into the
neighbouring version at the ends), N note, S skip. The focus advances once a whisper has
both ratings. A script in the transcript panel forwards key presses to hidden command
buttons, so every shortcut runs the same Python as a click and is testable headlessly.
The sidebar shows % rated, active time (breaks over 5 min not counted) and an ETA.

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


KEYMAP = {"1": "u1", "2": "u2", "3": "u3", "4": "u4", "5": "u5", "0": "s0", "9": "s1",
          "ArrowLeft": "prev", "ArrowRight": "next", "n": "note", "N": "note",
          "s": "skip", "S": "skip"}


def render_html(body: str, focus: int, rated: set[int], scroll_top: bool) -> str:
    """The whole conversation for one version: every whisper marked in place, the focused
    one highlighted, the turns after it marked as what decides levels 4-5. The script
    (a) scrolls this panel to the focused whisper, (b) on a new version scrolls the page
    to the top, and (c) forwards keyboard shortcuts to the app's hidden command buttons.
    Every piece of transcript text is html.escape()d; the only markup is ours."""
    parts, pos, after = [], 0, False
    for m in WHISPER_RE.finditer(body):
        parts.append(_turns(body[pos:m.start()], after))
        n, text = int(m.group(1)), html.escape(m.group(2).strip())
        tick = " &#10003;" if n in rated else ""
        if n == focus:
            parts.append(f'<div id="cur" class="w cur">&#9658; WHISPER {n}: <b>{text}</b>{tick}'
                         '</div><div class="nexthdr">&#9660; what happened next &mdash; read '
                         'this before rating</div>')
            after = True
        else:
            parts.append(f'<div class="w other">whisper {n}: {text}{tick}</div>')
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
      .other{background:#f1f1f1;color:#666;font-size:13px}
      .nexthdr{color:#4a90d9;font-size:12px;font-weight:600;margin:4px 0}
      @media (prefers-color-scheme: dark){body{background:#0e1117;color:#ddd}
        .cur{background:#5a4b00;border-color:#e0a800}.other{background:#262730;color:#aaa}}
    </style>"""
    script = """<script>
    (function(){
      const c=document.getElementById('cur'); if(c){c.scrollIntoView({block:'center'});}
      let P; try{P=window.parent.document;}catch(e){return;}
      if(%(top)s){try{const m=P.querySelector('[data-testid="stMain"]')||P.querySelector('section.main');
        if(m){m.scrollTo({top:0});} window.parent.scrollTo(0,0);}catch(e){}}
      const MAP=%(map)s;
      function onKey(e){
        if(e.ctrlKey||e.metaKey||e.altKey) return;
        const t=e.target, tag=(t&&t.tagName||'').toLowerCase();
        if(tag==='input'||tag==='textarea'||(t&&t.isContentEditable)) return;
        const cmd=MAP[e.key]; if(!cmd) return;
        const btn=[...P.querySelectorAll('.st-key-kbd button')]
          .find(b=>b.innerText.trim()==='kbd:'+cmd);
        if(btn){e.preventDefault(); btn.click();}
      }
      // one live handler at a time: every rerun replaces this iframe
      if(window.parent.__m6kbd){P.removeEventListener('keydown',window.parent.__m6kbd);}
      window.parent.__m6kbd=onKey; P.addEventListener('keydown',onKey);
      document.addEventListener('keydown',onKey);
    })();
    </script>""" % {"top": "true" if scroll_top else "false", "map": json.dumps(KEYMAP)}
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
# App state
# =============================================================================
#
# A CARD is one version of one conversation: every whisper of that version, rated on one
# page. Cards follow the sheet's own order (dialogue_order, version_order), so versions
# are still met one at a time, in the pre-randomised sequence, under their A/B/C labels.
# The FOCUS is the whisper the keyboard acts on.

IDLE_CAP_S = 300          # a gap longer than this between ratings is a break, not work


def done(r: dict) -> bool:
    return r["uptake_1to5"].strip() in {"1", "2", "3", "4", "5"} and \
        r["should_be_silent_0or1"].strip() in {"0", "1"}


def build_cards(rows: list[dict]) -> list[list[int]]:
    cards: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(rows):
        cards.setdefault((int(r["dialogue_order"]), int(r["version_order"])), []).append(i)
    return [sorted(v, key=lambda i: int(rows[i]["whisper_index"])) for _, v in sorted(cards.items())]


def save(ann: str, sync: bool = False) -> None:
    """Flush the sheet and progress to disk now; mirror to GitHub when asked."""
    s = st.session_state
    sheet = to_csv_text(s.fields, s.rows)
    prog = json.dumps({"card": s.card, "focus": s.focus, "active_s": round(s.active_s, 1),
                       "timed_ratings": s.timed}, indent=1)
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
    s.cards = build_cards(s.rows)
    prog = load_progress(ann, s.store)
    first_open = next((c for c, idx in enumerate(s.cards)
                       if not all(done(s.rows[i]) for i in idx)), len(s.cards) - 1)
    s.card = min(int(prog.get("card", first_open)), len(s.cards) - 1)
    s.focus = _first_open_focus(s.card)
    s.active_s = float(prog.get("active_s", 0.0))
    s.timed = int(prog.get("timed_ratings", 0))
    s.last_event = None
    s.show_note = set()
    s.scroll_top = True
    s.ann = ann
    s.sync = ("local only -- no GitHub token configured" if s.store is None
              else f"GitHub mirror on (loaded from {origin})")


def mirror_widgets(idx: list[int]) -> None:
    """Streamlit drops a widget's state on any run where the widget is not drawn, so
    other cards' radios forget their values. The sheet is the truth: copy it into the
    widget keys just before a card is drawn (no default is passed, so this is legal)."""
    s = st.session_state
    for i in idx:
        r = s.rows[i]
        u, v = r["uptake_1to5"].strip(), r["should_be_silent_0or1"].strip()
        s[f"u_{i}"] = int(u) if u in {"1", "2", "3", "4", "5"} else None
        s[f"s_{i}"] = int(v) if v in {"0", "1"} else None
        s[f"n_{i}"] = r["note"]


def _first_open_focus(card: int) -> int:
    s = st.session_state
    idx = s.cards[card]
    return next((k for k, i in enumerate(idx) if not done(s.rows[i])), 0)


def _tick_clock() -> None:
    """Active time: gaps between ratings, with breaks over IDLE_CAP_S not counted."""
    s = st.session_state
    now = time.time()
    if s.last_event is not None and now - s.last_event <= IDLE_CAP_S:
        s.active_s += now - s.last_event
    s.last_event = now


def _rated(i: int, was_done: bool) -> None:
    """After a rating: count it for the ETA, and auto-advance once the whisper is done."""
    s = st.session_state
    _tick_clock()
    if done(s.rows[i]) and not was_done:
        s.timed += 1
        idx = s.cards[s.card]
        nxt = next((k for k in range(len(idx)) if k > idx.index(i) and not done(s.rows[idx[k]])),
                   None)
        if nxt is None:
            nxt = next((k for k in range(len(idx)) if not done(s.rows[idx[k]])), s.focus)
        s.focus = nxt


def set_rating(ann: str, i: int, col: str, value) -> None:
    s = st.session_state
    was = done(s.rows[i])
    s.rows[i][col] = "" if value is None else str(value)
    key = {"uptake_1to5": f"u_{i}", "should_be_silent_0or1": f"s_{i}"}[col]
    s[key] = value
    s.focus = s.cards[s.card].index(i)
    _rated(i, was)
    save(ann)


def on_widget(ann: str, i: int, col: str, key: str) -> None:
    set_rating(ann, i, col, st.session_state[key])


def on_note(ann: str, i: int) -> None:
    st.session_state.rows[i]["note"] = st.session_state[f"n_{i}"]
    save(ann)


def go_card(ann: str, card: int, focus_last: bool = False) -> None:
    s = st.session_state
    card = max(0, min(card, len(s.cards) - 1))
    if card != s.card:
        s.scroll_top = True
    s.card = card
    s.focus = len(s.cards[card]) - 1 if focus_last else _first_open_focus(card)
    save(ann, sync=True)


def command(ann: str, cmd: str) -> None:
    """Every keyboard shortcut lands here (via a hidden button), so it is testable."""
    s = st.session_state
    idx = s.cards[s.card]
    i = idx[s.focus]
    if cmd[0] in "us" and len(cmd) == 2 and cmd[1].isdigit():
        if not s.get("ack_ok"):
            return                               # rating is locked until acknowledged
        col = "uptake_1to5" if cmd[0] == "u" else "should_be_silent_0or1"
        set_rating(ann, i, col, int(cmd[1]))
    elif cmd in ("next", "skip"):
        if s.focus < len(idx) - 1:
            s.focus += 1
        elif s.card < len(s.cards) - 1:
            go_card(ann, s.card + 1)
    elif cmd == "prev":
        if s.focus > 0:
            s.focus -= 1
        elif s.card > 0:
            go_card(ann, s.card - 1, focus_last=True)
    elif cmd == "note":
        s.show_note ^= {i}


def fmt_dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h {sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"


# =============================================================================
# UI
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="M6 whisper rating", page_icon="📝", layout="wide")
    st.markdown("<style>.st-key-kbd{display:none}</style>", unsafe_allow_html=True)
    anns = annotators()
    if not anns:
        st.error("No annotator sheets found in data/.")
        return

    qp = st.query_params.get("annotator")
    with st.sidebar:
        st.markdown("### Who are you?")
        ann = st.selectbox("Annotator ID", anns, index=anns.index(qp) if qp in anns else None,
                           placeholder="choose your ID", key="ann_pick")
    if not ann:
        st.title("M6 whisper rating")
        st.info("Choose your annotator ID in the sidebar to begin. Your progress is "
                "saved after every click and restored when you come back.")
        return
    if st.session_state.get("ann") != ann:
        start(ann)
    s = st.session_state
    rows, cards = s.rows, s.cards
    idx = cards[s.card]
    total, n_done = len(rows), sum(map(done, rows))

    # hidden command buttons: the keyboard bridge clicks these
    with st.container(key="kbd"):
        for cmd in sorted(set(KEYMAP.values())):
            st.button(f"kbd:{cmd}", key=f"kbd_{cmd}", on_click=command, args=(ann, cmd))

    # ---- sidebar: progress, time, ETA, tools -------------------------------------
    with st.sidebar:
        pct = 100 * n_done / total
        st.progress(n_done / total, text=f"{pct:.0f}% rated · {n_done} of {total} whispers")
        rate = s.active_s / s.timed if s.timed else None
        eta = fmt_dur(rate * (total - n_done)) if rate and s.timed >= 10 else "after 10 ratings"
        st.caption(f"Active time {fmt_dur(s.active_s)} · ETA {eta}")
        n_conv = max(int(r["dialogue_order"]) for r in rows)
        st.caption(f"Version {s.card + 1} of {len(cards)} · conversation "
                   f"{rows[idx[0]]['dialogue_order']} of {n_conv}")
        nxt = next((c for c, ix in enumerate(cards) if not all(done(rows[i]) for i in ix)), None)
        if nxt is not None and nxt != s.card and st.button("Jump to first unrated",
                                                          use_container_width=True):
            go_card(ann, nxt)
            st.rerun()
        st.caption(f"Save state: {s.sync}")
        if s.store is not None and st.button("Sync now", use_container_width=True):
            save(ann, sync=True)
            st.rerun()
        st.download_button("Download my sheet (CSV)", to_csv_text(s.fields, rows),
                           file_name=f"sheet_{ann}.csv", mime="text/csv",
                           use_container_width=True)
        with st.expander("Keyboard shortcuts", expanded=False):
            st.markdown("`1`–`5` uptake · `0` / `9` silent no / yes · `←` `→` previous / next "
                        "whisper · `N` note · `S` skip.\n\nThe cursor moves to the next whisper "
                        "once both ratings are set. Shortcuts pause while you type a note.")
        with st.expander("Instructions"):
            st.markdown(
                "- You read each conversation several times, each time with a different "
                "**version** of the assistant's whispers. Rate each version on its own; do "
                "not compare versions.\n- **Read past each whisper** before rating it: "
                "levels 4 and 5 depend on what the user says and does next.\n- *Stayed "
                "silent*: 1 if the assistant would have done better to say nothing at that "
                "pause.\n- Rate every whisper. Notes are optional.")

    # ---- once per session: the read-past acknowledgement -----------------------------
    if not s.get("ack_ok"):
        with st.container(border=True):
            st.markdown("#### Before you start")
            st.markdown("Every rating depends on **what happens after** the whisper: levels 4 "
                        "and 5 mean the user went on to use it. On every card, the turns after "
                        "the highlighted whisper are marked in blue. **Read them before you "
                        "rate.**")
            st.checkbox("I understand I must read past each whisper before rating it",
                        key="ack_box", on_change=lambda: s.__setitem__("ack_ok", s.ack_box))

    card_rows = [rows[i] for i in idx]
    head = card_rows[0]
    st.markdown(f"**Conversation {head['dialogue_order']} of {n_conv} · Version "
                f"{head['version_order']} of 7 (label {head['version']})** · "
                f"{len(idx)} whisper{'s' if len(idx) > 1 else ''} on this card")

    left, right = st.columns([3, 2], gap="large")
    with left:
        focus_w = int(rows[idx[s.focus]]["whisper_index"])
        rated_w = {int(rows[i]["whisper_index"]) for i in idx if done(rows[i])}
        st.iframe(render_html(transcript_body(str(transcript_path(ann, head))), focus_w,
                              rated_w, s.scroll_top), height=620)
        s.scroll_top = False
    with right:
        with st.expander("Uptake rubric (LlamaPIE D.4.1)", expanded=s.card == 0):
            st.markdown("\n".join(f"- **{n}** {name} — {desc}" for n, name, desc in RUBRIC))
        locked = not s.get("ack_ok")
        mirror_widgets(idx)
        for k, i in enumerate(idx):
            r = rows[i]
            with st.container(border=True, key=f"w_{i}"):
                mark = "▶ " if k == s.focus else ""
                st.markdown(f"{mark}**Whisper {r['whisper_index']}**"
                            f"{' ✓' if done(r) else ''}")
                st.radio("Uptake", [1, 2, 3, 4, 5], key=f"u_{i}", horizontal=True,
                         format_func=lambda n: f"{n}", disabled=locked,
                         on_change=on_widget, args=(ann, i, "uptake_1to5", f"u_{i}"),
                         help=" · ".join(f"{n} {name}" for n, name, _ in RUBRIC))
                st.radio("Should have stayed silent?", [0, 1], key=f"s_{i}", horizontal=True,
                         format_func=lambda n: SILENCE_LABELS[n], disabled=locked,
                         on_change=on_widget, args=(ann, i, "should_be_silent_0or1", f"s_{i}"))
                if r["note"] or i in s.show_note:
                    st.text_input("Note (optional)", key=f"n_{i}", on_change=on_note,
                                  args=(ann, i))
                elif st.button("Add note", key=f"addnote_{i}"):
                    s.show_note.add(i)
                    st.rerun()
        b1, b2 = st.columns(2)
        if b1.button("◀ Previous version", disabled=s.card == 0, use_container_width=True):
            go_card(ann, s.card - 1)
            st.rerun()
        card_done = all(done(rows[i]) for i in idx)
        if b2.button("Next version ▶", disabled=s.card == len(cards) - 1,
                     use_container_width=True, type="primary" if card_done else "secondary"):
            go_card(ann, s.card + 1)
            st.rerun()
        if card_done and s.card < len(cards) - 1:
            st.caption("All whispers on this version are rated — press → or *Next version*.")
        if n_done == total:
            st.success("All whispers rated. Thank you! Download your sheet from the sidebar "
                       "and send it to the study organiser.")


if __name__ == "__main__":
    main()
