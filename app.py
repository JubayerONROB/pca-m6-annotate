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


def _n_rated(rows: list[dict]) -> int:
    return sum(1 for r in rows if r["uptake_1to5"].strip() or r["should_be_silent_0or1"].strip())


def load_work(ann: str, store: GitHubStore | None) -> tuple[list[str], list[dict], str]:
    """Source sheet, overlaid with the saved copy that holds the MOST work.

    Ratings can be added but never cleared, so the copy with more rated whispers is the
    newer one. That matters when a sheet is restored into the GitHub mirror while a
    container still holds an older local copy: local-first would load the stale copy and
    then sync it back over the restore. A tie goes to local (fresher notes, unsynced).
    """
    fields, src = read_csv_text((DATA / f"sheet_{ann}.csv").read_text(encoding="utf-8"))
    best: tuple[int, int, str, list[dict]] | None = None
    for rank, (origin, text) in enumerate(
            (("local", _read_local(OUT / f"sheet_{ann}.csv")),
             ("github", _read_remote(store, f"{REMOTE_DIR}/sheet_{ann}.csv")))):
        if not text:
            continue
        f2, saved = read_csv_text(text)
        if f2 != fields or _identity(saved, fields) != _identity(src, fields):
            st.error(f"Saved work ({origin}) does not match the source sheet for {ann}; "
                     "it was NOT loaded. Contact the study organiser before rating.")
            st.stop()
        cand = (_n_rated(saved), -rank, origin, saved)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return fields, [dict(r) for r in src], "new"
    return fields, best[3], best[2]


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


def parse_blocks(body: str) -> list[tuple]:
    """The transcript in reading order: ("turns", text) and ("whisper", n, text) blocks."""
    out, pos = [], 0
    for m in WHISPER_RE.finditer(body):
        out.append(("turns", body[pos:m.start()]))
        out.append(("whisper", int(m.group(1)), m.group(2).strip()))
        pos = m.end()
    out.append(("turns", body[pos:]))
    return out


# Theme-neutral (works in light and dark): translucent fills, no fixed text colours.
CONV_CSS = """<style>
.m6t{margin:2px 0;line-height:1.55}.m6s{font-weight:600}.m6p{opacity:.45;font-size:12px}
.m6after .m6t{border-left:3px solid #4a90d9;padding-left:8px}
.m6w{padding:6px 10px;border-radius:6px;background:rgba(128,128,128,.14)}
.m6w.cur{background:rgba(255,200,0,.25);border:2px solid #e0a800}
.m6next{color:#4a90d9;font-size:13px;font-weight:600;margin-top:2px}
.st-key-kbdjs{height:0;overflow:hidden}
</style>"""


def turns_html(chunk: str, after: bool) -> str:
    """Dialogue turns, escaped. Turns after the focused whisper are marked: they are
    what decides levels 4-5."""
    out = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line or line == "|SILENCE >":
            continue
        safe = html.escape(line).replace("|SILENCE &gt;", '<span class="m6p">[pause]</span>')
        m = re.match(r"^(User:|Speaker \d+:)(.*)$", safe)
        safe = f'<span class="m6s">{m.group(1)}</span>{m.group(2)}' if m else safe
        out.append(f'<div class="m6t">{safe}</div>')
    cls = "m6after" if after else "m6before"
    return f'<div class="{cls}">{"".join(out)}</div>' if out else ""


def kbd_script(scroll_top: bool, scroll_key: str | None) -> str:
    """A zero-height frame whose only job is to (a) scroll the page to the top on a new
    version, or to the focused whisper after a key press, and (b) forward keyboard
    shortcuts to the hidden command buttons. It holds no transcript text."""
    return """<script>
    (function(){
      let P; try{P=window.parent.document;}catch(e){return;}
      const top=%(top)s, key=%(key)s;
      setTimeout(function(){
        try{
          if(top){const m=P.querySelector('[data-testid="stMain"]')||P.querySelector('section.main');
                  if(m){m.scrollTo({top:0});} window.parent.scrollTo(0,0);}
          else if(key){const el=P.querySelector('.st-key-'+key);
                  if(el){el.scrollIntoView({block:'center',behavior:'smooth'});}}
        }catch(e){}
      }, 120);
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
      if(window.parent.__m6kbd){P.removeEventListener('keydown',window.parent.__m6kbd);}
      window.parent.__m6kbd=onKey; P.addEventListener('keydown',onKey);
    })();
    </script>""" % {"top": "true" if scroll_top else "false",
                    "key": json.dumps(scroll_key), "map": json.dumps(KEYMAP)}


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
            # Never overwrite a remote copy that holds MORE work than this session: a
            # restore, or another tab, got there first. Reload it instead of clobbering it.
            remote = s.store.read_text(f"{REMOTE_DIR}/sheet_{ann}.csv")
            if remote is not None:
                _, rrows = read_csv_text(remote)
                if _n_rated(rrows) > _n_rated(s.rows):
                    s.notice = ("A newer copy of your sheet was found and loaded. Your place "
                                "may have moved -- please check it before continuing.")
                    s.ann = None                   # main() calls start() on the next run
                    return
            sha = s.store.write_text(f"{REMOTE_DIR}/sheet_{ann}.csv", sheet,
                                     f"{ann}: {sum(map(done, s.rows))}/{len(s.rows)} rated")
            s.store.write_text(f"{REMOTE_DIR}/progress_{ann}.json", prog, f"{ann}: progress")
            s.sync = f"synced to GitHub ({sha}) at {time.strftime('%H:%M:%S')}"
        except Exception as exc:  # noqa: BLE001 -- keep working, report honestly
            s.sync = f"GitHub sync FAILED ({type(exc).__name__}); saved locally only"


def load_progress(ann: str, store: GitHubStore | None, origin: str) -> dict:
    """Progress from the SAME copy the sheet was loaded from, so the resume position can
    never point into a different (older) sheet."""
    text = (_read_local(OUT / f"progress_{ann}.json") if origin == "local" else
            _read_remote(store, f"{REMOTE_DIR}/progress_{ann}.json") if origin == "github"
            else None)
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
    prog = load_progress(ann, s.store, origin)
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
    s.scroll_focus = True
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
    if s.get("notice"):
        st.warning(s.pop("notice"))
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
                f"{len(idx)} whisper{'s' if len(idx) > 1 else ''} on this version")
    with st.expander("Uptake rubric (LlamaPIE D.4.1)", expanded=s.card == 0):
        st.markdown("\n".join(f"- **{n}** {name} — {desc}" for n, name, desc in RUBRIC))

    locked = not s.get("ack_ok")
    mirror_widgets(idx)
    by_w = {int(rows[i]["whisper_index"]): i for i in idx}
    focus_i = idx[s.focus]
    st.html(CONV_CSS + f"<style>.st-key-w_{focus_i}{{border-color:#e0a800 !important}}</style>")
    body = transcript_body(str(transcript_path(ann, head)))
    after = False
    for block in parse_blocks(body):
        if block[0] == "turns":
            h = turns_html(block[1], after)
            if h:
                st.html(h)
            continue
        n, text = block[1], block[2]
        i = by_w.get(n)
        if i is None:                      # a whisper with no sheet row: show, don't rate
            st.html(f'<div class="m6w">whisper {n}: {html.escape(text)}</div>')
            continue
        r, cur = rows[i], i == focus_i
        with st.container(border=True, key=f"w_{i}"):
            tail = ('</div><div class="m6next">&#9660; read what the user does next (below) '
                    'before rating</div>') if cur else "</div>"
            st.html(f'<div class="m6w{" cur" if cur else ""}">{"&#9658; " if cur else ""}'
                    f'WHISPER {n}: <b>{html.escape(text)}</b>{" &#10003;" if done(r) else ""}'
                    + tail)
            c1, c2 = st.columns([3, 2])
            c1.radio("Uptake", [1, 2, 3, 4, 5], key=f"u_{i}", horizontal=True,
                     disabled=locked, on_change=on_widget,
                     args=(ann, i, "uptake_1to5", f"u_{i}"),
                     help=" · ".join(f"{k} {name}" for k, name, _ in RUBRIC))
            c2.radio("Should have stayed silent?", [0, 1], key=f"s_{i}", horizontal=True,
                     format_func=lambda v: SILENCE_LABELS[v], disabled=locked,
                     on_change=on_widget, args=(ann, i, "should_be_silent_0or1", f"s_{i}"))
            if r["note"] or i in s.show_note:
                st.text_input("Note (optional)", key=f"n_{i}", on_change=on_note, args=(ann, i))
            elif st.button("Add note", key=f"addnote_{i}"):
                s.show_note.add(i)
                st.rerun()
        if cur:
            after = True

    with st.container(key="kbdjs"):
        st.iframe(kbd_script(s.scroll_top, f"w_{focus_i}" if s.get("scroll_focus") else None),
                  height=1)
    s.scroll_top = False
    s.scroll_focus = False

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
