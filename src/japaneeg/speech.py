"""Speech measures from the released events.

* :func:`utterance_rates`: articulation rate of every overt utterance from the released phone
  intervals (forced alignment): phones whose midpoint lies inside the speech event, divided by
  their summed duration, so pauses do not count. Utterances with a flagged phone
  (``alignment_flag``) or without phones are skipped. Optionally the words of the utterance
  are read as katakana with pyopenjtalk and their morae counted (contracted small kana do not
  count; the sokuon does); utterances with an unknown-word phone (``spn``) get no mora rate.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from pathlib import Path


from japaneeg.bids import read_tsv

NON_PHONES: frozenset[str] = frozenset({"", "sil", "sp", "spn", "n/a"})
KATAKANA = re.compile(r"[ァ-ヺー]")
SMALL_KANA: frozenset[str] = frozenset("ァィゥェォャュョヮヵヶ")


def count_morae(text: str) -> int:
    """Number of morae of a Japanese text read by pyopenjtalk.

    Args:
        text: Japanese text (kanji, kana or mixed).

    Returns:
        Mora count (0 if the text has no reading).
    """
    import pyopenjtalk

    kana: str = pyopenjtalk.g2p(text, kana=True)
    return sum(1 for ch in KATAKANA.findall(kana) if ch not in SMALL_KANA)


def utterance_rates(events_tsv: str | Path, morae: bool = False) -> list[dict[str, str]]:
    """Articulation rate of every overt utterance of one ``events.tsv``.

    Args:
        events_tsv: A released ``*_events.tsv``.
        morae: Also count the morae of the aligned words (needs pyopenjtalk).

    Returns:
        One row per utterance: participant, file, segment, onset, phone count, phonation time
        (s), phone rate (1/s) and, with ``morae``, mora count and mora rate.
    """
    path = Path(events_tsv)
    rows = read_tsv(path)
    events: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        if r.get("trial_type") == "overt":
            events[r.get("frag_id", "") or ""].append((float(r["onset"]), float(r["duration"])))
    if not events:
        return []
    for spans in events.values():
        spans.sort()
    starts = {k: [s for s, _ in v] for k, v in events.items()}

    def utterance_of(r: dict[str, str]) -> tuple[str, int] | None:
        key = r.get("frag_id", "") or ""
        if key not in events:
            return None
        mid = float(r["onset"]) + float(r["duration"]) / 2
        k = bisect.bisect_right(starts[key], mid) - 1
        if k < 0 or mid >= events[key][k][0] + events[key][k][1]:
            return None
        return key, k

    count: dict[tuple[str, int], int] = defaultdict(int)
    time: dict[tuple[str, int], float] = defaultdict(float)
    flagged: set[tuple[str, int]] = set()
    unknown: set[tuple[str, int]] = set()
    words: dict[tuple[str, int], list[tuple[float, str]]] = defaultdict(list)
    for r in rows:
        kind = r.get("trial_type", "")
        if kind == "word" and morae:
            utt = utterance_of(r)
            if utt is not None:
                words[utt].append((float(r["onset"]), r.get("value", "").strip()))
            continue
        if kind != "phone":
            continue
        label = r.get("value", "").strip()
        utt = utterance_of(r)
        if utt is None:
            continue
        if label == "spn":
            unknown.add(utt)
        if label in NON_PHONES:
            continue
        if (r.get("alignment_flag", "") or "").strip() not in ("", "n/a"):
            flagged.add(utt)
        count[utt] += 1
        time[utt] += float(r["duration"])
    m = re.search(r"sub-(\d+)", path.name)
    pid = f"P{m.group(1)}" if m else "unknown"
    out = []
    for utt, n in count.items():
        if utt in flagged or time[utt] <= 0:
            continue
        row = {"participant": pid, "file": path.name, "frag_id": utt[0], "onset": f"{events[utt[0]][utt[1]][0]:.4f}",
               "n_phones": str(n), "phonation_s": f"{time[utt]:.4f}", "rate": f"{n / time[utt]:.4f}",
               "n_morae": "", "mora_rate": ""}  # fmt: skip
        if morae and utt not in unknown and words[utt]:
            text = "".join(w for _, w in sorted(words[utt]) if w not in NON_PHONES)
            n_morae = count_morae(text) if text else 0
            if n_morae > 0:
                row["n_morae"] = str(n_morae)
                row["mora_rate"] = f"{n_morae / time[utt]:.4f}"
        out.append(row)
    return out
