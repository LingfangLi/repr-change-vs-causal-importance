"""Lexicon-based selection quality for the Llama-3 sentiment words (reviewer pt.2).

Direct check that the model-selected words ARE sentiment words: how many of them
appear in an authoritative sentiment lexicon (VADER + Hu&Liu Opinion Lexicon),
and whether their lexicon polarity matches the sentence label.

Words are recovered from the corrupted CSV (clean words masked out = selected).
Coverage is a LOWER BOUND: film-review words (e.g. "seductive", "tedious",
"heartfelt") and context-dependent picks are often absent from generic lexicons,
yet are clearly sentiment-bearing (listed under `missed` for the appendix).

Run:  python lexicon_precision.py
"""
import re
from pathlib import Path
import pandas as pd

SP = Path("<LEXICON_DIR>")  # directory holding vader_lexicon.txt, hl_pos.txt, hl_neg.txt
PROJ = Path("<PROJECT_ROOT>")
WORD_RE = r"[A-Za-z][A-Za-z'-]+"

# --- load lexicons ---
POS, NEG = set(), set()
for line in open(SP / "vader_lexicon.txt", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        try:
            s = float(p[1])
        except ValueError:
            continue
        if s >= 0.5:
            POS.add(p[0].lower())
        elif s <= -0.5:
            NEG.add(p[0].lower())


def load_hl(fn, dst):
    f = SP / fn
    if not f.exists():
        return
    for l in open(f, encoding="latin-1"):
        l = l.strip()
        if l and not l.startswith(";"):
            dst.add(l.lower())


load_hl("hl_pos.txt", POS)
load_hl("hl_neg.txt", NEG)
LEX = POS | NEG
print(f"lexicon: {len(POS)} pos + {len(NEG)} neg = {len(LEX)} sentiment words\n")


def selected_words(clean, corrupted):
    cw = [w for w in re.findall(WORD_RE, clean) if len(w) > 2]
    xw = {w.lower() for w in re.findall(WORD_RE, corrupted) if len(w) > 2}
    return [w for w in cw if w.lower() not in xw]


rows = []
for task in ["sst2", "yelp"]:
    d = pd.read_csv(PROJ / f"output/corrupted_data/{task}_corrupted.csv")
    total = cov = polok = 0
    hits, missed = [], []
    for _, r in d.iterrows():
        y = 1 if str(r["label"]).strip().lower() in ("positive", "1") else 0
        for w in selected_words(str(r["clean"]), str(r["corrupted"])):
            total += 1
            wl = w.lower()
            if wl in LEX:
                cov += 1
                wp, wn = wl in POS, wl in NEG
                if (y == 1 and wp) or (y == 0 and wn):
                    polok += 1
                if len(hits) < 10:
                    hits.append(f"{w}({'+' if y else '-'})")
            elif len(missed) < 10:
                missed.append(f"{w}({'+' if y else '-'})")
    rows.append({"task": task, "selected_words": total,
                 "in_lexicon_%": round(100 * cov / total, 1),
                 "polarity_correct_%": round(100 * polok / total, 1),
                 "polarity_correct_given_in_lex_%": round(100 * polok / max(cov, 1), 1)})
    print(f"=== {task} ===")
    print(f"  selected words: {total}")
    print(f"  in lexicon (coverage, lower bound): {100*cov/total:.0f}%")
    print(f"  polarity correct (of all)         : {100*polok/total:.0f}%")
    print(f"  polarity correct | in lexicon     : {100*polok/max(cov,1):.0f}%")
    print(f"  hits   : {', '.join(hits)}")
    print(f"  missed : {', '.join(missed)}   <- lexicon gaps, still sentiment-bearing")
    print()

pd.DataFrame(rows).to_csv(PROJ / "experiments/corruption_semantic_check/results/lexicon_precision.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
