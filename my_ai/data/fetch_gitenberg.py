"""
Download public-domain books from GITenberg (Project Gutenberg mirrored on
GitHub) via codeload tarballs. Works even where gutenberg.org is blocked.

Plain HTTP download + tar extraction -- no AI involved. Strips the Gutenberg
license header/footer so only the actual book text is kept.

Usage:
  python -m my_ai.data.fetch_gitenberg               # download default book list
  python -m my_ai.data.fetch_gitenberg --out DIR
"""
from __future__ import annotations

import argparse
import io
import os
import re
import tarfile
import urllib.request

# repo-name -> gutenberg id (title_id naming convention of GITenberg)
BOOKS = {
    "Alices-Adventures-in-Wonderland_11": 11,
    "Through-the-Looking-Glass_12": 12,
    "The-Adventures-of-Tom-Sawyer_74": 74,
    "Adventures-of-Huckleberry-Finn_76": 76,
    "Frankenstein--Or-The-Modern-Prometheus_84": 84,
    "A-Tale-of-Two-Cities_98": 98,
    "Treasure-Island_120": 120,
    "Emma_158": 158,
    "Sense-and-Sensibility_161": 161,
    "The-Picture-of-Dorian-Gray_174": 174,
    "The-Call-of-the-Wild_215": 215,
    "A-Christmas-Carol-in-Prose--Being-a-Ghost-Story-of-Christmas_46": 46,
    "Dracula_345": 345,
    "The-Time-Machine_35": 35,
    "The-War-of-the-Worlds_36": 36,
    "The-Wonderful-Wizard-of-Oz_55": 55,
    "Peter-Pan_16": 16,
    "The-Jungle-Book_236": 236,
    "The-Secret-Garden_113": 113,
    "Anne-of-Green-Gables_45": 45,
    "Black-Beauty_271": 271,
    "The-Adventures-of-Sherlock-Holmes_1661": 1661,
    "The-Hound-of-the-Baskervilles_2852": 2852,
    "Pride-and-Prejudice_1342": 1342,
    "Great-Expectations_1400": 1400,
    "Oliver-Twist_730": 730,
    "David-Copperfield_766": 766,
    "Moby-Dick--Or-The-Whale_2701": 2701,
    "Grimms-Fairy-Tales_2591": 2591,
    "Andersen-s-Fairy-Tales_1597": 1597,
    "Aesop-s-Fables_11339": 11339,
    "The-Strange-Case-of-Dr.-Jekyll-and-Mr.-Hyde_43": 43,
    "Around-the-World-in-80-Days_103": 103,
    "Twenty-Thousand-Leagues-under-the-Sea_164": 164,
    "The-Legend-of-Sleepy-Hollow_41": 41,
    "Little-Women_514": 514,
    "Heidi_1448": 1448,
    "The-Railway-Children_1874": 1874,
    "Just-So-Stories_2781": 2781,
    "The-Wind-in-the-Willows_289": 289,
    # extra volume for the bigger Fix-2 model
    "Jane-Eyre-An-Autobiography_1260": 1260,
    "Wuthering-Heights_768": 768,
    "The-Count-of-Monte-Cristo_1184": 1184,
    "Les-Miserables_135": 135,
    "War-and-Peace_2600": 2600,
    "Anna-Karenina_1399": 1399,
    "Crime-and-Punishment_2554": 2554,
    "The-Brothers-Karamazov_28054": 28054,
    "A-Study-in-Scarlet_244": 244,
    "The-Sign-of-the-Four_2097": 2097,
    "The-Return-of-Sherlock-Holmes_108": 108,
    "The-Memoirs-of-Sherlock-Holmes_834": 834,
    "The-Innocents-Abroad_3176": 3176,
    "The-Prince-and-the-Pauper_1837": 1837,
    "A-Connecticut-Yankee-in-King-Arthur-s-Court_86": 86,
    "The-Mysterious-Island_1268": 1268,
    "Journey-to-the-Center-of-the-Earth_18857": 18857,
    "From-the-Earth-to-the-Moon--and-a-trip-around-it_83": 83,
    "The-Invisible-Man--A-Grotesque-Romance_5230": 5230,
    "The-Island-of-Doctor-Moreau_159": 159,
    "The-First-Men-in-the-Moon_1013": 1013,
    "Gulliver-s-Travels_829": 829,
    "Robinson-Crusoe_521": 521,
    "The-Three-Musketeers_1257": 1257,
    "White-Fang_910": 910,
    "The-Sea-Wolf_1074": 1074,
    "Martin-Eden_1056": 1056,
    "The-Scarlet-Letter_25344": 25344,
    "The-Adventures-of-Pinocchio_500": 500,
    "The-Merry-Adventures-of-Robin-Hood_964": 964,
}

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.S | re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*", re.S | re.I)


def strip_license(text: str) -> str:
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def fetch_book(repo: str, gid: int, out_dir: str) -> bool:
    dest = os.path.join(out_dir, f"gutenberg_{gid}.txt")
    if os.path.exists(dest):
        return True
    url = f"https://codeload.github.com/GITenberg/{repo}/tar.gz/master"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "my_ai/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            # prefer utf-8 file (-0.txt), fall back to plain id.txt
            names = tf.getnames()
            candidates = [n for n in names if n.endswith(f"{gid}-0.txt")] or \
                         [n for n in names if n.endswith(f"{gid}.txt")] or \
                         [n for n in names if n.endswith(".txt")]
            if not candidates:
                return False
            raw = tf.extractfile(candidates[0]).read().decode("utf-8", errors="replace")
        text = strip_license(raw)
        if len(text) < 10_000:
            return False
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"  FAIL {repo}: {type(e).__name__} {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="my_ai/data/raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for repo, gid in BOOKS.items():
        ok = fetch_book(repo, gid, args.out)
        if ok:
            size = os.path.getsize(os.path.join(args.out, f"gutenberg_{gid}.txt"))
            total += size
            print(f"  OK {repo} ({size/1e6:.2f} MB)")
    print(f"\ncorpus books total: {total/1e6:.1f} MB in {args.out}")


if __name__ == "__main__":
    main()
