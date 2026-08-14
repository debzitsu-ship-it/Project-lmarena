"""
Feed the model data from the web: download URLs, strip HTML, save as .txt
into my_ai/data/raw/ so prepare_data picks them up.

TRANSPARENCY: this is a plain downloader (urllib + a tiny HTML-tag stripper).
It does NOT call any AI. It is how you "feed data from the browser":
give it the URLs of pages/articles/books you want the model to learn from.

Usage:
  python -m my_ai.data.ingest_urls https://example.com/page1 https://example.com/page2
  python -m my_ai.data.ingest_urls --list urls.txt        # one URL per line

Note: works on your own machine / Colab. (Outbound internet is blocked in
some sandboxes.) Respect websites' terms and copyright: public-domain and
your-own content are always safe to train on.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.request
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "header", "footer", "nav", "aside"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0:
            self.parts.append(data)


def fetch_url(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (my_ai data ingester)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", errors="replace")
    if "<html" in raw[:2000].lower() or "<body" in raw[:5000].lower():
        p = TextExtractor()
        p.feed(raw)
        text = "".join(p.parts)
    else:
        text = raw  # already plain text
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text))
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--list", help="file with one URL per line")
    ap.add_argument("--out", default="my_ai/data/raw")
    ap.add_argument("--min-chars", type=int, default=500)
    args = ap.parse_args()

    urls = list(args.urls)
    if args.list:
        with open(args.list) as f:
            urls += [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not urls:
        ap.error("give URLs as arguments or via --list")

    os.makedirs(args.out, exist_ok=True)
    ok = 0
    for url in urls:
        try:
            text = fetch_url(url)
        except Exception as e:
            print(f"SKIP {url}: {e}", file=sys.stderr)
            continue
        if len(text) < args.min_chars:
            print(f"SKIP {url}: only {len(text)} chars after cleaning", file=sys.stderr)
            continue
        name = "web_" + hashlib.sha1(url.encode()).hexdigest()[:12] + ".txt"
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# source: {url}\n\n{text}")
        print(f"OK   {url} -> {path} ({len(text):,} chars)")
        ok += 1
    print(f"\n{ok}/{len(urls)} pages saved. Next: python -m my_ai.prepare_data --input {args.out}")


if __name__ == "__main__":
    main()
