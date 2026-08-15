"""
Download real, openly licensed Python source code as training data, via
GitHub codeload tarballs (works where other hosts are blocked).

Plain HTTP + tar extraction -- no AI involved. We take .py files from
well-known permissively licensed projects (MIT/BSD/PSF), skip tests and
generated files, and concatenate them into training documents.

This is Phase-1/2 "coding diet": the model learns Python SYNTAX and
structure (def/class/if/for, indentation, docstrings) the same way it
learns English from books.

Usage: python -m my_ai.data.fetch_python_code [--out my_ai/data/raw]
"""
from __future__ import annotations

import argparse
import io
import os
import tarfile
import urllib.request

# repo -> ref. Small-to-medium, permissively licensed, mostly pure Python.
REPOS = {
    "pallets/flask": "main",
    "pallets/click": "main",
    "pallets/jinja": "main",
    "pallets/werkzeug": "main",
    "psf/requests": "main",
    "pytest-dev/pytest": "main",
    "pypa/packaging": "main",
    "more-itertools/more-itertools": "master",
    "python-attrs/attrs": "main",
    "agronholm/exceptiongroup": "main",
}

SKIP_DIRS = ("/tests/", "/test/", "/docs/", "/examples/", "/.github/",
             "/benchmarks/", "/scripts/")


def fetch_repo(repo: str, ref: str, out_dir: str, max_bytes: int = 2_500_000) -> int:
    owner_name = repo.replace("/", "_")
    dest = os.path.join(out_dir, f"code_{owner_name}.txt")
    if os.path.exists(dest):
        return os.path.getsize(dest)
    url = f"https://codeload.github.com/{repo}/tar.gz/{ref}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "my_ai/1.0"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
    except Exception as e:
        print(f"  FAIL {repo}: {type(e).__name__} {e}")
        return 0
    chunks: list[str] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf.getmembers():
            n = member.name
            if not n.endswith(".py") or any(s in n for s in SKIP_DIRS):
                continue
            if member.size == 0 or member.size > 120_000:
                continue
            try:
                src = tf.extractfile(member).read().decode("utf-8", errors="replace")
            except Exception:
                continue
            rel = n.split("/", 1)[1] if "/" in n else n
            chunks.append(f"# ==== file: {repo}/{rel} ====\n{src}")
            total += len(src)
            if total >= max_bytes:
                break
    if total < 20_000:
        print(f"  SKIP {repo}: too little code ({total} bytes)")
        return 0
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))
    return os.path.getsize(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="my_ai/data/raw")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for repo, ref in REPOS.items():
        size = fetch_repo(repo, ref, args.out)
        if size:
            total += size
            print(f"  OK {repo} ({size/1e6:.2f} MB)")
    print(f"\npython code total: {total/1e6:.1f} MB in {args.out}")


if __name__ == "__main__":
    main()
