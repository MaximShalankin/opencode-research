#!/usr/bin/env python3
"""PDF статьи → markdown с постраничными якорями <!-- page: N -->.

Использует pdftotext (poppler) постранично; результат — один markdown-файл,
который vault-core индексирует как внешнюю коллекцию `papers`
(chunk_boundary: page). PDF считается данными: скрипт только читает его
через pdftotext и пишет текст, ничего из PDF не исполняет.

Пример:
    python3 tools/paper_ingest.py source.pdf --out papers/aiayn --slug aiayn

Требует: poppler (pdfinfo, pdftotext).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PAGE_MARK = "<!-- page: {n} -->"
_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_pages(pdf: Path) -> int:
    out = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.MULTILINE)
    if not m:
        raise RuntimeError("pdfinfo: не найден Pages")
    return int(m.group(1))


def normalize_page(text: str, page_no: int) -> str:
    """Убрать хвостовой номер страницы и лишние пробелы/пустые строки."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == str(page_no):
        lines.pop()
    text = "\n".join(lines)
    text = _WS_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n\n", text)
    return text.strip()


def extract_page(pdf: Path, page_no: int) -> str:
    out = subprocess.run(
        ["pdftotext", "-f", str(page_no), "-l", str(page_no), str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return normalize_page(out, page_no)


def build_markdown(pdf: Path, *, title: str, arxiv: str, pages: int) -> tuple[str, list[int]]:
    parts = [
        "---",
        f'title: "{title}"',
        "type: paper",
        "status: actual",
        f'source: "arXiv:{arxiv}"',
        f"created: {datetime.now(timezone.utc).date().isoformat()}",
        "---",
        "",
        f"# {title} (arXiv:{arxiv})",
        "",
    ]
    lengths: list[int] = []
    for n in range(1, pages + 1):
        body = extract_page(pdf, n)
        lengths.append(len(body))
        parts.append(PAGE_MARK.format(n=n))
        parts.append(body)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n", lengths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="папка papers/<slug> в корне проекта")
    ap.add_argument("--slug", default="aiayn")
    ap.add_argument("--arxiv", default="1706.03762")
    ap.add_argument("--title", default="Attention Is All You Need")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    pdf = args.pdf.resolve()
    if not pdf.is_file():
        print(f"нет файла: {pdf}", file=sys.stderr)
        return 2
    if shutil.which("pdftotext") is None:
        print("pdftotext не найден (brew install poppler)", file=sys.stderr)
        return 2

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.slug}.md"
    meta_path = out_dir / "meta.json"
    pdf_sha = sha256(pdf)

    if meta_path.is_file() and not args.force:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("pdf_sha256") == pdf_sha and md_path.is_file():
            print(f"актуально: {md_path} (pdf sha совпадает); --force для пересборки")
            return 0

    pages = pdf_pages(pdf)
    markdown, lengths = build_markdown(pdf, title=args.title, arxiv=args.arxiv, pages=pages)
    md_path.write_text(markdown, encoding="utf-8")

    meta = {
        "title": args.title,
        "arxiv": args.arxiv,
        "slug": args.slug,
        "pdf": pdf.name,
        "pdf_sha256": pdf_sha,
        "markdown": md_path.name,
        "markdown_sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "pages": pages,
        "page_chars": lengths,
        "extractor": "pdftotext",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"собрано: {md_path} ({pages} стр., {len(markdown)} символов)")
    print(f"метаданные: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
