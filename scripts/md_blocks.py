"""Parse the Markdown subset used by report/part-e.md into blocks.

The report is authored in Markdown; this turns it into plain blocks that
`build_part_e_docx.py` renders with the unit template's styles. Supported:

  # / ## / ###        headings
  - item              bullets
  | a | b |           pipe tables (first row is the header)
  :caption            caption for the table or figure just before it
  ![alt](path){w}     figures, width as a percentage
  **bold**, *italic*  inline emphasis
  ```{=openxml} ...   raw page breaks
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

BLOCK_OPEN = "```{=openxml}"


@dataclass
class Block:
    kind: str                      # heading | para | bullet | table | figure | caption | break
    text: str = ""
    level: int = 0
    rows: list[list[str]] = field(default_factory=list)
    align: list[str] = field(default_factory=list)
    src: str = ""
    width: float = 1.0


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(md: str) -> list[Block]:
    blocks: list[Block] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith(BLOCK_OPEN):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            blocks.append(Block("break"))
            continue

        if stripped.startswith(":::"):           # custom-style fence: ignore
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            blocks.append(Block("heading", text=m.group(2).strip(), level=len(m.group(1))))
            i += 1
            continue

        if stripped.startswith("- "):
            blocks.append(Block("bullet", text=stripped[2:].strip()))
            i += 1
            continue

        if stripped.startswith("|"):
            header = _split_row(stripped)
            sep = _split_row(lines[i + 1])
            align = []
            for cell in sep:
                if cell.startswith(":") and cell.endswith(":"):
                    align.append("c")
                elif cell.endswith(":"):
                    align.append("r")
                else:
                    align.append("l")
            rows = [header]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i].strip()))
                i += 1
            blocks.append(Block("table", rows=rows, align=align))
            continue

        m = re.match(r"^!\[(.*?)\]\(([^)]+)\)(?:\{width=([\d.]+)%\})?$", stripped)
        if m:
            blocks.append(Block("figure", text=m.group(1), src=m.group(2),
                                width=float(m.group(3) or 100) / 100))
            i += 1
            continue

        if stripped.startswith(":"):
            blocks.append(Block("caption", text=stripped[1:].strip()))
            i += 1
            continue

        # a paragraph: consume until a blank line or a new block
        para = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", "- ", "![", ":", "```"))
                    or re.match(r"^!\[", nxt)):
                break
            para.append(nxt)
            i += 1
        blocks.append(Block("para", text=" ".join(para)))
    return blocks


def inline(text: str) -> list[tuple[str, bool, bool]]:
    """Split inline markdown into (text, bold, italic) runs."""
    runs: list[tuple[str, bool, bool]] = []
    for piece in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            runs.append((piece[2:-2], True, False))
        elif piece.startswith("*") and piece.endswith("*"):
            runs.append((piece[1:-1], False, True))
        else:
            runs.append((piece, False, False))
    return runs


def load(path: Path) -> list[Block]:
    return parse(path.read_text())
