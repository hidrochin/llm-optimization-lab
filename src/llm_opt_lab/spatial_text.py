"""Render OCR words with boxes as layout-preserving text ("spatial text").

Used to build the spatial-text records of public datasets. The production format may
differ; change `render_spatial_text` to match it once it is known.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2


def group_lines(words: list[Word]) -> list[list[Word]]:
    """Group words into text lines: a word joins a line when its vertical center is
    inside the line's vertical span. Lines are top to bottom, words left to right."""
    lines: list[list[Word]] = []
    spans: list[tuple[float, float]] = []
    for word in sorted(words, key=lambda w: (w.y_center, w.x0)):
        for i, (top, bottom) in enumerate(spans):
            if top <= word.y_center <= bottom:
                lines[i].append(word)
                spans[i] = (min(top, word.y0), max(bottom, word.y1))
                break
        else:
            lines.append([word])
            spans.append((word.y0, word.y1))
    order = sorted(range(len(lines)), key=lambda i: spans[i][0])
    return [sorted(lines[i], key=lambda w: w.x0) for i in order]


def render_spatial_text(words: list[Word], max_columns: int = 160) -> str:
    """Place each word at the column matching its x position, one text line per row."""
    words = [w for w in words if w.text.strip()]
    if not words:
        return ""
    char_width = statistics.median((w.x1 - w.x0) / len(w.text) for w in words) or 1.0
    left = min(w.x0 for w in words)
    width = max(w.x1 for w in words) - left
    char_width = max(char_width, width / max_columns)

    rows = []
    for line in group_lines(words):
        row = ""
        for word in line:
            column = round((word.x0 - left) / char_width)
            gap = column - len(row) if not row else max(1, column - len(row))
            row += " " * gap + word.text
        rows.append(row.rstrip())
    return "\n".join(rows)
