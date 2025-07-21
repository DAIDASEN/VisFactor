"""Word image random white masker
---------------------

Synthetic dataset generator – **OpenCV‑only** version
====================================================
This uses the built‑in OpenCV raster fonts. The image is automatically scaled
so the rendered word **fills the entire canvas**, and the word list is fetched
*inside* the dataset constructor – simply specify `min_len` and `max_len`.

Quick start
===========
```python
from word_image_dataset import WordImageDataset

ds = WordImageDataset(min_len=3, max_len=7, sample=500, severity=0.4,
                      img_size=(64, 256))

for idx, (img, lbl) in enumerate(ds):
    cv2.imwrite(f"out/{idx:05d}_{lbl}.png", img)
```

*   No need for `fonts_dir`; OpenCV fonts are picked at random.
*   `severity` ∈ [0, 1] controls the density of white masks.
*   The iterator yields **NumPy BGR images** plus their label.
"""

from __future__ import annotations

import os
import random
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from wordfreq import top_n_list

__all__ = ["WordImageDataset"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_words(
    *,
    min_len: int,
    max_len: int,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
) -> List[str]:
    """Fetch and filter common English words from *nltk.corpus.words*."""
    common_words = top_n_list('en', 1000)

    wl = [
        w.lower()
        for w in common_words
        if min_len <= len(w) <= max_len and w.isalpha()
    ]

    if sample is not None and sample < len(wl):
        rng = random.Random(random_state)
        wl = rng.sample(wl, sample)
    return wl


# Choose from OpenCVʼs built‑in fonts – these are always available.
_CV2_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
]


def _render_word(word: str, img_size: Tuple[int, int]) -> np.ndarray:
    """Render *word* (black) onto a white canvas filling the area."""
    h, w = img_size
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # Start with scale=1, then compute a scale factor to fit.
    font = random.choice(_CV2_FONTS)
    base_scale = 1.0
    thickness = 2
    (tw, th), bl = cv2.getTextSize(word, font, base_scale, thickness)

    # Scale so that width/height occupy ~90 % of canvas.
    scale_factor = min((w * 0.9) / tw, (h * 0.7) / th)
    scale = max(scale_factor, 0.1)  # keep >0
    thickness = max(1, int(scale * 2))

    # Recompute final size for centring.
    (tw, th), bl = cv2.getTextSize(word, font, scale, thickness)

    x = (w - tw) // 2
    y = (h + th) // 2  # baseline is at +th/2 below top of text bbox

    cv2.putText(img, word, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def _apply_white_mask(img: np.ndarray, severity: float, *, max_lines: int = 40, max_spots: int = 15) -> np.ndarray:
    """Super‑impose random white lines & spots proportional to *severity*."""
    severity = float(np.clip(severity, 0.0, 1.0))
    h, w = img.shape[:2]
    n_lines = int(1 + severity * max_lines)
    n_spots = int(1 + severity * max_spots)

    for _ in range(n_lines):
        thickness = random.randint(5, int(5 + severity * 2))
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        angle = random.uniform(0, 360)
        length = random.randint(int(w * 0.2), int(w * 0.6))
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = int(y1 + length * np.sin(np.radians(angle)))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)

    for _ in range(n_spots):
        radius = random.randint(5, int(5 + severity * 5))
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        cv2.circle(img, (x, y), radius, (255, 255, 255), -1)

    return img


# ---------------------------------------------------------------------------
# Iterable dataset
# ---------------------------------------------------------------------------

class WordImageDataset(Iterable):
    """Stream `(img, label)` pairs; words are fetched internally via NLTK."""

    def __init__(
        self,
        min_len: int = 3,
        max_len: int = 8,
        *,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        img_size: Tuple[int, int] = (64, 256),
        severity: float = 0.3,
        shuffle: bool = True,
        infinite: bool = False,
    ) -> None:
        self.word_list = _load_words(
            min_len=min_len, max_len=max_len, sample=sample, random_state=random_state
        )
        self.img_size = img_size
        self.severity = float(np.clip(severity, 0.0, 1.0))
        self.shuffle = shuffle
        self.infinite = infinite

    def __iter__(self):
        idxs = list(range(len(self.word_list)))
        rng = random.Random()
        while True:
            if self.shuffle:
                rng.shuffle(idxs)
            for idx in idxs:
                word = self.word_list[idx]
                img = _render_word(word, self.img_size)
                img = _apply_white_mask(img, self.severity)
                yield img, word
            if not self.infinite:
                break

    def __len__(self):
        return len(self.word_list)
