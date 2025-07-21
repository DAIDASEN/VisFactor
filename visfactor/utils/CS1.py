"""Random white masker
================================
Utilities for creating vision‑recognition quizzes by partially **erasing** black‑on‑white silhouette
images with random white strokes *and* returning their ground‑truth labels.

What’s inside
-------------
1. **RandomMasker** – applies a configurable white‑stroke mask to any `PIL.Image`.
2. **MaskedImageDataset** – an *iterable* that yields `(masked_image, label)` tuples.  
   It reads a companion **answer.txt** in the *input* folder where each line follows the
   pattern:  
   `file_name object_name`  
   (e.g. `001.png hammer_head`).

Quick start
~~~~~~~~~~~
```bash
python random_white_masker.py Images Quiz --severity 0.6
```
Each image will be saved as *Quiz/masked_<idx>.jpg* while labels are printed for demo.

The module has **no third‑party dependencies except Pillow** and can be embedded in
`torchvision`/`tf.data` pipelines.
"""
from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from PIL import Image, ImageDraw

__all__ = ["RandomMasker", "MaskedImageDataset"]

###############################################################################
# Random white‑stroke masker                                                  #
###############################################################################
class RandomMasker:
    """Apply random white masking to an image.

    Parameters
    ----------
    severity : float, default=0.5
        0 → no mask, 1 → heavy mask. The value interpolates the number of strokes.
    num_strokes : tuple(int, int), default=(4, 40)
        Minimum/maximum strokes when *severity* equals 1.
    stroke_width : tuple(int, int), default=(10, 60)
        Width range in **pixels** for each white stroke.
    rng : random.Random | None
        Optional RNG for reproducibility.
    """

    def __init__(
        self,
        severity: float = 0.5,
        num_strokes: Tuple[int, int] = (4, 40),
        stroke_width: Tuple[int, int] = (10, 60),
        rng: random.Random | None = None,
    ) -> None:
        self.severity = float(max(0.0, min(1.0, severity)))
        self._num_strokes_bounds = num_strokes
        self._stroke_width_bounds = stroke_width
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def with_severity(self, severity: float) -> "RandomMasker":
        """Return a clone configured with another *severity* value."""
        return RandomMasker(severity, self._num_strokes_bounds, self._stroke_width_bounds, self._rng)

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------
    def __call__(self, img: Image.Image) -> Image.Image:
        """Return a *new* `PIL.Image` with random white strokes applied."""
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        min_s, max_s = self._num_strokes_bounds
        n_strokes = int(min_s + self.severity * (max_s - min_s))
        n_strokes = max(0, n_strokes)

        for _ in range(n_strokes):
            x1, y1 = self._rng.randint(0, w), self._rng.randint(0, h)
            x2, y2 = self._rng.randint(0, w), self._rng.randint(0, h)
            width = self._rng.randint(*self._stroke_width_bounds)
            draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255), width=width)

        return img

###############################################################################
# Iterable dataset                                                            #
###############################################################################
class MaskedImageDataset:
    """Lazily produces `(masked_img, label)` tuples for every image in *input_folder*.

    It expects an **answer.txt** residing in *input_folder* where each row is:
    `file_name object_name`. The file name may include or omit the extension.

    Parameters
    ----------
    input_folder : str | Path
        Directory containing source images.
    severity : float, default=0.5
        Passed to :class:`RandomMasker`.
    repeat : bool, default=False
        When *True*, iteration never ends – useful for endless training loops.
    answer_file : str | Path | None
        Custom path to *answer.txt*. Defaults to `<input_folder>/answer.txt`.
    patterns : tuple(str, ...), default=("*.png", "*.jpg", "*.jpeg")
        Glob patterns used to locate images.
    rng : random.Random | None
        RNG shared with :class:`RandomMasker`.
    """

    def __init__(
        self,
        input_folder: str | Path,
        severity: float = 0.5,
        repeat: bool = False,
        answer_file: str | Path | None = None,
        patterns: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg"),
        rng: random.Random | None = None,
    ) -> None:
        self._paths: List[Path] = []
        input_folder = Path(input_folder)
        for pat in patterns:
            self._paths.extend(input_folder.glob(pat))
        self._paths.sort()
        if not self._paths:
            raise FileNotFoundError(f"No images found in folder: {input_folder!s}")

        # ------------------------------------------------------------------
        # Parse ground‑truth file                                            
        # ------------------------------------------------------------------
        self._labels: Dict[str, str] = {}
        answer_path = Path(answer_file) if answer_file else input_folder / "answers.txt"
        if not answer_path.exists():
            raise FileNotFoundError(f"Ground‑truth file missing: {answer_path!s}")
        with answer_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                fname, label = line.strip().split(maxsplit=1)
                self._labels[fname] = label
                # also index without extension for robustness
                self._labels[Path(fname).stem] = label

        self._masker = RandomMasker(severity=severity, rng=rng)
        self._repeat = repeat

    # ------------------------------------------------------------------
    # Python iteration protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[Image.Image, str]]:
        while True:
            for path in self._paths:
                with Image.open(path) as img:
                    masked = self._masker(img.copy())
                label = self._labels.get(path.name) or self._labels.get(path.stem)
                if label is None:
                    raise KeyError(f"No ground truth for image: {path.name}")
                yield masked, label
            if not self._repeat:
                break

    def __len__(self) -> int:  # noqa: D401
        """Number of *source* images (not number of masked variants)."""
        return len(self._paths)
