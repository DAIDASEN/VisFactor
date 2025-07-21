"""Image noise adder

Utility for on‑the‑fly generation of *structured noise* on images and a
light‑weight iterable dataset wrapper.

Noise model
===========
Two independent corruptions are applied, both controlled by a single
``severity`` float in the range ``[0, 1]``:

1. **White rectangles** – randomly placed and sized white rectangles that
   break object continuity.
2. **Black line segments** – short, randomly‐oriented dark strokes acting as
   visual clutter.

When ``severity == 0`` the image is returned unchanged; when
``severity == 1`` the maximum numbers of rectangles and lines you specify are
injected.

Example
-------
```python
from noisy_dataset import NoisyImageDataset
from torchvision.transforms import ToTensor

dataset = NoisyImageDataset(
    root_dir="/path/to/data",   # must contain images *and* answers.txt
    severity=0.6,               # fiddle with this!
    transform=ToTensor()        # optional
)

for idx, (img, label) in enumerate(dataset):
    print(idx, label, img.shape)
```

To pre‑export a corrupted copy of the whole dataset:
```python
from noisy_dataset import batch_export, NoisyImageDataset

ds = NoisyImageDataset("data", severity=1.0)
batch_export(ds, "corrupted_pngs")
```
"""
from __future__ import annotations

import glob
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Noise injection helpers
# ---------------------------------------------------------------------------

def add_noise(
    img: Image.Image,
    severity: float,
    *,
    max_white_rectangles: int = 100,
    max_lines: int = 5000,
    max_rect_size_ratio: float = 0.1,
    line_length: int = 20,
    line_width_range: Tuple[int, int] = (1, 10),
) -> Image.Image:
    """Return *copy* of ``img`` with random rectangles + lines applied.

    Parameters
    ----------
    img : PIL.Image
        Source image. Mode will be converted to ``RGB`` internally.
    severity : float
        Strength of corruption in *[0, 1]*.
    max_white_rectangles : int, optional
        Ceiling on number of rectangles at *severity == 1*.
    max_lines : int, optional
        Ceiling on number of line segments at *severity == 1*.
    max_rect_size_ratio : float, optional
        Longest side of a rectangle relative to the shorter image side.
    line_length : int, optional
        Base length (px) of each line segment.
    line_width_range : tuple, optional
        Inclusive range for line width selection.
    """
    if not (0.0 <= severity <= 1.0):  # cheap guard – saves silent bugs
        raise ValueError("severity must be within [0, 1]")

    # Early exit – keep it zero‑cost if no noise is requested.
    if severity == 0:
        return img

    img = img.convert("RGB")  # defensive copy – never mutate caller's image
    width, height = img.size
    draw = ImageDraw.Draw(img)

    # --- white rectangles --------------------------------------------------
    num_rects = int(max_white_rectangles * severity)
    short_side = min(width, height)
    max_rect_side = max(1, int(short_side * max_rect_size_ratio))

    for _ in range(num_rects):
        w = random.randint(1, max_rect_side)
        h = random.randint(1, max_rect_side)
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = min(x1 + w, width)
        y2 = min(y1 + h, height)
        draw.rectangle([x1, y1, x2, y2], fill="white")

    # --- black line segments ----------------------------------------------
    num_lines = int(max_lines * severity)
    two_pi = 2 * math.pi

    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        angle = random.random() * two_pi
        x2 = int(x1 + line_length * math.cos(angle))
        y2 = int(y1 + line_length * math.sin(angle))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        lw = random.randint(*line_width_range)
        draw.line((x1, y1, x2, y2), fill="black", width=lw)

    return img

# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class NoisyImageDataset:
    """Iterable dataset that injects noise on‑the‑fly.

    Expected directory structure::

        dataset_root/
        ├─ my_image_0.png
        ├─ my_image_1.png
        ├─ ...
        └─ answers.txt         # lines: "<file_name> <label>"
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        *,
        severity: float = 0.5,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
        cache_processed: bool = False,
        transform=None,
    ) -> None:
        if not (0 <= severity <= 1):
            raise ValueError("severity must be within [0, 1]")

        self.root_dir = Path(root_dir)
        self.severity = severity
        self.transform = transform
        self.cache_processed = cache_processed
        self._cache: Dict[int, Image.Image] = {}

        answers_path = self.root_dir / "answers.txt"
        if not answers_path.exists():
            raise FileNotFoundError(f"{answers_path} not found.")

        # Parse labels
        self._label_map: Dict[str, str] = {}
        with answers_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                file_name, label = line.split(maxsplit=1)
                self._label_map[file_name] = label

        # Collect image paths that have labels
        self._image_paths: List[Path] = []
        for ext in extensions:
            self._image_paths.extend(self.root_dir.glob(f"*{ext}"))
        self._image_paths = [p for p in self._image_paths if p.name in self._label_map]
        self._image_paths.sort()  # deterministic ordering

    # ------------------------------------------------------------------
    # Dunder methods – make it behave like a PyTorch Dataset but also
    # iterable out‑of‑the‑box:  for idx, (img, lbl) in enumerate(ds): ...
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401 – simple function
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, str]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self.cache_processed and idx in self._cache:
            return self._cache[idx]

        path = self._image_paths[idx]
        label = self._label_map[path.name]

        img = Image.open(path)
        img = add_noise(img, self.severity)

        if self.transform is not None:
            img = self.transform(img)

        sample = (img, label)
        if self.cache_processed:
            self._cache[idx] = sample  # type: ignore[assignment]
        return sample

    def __iter__(self) -> Iterator[Tuple[Image.Image, str]]:  # pragma: no cover
        for idx in range(len(self)):
            yield self[idx]
