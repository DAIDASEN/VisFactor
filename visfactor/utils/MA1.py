"""Iterable dataset that yields *(grid_img, query_img, label)* tuples.

Parameters
----------
image_dir : str
    Directory containing the source object images.
capacity : int, default ``40``
    Number of (object, number) pairs per grid.
cell_size : int, default ``96``
    Pixel edge length of each square cell.
padding : int, default ``4``
    Whitespace margin (in pixels) between the rendered content and its 1‑px
    black border.
    When *True*, the right‑hand numeric cells are rendered with **OpenCV**
    (``cv2.putText``). This avoids the font‑scaling limitations of Pillow
    when no TrueType font is available.
seed : int or None, default ``None``
    Optional RNG seed for reproducible sampling.

Notes
-----
*   The iterator is **infinite** – every ``next(ds)`` call reshuffles both
    the objects and their numbers.
*   Returned images are **PIL.Image** objects in *RGB* mode; ready for
    ``torchvision`` transforms if needed.
"""
import os
import random
import math
from typing import Tuple, List, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

__all__ = ["CompositeGridDataset"]


class CompositeGridDataset:
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(self,
                 image_dir: str,
                 capacity: int = 40,
                 cell_size: int = 96,
                 padding: int = 4,
                 seed: Optional[int] = None):
        self.image_dir = os.path.expanduser(image_dir)
        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Directory not found: {self.image_dir}")

        self.capacity = capacity
        self.cell_size = cell_size
        self.padding = padding
        self.random = random.Random(seed)

        # --------------------------------------------------------------
        # Collect image paths
        # --------------------------------------------------------------
        self._image_paths: List[str] = [
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if os.path.splitext(f.lower())[1] in self.IMG_EXTS
        ]
        if len(self._image_paths) < self.capacity:
            raise ValueError(
                f"Not enough images in '{self.image_dir}'. Required {self.capacity}, found {len(self._image_paths)}."
            )

    # ------------------------------------------------------------------
    # Public iterator interface
    # ------------------------------------------------------------------
    def __iter__(self) -> "CompositeGridDataset":
        return self

    def __next__(self) -> Tuple[Image.Image, Image.Image, int]:
        # 1. Sample *capacity* distinct images & unique numbers
        paths = self.random.sample(self._image_paths, self.capacity)
        numbers = self.random.sample(range(10, 100), self.capacity)

        # 2. Build paired cells: (object | number)
        object_cells, pair_imgs = [], []
        for pth, num in zip(paths, numbers):
            obj_cell = self._make_object_cell(pth)
            num_cell = self._make_number_cell(num)
            pair = Image.new("RGB", (self.cell_size * 2, self.cell_size), "white")
            pair.paste(obj_cell, (0, 0))
            pair.paste(num_cell, (self.cell_size, 0))
            object_cells.append(obj_cell)
            pair_imgs.append(pair)

        # 3. Compose near‑square grid
        rows, cols = self._best_grid(self.capacity)
        grid = Image.new("RGB", (cols * self.cell_size * 2, rows * self.cell_size), "white")
        for idx, pair in enumerate(pair_imgs):
            r, c = divmod(idx, cols)
            grid.paste(pair, (c * self.cell_size * 2, r * self.cell_size))

        # 4. Pick random query
        qi = self.random.randrange(self.capacity)
        return grid, object_cells[qi], numbers[qi]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_object_cell(self, img_path: str) -> Image.Image:
        with Image.open(img_path) as im:
            im = im.convert("RGBA")
            max_inner = self.cell_size - 2 * self.padding
            ratio = min(max_inner / im.width, max_inner / im.height)
            im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))),
                           Image.Resampling.LANCZOS)

        cell = Image.new("RGB", (self.cell_size, self.cell_size), "white")
        cell.paste(im, ((self.cell_size - im.width) // 2, (self.cell_size - im.height) // 2), im)
        self._draw_border_pil(cell)
        return cell

    # -------------- Number‑cell renderers ---------------------------------
    def _make_number_cell(self, num: int) -> Image.Image:
        text = f"{num:02d}"
        max_dim = self.cell_size - 2 * self.padding
        cell = np.full((self.cell_size, self.cell_size, 3), 255, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Binary search the largest scale that fits
        low, high = 0.1, 10.0
        for _ in range(20):  # sufficient iterations for convergence
            scale = (low + high) / 2.0
            (w, h), baseline = cv2.getTextSize(text, font, scale, thickness=1)
            if w <= max_dim and h <= max_dim:
                low = scale  # fits – try bigger
            else:
                high = scale  # too big – shrink
        scale = low
        thickness = max(1, int(scale * 2 / 3))
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = (self.cell_size - w) // 2
        y = (self.cell_size + h) // 2 - baseline
        cv2.putText(cell, text, (x, y), font, scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
        cv2.rectangle(cell, (0, 0), (self.cell_size - 1, self.cell_size - 1), (0, 0, 0), 1)
        return Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))

    # -------------- Misc ---------------------------------------------------
    def _draw_border_pil(self, img: Image.Image) -> None:
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (img.width - 1, img.height - 1)], outline="black", width=1)

    @staticmethod
    def _best_grid(n: int) -> Tuple[int, int]:
        root = int(math.sqrt(n))
        rows, cols = root, root
        if rows * cols < n:
            cols += 1
        while rows * cols < n:
            rows += 1 if cols > rows else 0
            cols += 1 if rows > cols else 0
        return rows, cols
