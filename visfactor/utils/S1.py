"""Random polygon generator
--------------------
An iterable PyTorch‐style dataset (but without the torch dependency) that

a)  generates *asymmetric* simple polygons with a random number of vertices
b)  renders each polygon to a square PIL.Image
c)  creates <eval_num> augmented views of that polygon with a **random
    rotation** and an **optional horizontal flip**

d)  returns `(base_img, imgs, labels)` where:
       imgs   – `List[PIL.Image]` length <eval_num>
       labels – `List[bool]`     length <eval_num>
                 True  ⇢  only rotation (no flip)
                 False ⇢  flip + rotation

For every polygon at least one label is True *and* at least one label is
False, satisfying the user’s constraint that the data for a polygon may
not be all‑True or all‑False.

Example
~~~~~~~
>>> from random_polygon_dataset import RandomPolygonDataset
>>> ds = RandomPolygonDataset(num_polygons=3, eval_num=6, seed=42)
>>> for idx, (imgs, labels) in enumerate(ds):
...     print(idx, labels)
...     imgs[0].show()  # inspect one sample if desired

The class is dependency‑light (only Pillow).
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Iterator, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageOps

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _generate_simple_polygon(
    rng: random.Random,
    n_vertices: int,
    *,
    r_min: float,
    r_max: float,
    min_edge_ratio: float,
    tol: float = 1e-6,
    max_attempts: int = 2000,
) -> List[Tuple[float, float]]:
    """Generate a *simple* polygon whose *shortest edge* ≥ min_edge_len.

    Edge‑length uniqueness (|Δℓ|>tol), keep to avoid symmetric
    """

    if n_vertices < 3:
        raise ValueError("Number of vertices must be ≥ 3")

    min_edge_len = min_edge_ratio * (r_min + r_max) / 2.0

    for _ in range(max_attempts):
        angles = sorted(rng.uniform(0.0, 2 * math.pi) for _ in range(n_vertices))
        radii = [rng.uniform(r_min, r_max) for _ in range(n_vertices)]
        pts = [
            (radii[i] * math.cos(angles[i]), radii[i] * math.sin(angles[i]))
            for i in range(n_vertices)
        ]

        edge_lens = [
            math.hypot(
                pts[(i + 1) % n_vertices][0] - pts[i][0],
                pts[(i + 1) % n_vertices][1] - pts[i][1],
            )
            for i in range(n_vertices)
        ]

        if min(edge_lens) < min_edge_len:
            continue

        edge_lens_sorted = sorted(edge_lens)
        if not all(abs(a - b) > tol for a, b in zip(edge_lens_sorted, edge_lens_sorted[1:])):
            continue

        return pts

    raise RuntimeError("Polygon generation failed – try relaxing constraints.")


def _draw_polygon(
    pts: Sequence[Tuple[float, float]],
    *,
    canvas_size: int,
    shape_size: int,
    outline_width: int,
    render_scale: int = 4,
) -> Image.Image:
    """Render polygon to a square PIL.Image (RGB, white bg)."""

    CS = canvas_size * render_scale
    SS = shape_size * render_scale
    OW = outline_width * render_scale

    xs, ys = zip(*pts)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = SS / max(max_x - min_x, max_y - min_y)
    cx = cy = CS / 2.0

    mapped = [
        (
            (x - (min_x + max_x) / 2) * scale + cx,
            -(y - (min_y + max_y) / 2) * scale + cy,
        )
        for x, y in pts
    ]

    img_hr = Image.new("RGB", (CS, CS), "white")
    d = ImageDraw.Draw(img_hr)
    d.polygon(mapped, fill="white")
    d.line(mapped + [mapped[0]], fill="black", width=OW)

    return img_hr.resize((canvas_size, canvas_size), Image.LANCZOS)


# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------


@dataclass
class RandomPolygonDataset:
    """Iterable dataset yielding (<eval_num> images, <eval_num> labels)."""

    num_polygons: int
    eval_num: int = 8
    n_vertices_range: Tuple[int, int] = (5, 10)
    r_min: float = 0.5
    r_max: float = 1.5
    min_edge_ratio: float = 0.2
    canvas_size: int = 128
    shape_size: int = 96
    outline_width: int = 2
    render_scale: int = 4
    pad_ratio: float = 0.35
    seed: int | None = None

    _rng: random.Random = field(init=False, repr=False)

    # ---------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: DunderLen
        return self.num_polygons

    def __iter__(self) -> Iterator[Tuple[List[Image.Image], List[bool]]]:  # noqa: DunderIter
        for poly_idx in range(self.num_polygons):
            n_vertices = self._rng.randint(*self.n_vertices_range)
            pts = _generate_simple_polygon(
                rng=self._rng,
                n_vertices=n_vertices,
                r_min=self.r_min,
                r_max=self.r_max,
                min_edge_ratio=self.min_edge_ratio,
            )

            base = _draw_polygon(
                pts,
                canvas_size=self.canvas_size,
                shape_size=self.shape_size,
                outline_width=self.outline_width,
                render_scale=self.render_scale,
            )

            imgs, labels = self._make_eval_set(base)
            yield base, imgs, labels

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------

    def _make_eval_set(self, base_img: Image.Image) -> Tuple[List[Image.Image], List[bool]]:
        imgs: List[Image.Image] = []
        labels: List[bool] = []

        for _ in range(self.eval_num):
            flip = self._rng.choice([True, False])
            angle = self._rng.uniform(0.0, 360.0)
            imgs.append(self._augment_once(base_img, flip, angle))
            labels.append(not flip)

        # 至少 1 True & 1 False ------------------------------------------------
        if not any(labels):
            imgs[0] = self._augment_once(base_img, False, self._rng.uniform(0, 360))
            labels[0] = True
        elif all(labels):
            imgs[0] = self._augment_once(base_img, True, self._rng.uniform(0, 360))
            labels[0] = False

        return imgs, labels

    # ------------------------------------------------------------------
    def _augment_once(self, img: Image.Image, flip: bool, angle: float) -> Image.Image:
        pad = int(self.canvas_size * self.pad_ratio)
        big = self.canvas_size + 2 * pad

        canvas = Image.new("RGB", (big, big), "white")
        canvas.paste(img, (pad, pad))
        if flip:
            canvas = ImageOps.mirror(canvas)

        canvas = canvas.rotate(angle, expand=False, resample=Image.BICUBIC, fillcolor="white")

        left = pad
        upper = pad
        right = left + self.canvas_size
        lower = upper + self.canvas_size
        return canvas.crop((left, upper, right, lower))
