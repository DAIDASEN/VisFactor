"""Generate dissect-and-assemble geometry puzzles like the sample image.

Usage example (see __main__ block):
    # Describe a cross shape on a 5×5 grid
    Puzzle.from_edges(target_edges).export("out")

Key ideas
---------
* The **target** polygon lives on an *n × n* integer grid (default n=5).
* A random integer k∈{3,4,5} decides how many **solution pieces** we cut the
  target into.
* Allowed cut lines are straight segments through grid points with slope
  ∞, 0, ±1, ±2, ±3.  Each cut must split at least one existing piece.
* **Distractors** are obtained by recursively sub-dividing *one* correct piece
  so that no combination of distractors reproduces the exact area of any
  single correct piece.  (We enforce this exhaustively.)
* Each output PNG is rendered with matplotlib — target outline only, while
  pieces and distractors are filled using a dotted hatch pattern similar to
  the book example.

Dependencies
------------
* shapely (geometry operations)
* matplotlib
* numpy
"""
from __future__ import annotations

import math
import random
import textwrap
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
from shapely.geometry import GeometryCollection, LineString, Polygon
from shapely.ops import split

import os
import re
from PIL import Image

GridPoint = Tuple[int, int]
Edge = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

ALLOWED_SLOPES = {float("inf"), 0, 1, -1, 2, -2, 3, -3}
DEFAULT_PPU = 100  # pixels per grid unit when exporting PNGs
DPI = 300  # matplotlib resolution


# ---------------------------------------------------------------------------
#   Geometry helpers
# ---------------------------------------------------------------------------
def _parse_edges(text: str) -> List[Edge]:
    segments = [seg.strip() for seg in text.split(';')]
    
    square_edges = []
    for seg in segments:
        start, end = seg.split('-')
        y1, x1 = map(int, start.split(','))
        y2, x2 = map(int, end.split(','))
        square_edges.append((x1, y1, x2, y2))
    
    return square_edges


def _edge_list_to_polygon(edges: str) -> Polygon:
    """Return a (possibly non‑convex) shapely Polygon from ordered *edges*."""
    edges = _parse_edges(edges)
    vertices: List[GridPoint] = [(edges[0][0], edges[0][1])]
    for (_, _, x2, y2) in edges:
        vertices.append((x2, y2))
    return Polygon(vertices)


def _random_grid_line(n: int) -> LineString:
    """Random straight line through two grid points whose slope is allowed."""
    while True:
        p1 = (random.randint(0, n), random.randint(0, n))
        p2 = (random.randint(0, n), random.randint(0, n))
        if p1 == p2:
            continue
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        slope = float("inf") if dx == 0 else dy / dx
        if slope in ALLOWED_SLOPES:
            return LineString([p1, p2])


def _split_polygon(poly: Polygon, n: int, max_tries: int = 200) -> List[Polygon]:
    """Attempt to cut *poly*; return list of pieces or ``[poly]`` if none."""
    for _ in range(max_tries):
        line = _random_grid_line(n)
        try:
            gc = split(poly, line)
        except ValueError:
            continue  # line did not intersect the polygon
        pieces = [g for g in (gc.geoms if isinstance(gc, GeometryCollection) else [gc]) if not g.is_empty]
        if len(pieces) > 1:
            return pieces
    return [poly]


def _merge_images(folder: str, interval: int = 25):
    pattern = re.compile(r"(\d+)-(\d+)\.png")
    images_dict = {}

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            num1, num2 = match.groups()
            if num2 == '0':
                continue
            num1, num2 = int(num1), int(num2)
            images_dict.setdefault(num1, []).append((num2, filename))

    for num1, images_info in images_dict.items():
        images_info.sort()

        images = []
        total_width = 0
        max_height = 0
        for num2, filename in images_info:
            img = Image.open(os.path.join(folder, filename))
            images.append(img)
            total_width += img.width
            if img.height > max_height:
                max_height = img.height

        combined_image = Image.new("RGBA", (total_width + interval * (len(images) - 1), max_height), (255, 255, 255, 0))
        x_offset = 0
        for img in images:
            y_offset = max_height - img.height
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.width + interval

        save_path = os.path.join(folder, f"{num1}-choices.png")
        combined_image.save(save_path)


# ---------------------------------------------------------------------------
#   Main class
# ---------------------------------------------------------------------------
class Puzzle:
    def __init__(self, target: Polygon, grid_size: int = 5):
        if not target.is_valid:
            raise ValueError("Target polygon is invalid / self‑intersecting")
        self.target = target
        self.n = grid_size  # size of underlying grid (0…n)

    # ------------------------------------------------------------------
    #   Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_edges(cls, edges: str, grid_size: int = 5) -> "Puzzle":
        return cls(_edge_list_to_polygon(edges), grid_size)

    def _generate_solution_pieces(self, k: int) -> List[Polygon]:
        pieces = [self.target]
        guard = 0
        while len(pieces) < k and guard < 1000:
            guard += 1
            pieces.sort(key=lambda p: p.area, reverse=True)
            largest = pieces.pop(0)
            new_pieces = _split_polygon(largest, self.n)
            if len(new_pieces) == 1:
                pieces.append(largest)
                continue
            pieces.extend(new_pieces)
        if len(pieces) < k:
            raise RuntimeError("Could not split target into k pieces — try a simpler shape or increase attempts.")
        return pieces[:k]

    def _generate_distractors(self, pieces: List[Polygon], num_distractors: int) -> List[Polygon]:
        distractors: List[Polygon] = []
        tries = 0
        while len(distractors) < num_distractors and tries < 2000:
            tries += 1
            base = random.choice(pieces)
            frags = _split_polygon(base, self.n)
            if len(frags) == 1:
                continue
            if any(math.isclose(f.area, p.area, abs_tol=1e-6) for f in frags for p in pieces + distractors):
                continue
            if any(math.isclose(f.area + d.area, p.area, abs_tol=1e-6) for f in frags for d in distractors for p in pieces):
                continue
            distractors.append(random.choice(frags))
        if len(distractors) < num_distractors:
            raise RuntimeError("Unable to create enough distractors without area clashes.")
        return distractors

    # ------------------------------------------------------------------
    #   Rendering helpers
    # ------------------------------------------------------------------
    def _render_polygon(
        self,
        poly: Polygon,
        path: Path,
        shaded: bool = True,
        ppu: int = DEFAULT_PPU,
        dpi: int = DPI,
        margin_units: float = 0.05,
        line_width: float = 2.0,
    ) -> None:
        """Save *poly* to *path* keeping global scale (*ppu*)."""
        minx, miny, maxx, maxy = poly.bounds
        width_units = maxx - minx
        height_units = maxy - miny
        # Pad a small margin so stroke is not clipped
        minx -= margin_units
        miny -= margin_units
        maxx += margin_units
        maxy += margin_units
        width_units += 2 * margin_units
        height_units += 2 * margin_units

        # Convert to figure size in inches
        fig_w = (width_units * ppu) / dpi
        fig_h = (height_units * ppu) / dpi
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        if shaded:
            x, y = poly.exterior.xy
            ax.fill(x, y, hatch="..", edgecolor="black", linewidth=line_width, facecolor="none")
        else:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=line_width)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_axis_off()
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(path, transparent=True)
        plt.close(fig)

    # ------------------------------------------------------------------
    #   Public API
    # ------------------------------------------------------------------
    def export(self, out_dir: str, ppu: int = DEFAULT_PPU) -> None:
        """Generate one puzzle (target + 5 options) in *out_dir* keeping scale."""
        k = random.randint(2, 5)
        pieces = self._generate_solution_pieces(k)
        distractors = self._generate_distractors(pieces, 5 - k)

        # Target (outline only)
        self._render_polygon(self.target, out_dir.replace(".png", "-0.png"), shaded=False, ppu=ppu)

        # Options — shuffle order
        options = [(True, p) for p in pieces] + [(False, d) for d in distractors]
        random.shuffle(options)
        ret = []
        for idx, (is_sol, poly) in enumerate(options, 1):
            ret.append('T' if is_sol else 'F')
            self._render_polygon(poly, out_dir.replace(".png", f"-{idx}.png"), shaded=True, ppu=ppu)
        
        _merge_images("/".join(out_dir.split("/")[:-1]))
        return ret
