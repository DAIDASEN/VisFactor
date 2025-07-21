"""Figure generator

Generate connected edge‑patterns on a regular m × n grid, draw them, and test
whether a given "model" sub‑pattern occurs inside the generated pattern.

The generator obeys four rules:
1.  Patterns are embedded in an axis‑aligned rectangular grid of size
   ``rows × cols``.  A pattern is simply a set of *edges*; an edge joins two
   *adjacent* grid points either horizontally, vertically, or diagonally.
2.  The pattern *must be connected*: every edge can be reached from every
   other edge by walking along the drawn edges.
3.  The *outer rectangle* (the four sides of the grid) is *always* present.
4.  The expected number of edges is controlled by a *density* parameter
   ``d ∈ [0,1]``.  The actual density for each sample is drawn from
   ``𝒩(d, σ²)`` (clipped to [0, 1]), where ``σ`` is configurable.

Typical CLI usage
-----------------
::

    python pattern_generator.py \
        --rows 6 --cols 6 --density 0.45 --sigma 0.07 --samples 100 \
        --model "0,1-0,2;0,2-0,3;0,3-1,2;1,2-2,2;2,2-2,1;2,1-1,0;1,0-0,1" \
        --out-dir ./output

For *each* generated pattern the script

* saves an image ``pattern_<idx>.png``;
* saves an image of the model itself (once) as ``model.png``;
* prints ``YES``/``NO`` depending on whether the model occurs in the pattern.

The public API is organised around four helpers:

* :class:`Grid` – neighbourhood logic and edge bookkeeping.
* :class:`PatternGenerator` – random pattern generation.
* :func:`parse_model` – convert the compact edge description into Python data.
* :func:`model_in_pattern` – set‑inclusion test for edge patterns.

The code is pure‑Python (🄿3.8+) and uses only *numpy* and *matplotlib*.
"""

from __future__ import annotations

import argparse
import itertools
import random
from pathlib import Path
from typing import FrozenSet, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

Point = Tuple[int, int]  # (row, col)
Edge = FrozenSet[Point]  # {p, q} — order‑free edge representation

###############################################################################
# Grid helpers
###############################################################################


class Grid:
    """All combinatorial knowledge about a *rows × cols* lattice."""

    _OFFSETS: Tuple[Point, ...] = (
        (-1, 0), (1, 0), (0, -1), (0, 1),  # N S W E
        (-1, -1), (-1, 1), (1, -1), (1, 1),  # diagonals NW NE SW SE
    )

    def __init__(self, rows: int, cols: int):
        if rows < 2 or cols < 2:
            raise ValueError("Grid must be at least 2×2 to have a rectangle.")
        self.rows, self.cols = rows, cols

        # All points
        self._pts: Set[Point] = {(r, c) for r in range(rows) for c in range(cols)}

        # ------------------------------------------------------------------
        # Outer rectangle (rule 3)
        # ------------------------------------------------------------------
        outer: Set[Edge] = set()
        for c in range(cols - 1):  # horizontal borders
            outer.add(frozenset({(0, c), (0, c + 1)}))
            outer.add(frozenset({(rows - 1, c), (rows - 1, c + 1)}))
        for r in range(rows - 1):  # vertical borders
            outer.add(frozenset({(r, 0), (r + 1, 0)}))
            outer.add(frozenset({(r, cols - 1), (r + 1, cols - 1)}))
        self.outer_edges: Set[Edge] = outer

        # All possible edges (orthogonal + diagonals)
        all_edges: Set[Edge] = set()
        for p in self._pts:
            for dr, dc in self._OFFSETS:
                q = (p[0] + dr, p[1] + dc)
                if q in self._pts:
                    all_edges.add(frozenset({p, q}))
        self.all_edges: Set[Edge] = all_edges  # includes outer rectangle

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def neighbours(self, p: Point) -> Iterable[Point]:
        for dr, dc in self._OFFSETS:
            q = (p[0] + dr, p[1] + dc)
            if q in self._pts:
                yield q

    # ------------------------------------------------------------------
    # Drawing (matplotlib) -------------------------------------------------
    # ------------------------------------------------------------------

    def draw(self, edges: Set[Edge], out_path: Path, *, lw: float = 2,
             tight: bool = False, margin: float = 0.5) -> None:
        """Render *edges* to *out_path* as a PNG.

        Parameters
        ----------
        edges : Set[Edge]
            The edges to draw.
        out_path : pathlib.Path
            Output file path (PNG).
        lw : float, default 2
            Line width.
        tight : bool, default ``False``
            If *True* axes limits are cropped to the **minimal bounding box**
            of *edges*, yielding a compact image – useful for models smaller
            than the full grid.
        margin : float, default 0.5
            Extra margin (in grid units) added to each side when ``tight``.
        """
        # ------------------------------------------------------------------
        # Determine figure size & limits
        # ------------------------------------------------------------------
        if tight:
            pts = [p for e in edges for p in e]
            rows = [p[0] for p in pts]
            cols = [p[1] for p in pts]
            r_min, r_max = min(rows), max(rows)
            c_min, c_max = min(cols), max(cols)
            width = c_max - c_min + 1
            height = r_max - r_min + 1
            figsize = (width * 1.2, height * 1.2)
        else:
            r_min, r_max, c_min, c_max = 0, self.rows - 1, 0, self.cols - 1
            figsize = (6, 6)

        # ------------------------------------------------------------------
        # Create figure
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.axis("off")

        # Plot edges (raw grid coordinates)
        for e in edges:
            (r1, c1), (r2, c2) = tuple(e)
            ax.plot([c1, c2], [r1, r2], "k-", lw=lw)

        # Invert y‑axis so origin appears bottom‑left
        ax.set_xlim(c_min - margin, c_max + margin)
        ax.set_ylim(r_max + margin, r_min - margin)

        fig.tight_layout(pad=0)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

###############################################################################
# Pattern generation ---------------------------------------------------------
###############################################################################


class GridFigureGenerator:
    """Generate random **connected** edge patterns on *grid*."""

    def __init__(self, grid: Grid, *, density: float = 0.4, sigma: float = 0.2,
                 seed: int | None = None):
        if not 0.0 <= density <= 1.0:
            raise ValueError("density must lie in [0,1].")
        self.grid = grid
        self.density = density
        self.sigma = sigma
        self._max_edges = len(grid.all_edges)
        self._perimeter = grid.outer_edges
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self) -> Set[Edge]:
        """Return a **connected** edge pattern (outer rectangle included)."""
        target = self._draw_target_edge_count()
        edges: Set[Edge] = set(self._perimeter)
        remaining = list(self.grid.all_edges - edges)
        random.shuffle(remaining)

        # Union‑Find on vertices
        parent: dict[Point, Point] = {}

        def find(x: Point) -> Point:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: Point, b: Point) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # init with perimeter vertices
        for e in edges:
            p, q = tuple(e)
            parent.setdefault(p, p)
            parent.setdefault(q, q)
            union(p, q)

        def connects(e: Edge) -> bool:
            p, q = tuple(e)
            in_p, in_q = p in parent, q in parent
            if in_p and in_q:
                return find(p) != find(q)
            return in_p or in_q

        for e in remaining:
            if len(edges) >= target:
                break
            if connects(e):
                edges.add(e)
                p, q = tuple(e)
                parent.setdefault(p, p)
                parent.setdefault(q, q)
                union(p, q)

        if len(edges) < target:
            leftovers = [e for e in remaining if e not in edges]
            edges.update(leftovers[: target - len(edges)])
        return edges

    # ------------------------------------------------------------------
    # internals ---------------------------------------------------------
    # ------------------------------------------------------------------

    def _draw_target_edge_count(self) -> int:
        val = float(np.clip(np.random.normal(self.density, self.sigma), 0.0, 1.0))
        return max(len(self._perimeter), round(val * self._max_edges))

###############################################################################
# Model parsing / inclusion test --------------------------------------------
###############################################################################


def parse_model(s: str) -> Set[Edge]:
    edges: Set[Edge] = set()
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        p_raw, q_raw = (t.strip() for t in tok.split("-"))
        p = tuple(int(v) for v in p_raw.split(","))  # type: ignore[arg-type]
        q = tuple(int(v) for v in q_raw.split(","))  # type: ignore[arg-type]
        edges.add(frozenset({p, q}))
    return edges


# translation‑aware inclusion ------------------------------------------------

def _translate_edge(e: Edge, dr: int, dc: int) -> Edge:
    p, q = tuple(e)
    return frozenset({(p[0] + dr, p[1] + dc), (q[0] + dr, q[1] + dc)})


def _translate_edges(edges: Set[Edge], dr: int, dc: int) -> Set[Edge]:
    return {_translate_edge(e, dr, dc) for e in edges}


def model_in_pattern(model: Set[Edge], pattern: Set[Edge]) -> bool:
    model_pts = {p for e in model for p in e}
    pattern_pts = {p for e in pattern for p in e}
    for p_pat, p_mod in itertools.product(pattern_pts, model_pts):
        dr, dc = p_pat[0] - p_mod[0], p_pat[1] - p_mod[1]
        if _translate_edges(model, dr, dc) <= pattern:
            return True
    return False
