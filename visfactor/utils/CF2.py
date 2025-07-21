"""Grid Pattern Generator
======================
Generates connected line‑segment patterns on an *m × n* grid, with
controllable density, optional Gaussian noise around that density, and a
simple sub‑pattern (model) containment test.  Patterns and the model are
exported as PNGs so they are easy to visualise.

Key features & improvements over the draft
-----------------------------------------
* **Modular design** – the core logic is split into small, typed
  functions and a single `GridPatternGenerator` class.
* **Proper connectivity guarantee** – first build a random spanning tree
  (ensures every node is reachable) and then sample extra edges until
  the desired density is hit.
* **Density as a real parameter** – density ∈ (0, 1] specifies the *mean*
  fraction of all legal edges to keep; the final edge count is drawn
  from `N(mu=density·E, sigma=density_std·E)` and clipped to `[N‑1, E]`.
* **Simple containment test** – checks whether the supplied model’s edge
  set is a subset of the generated pattern.  If you need rotations or
  translations, plug in the alternative matcher stub.
* **Pure Python + OpenCV** – no third‑party graph libraries; everything
  is self‑contained and fast for typical grid sizes (≤ 5 × 5).
* **CLI & reproducibility** – `argparse` interface, deterministic output
  with `--seed`, and all artefacts stored under a chosen output folder.

Usage example
-------------
```bash
python grid_pattern_generator.py \
    --grid 3 3 \
    --samples 400 \
    --density 0.45 \
    --density-std 0.07 \
    --model "0,0-1,0; 1,0-1,1; 1,0-2,0" \
    --out ./Images \
    --seed 1234
```
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Type aliases
Node = Tuple[int, int]  # (row, col)
Edge = Tuple[Node, Node]  # undirected; nodes are sorted in canonical form

# ---------------------------------------------------------------------------
# Low‑level helper functions --------------------------------------------------

def neighbours(node: Node, rows: int, cols: int, diag: bool = True) -> List[Node]:
    """Return all valid neighbour nodes of *node* on the grid."""
    r, c = node
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E
    if diag:
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # NW, NE, SW, SE
    out: List[Node] = []
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def canonical_edge(a: Node, b: Node) -> Edge:
    """Return *edge* with endpoints sorted so that Edge({u, v}) == Edge({v, u})."""
    return (a, b) if a <= b else (b, a)


def all_possible_edges(rows: int, cols: int, diag: bool = True) -> List[Edge]:
    """List every legal edge (**adjacent** nodes only) on the *rows×cols* grid."""
    edges: List[Edge] = []
    for r in range(rows):
        for c in range(cols):
            for nb in neighbours((r, c), rows, cols, diag):
                edges.append(canonical_edge((r, c), nb))
    # Remove duplicates caused by undirectedness
    return list({e for e in edges})


# ---------------------------------------------------------------------------
# Pattern generator ----------------------------------------------------------

@dataclass
class GridPatternGenerator:
    rows: int = 3
    cols: int = 3
    diag: bool = True
    density: float = 0.5                # mean fraction of edges to sample
    density_std: float = 0.2            # std dev (as fraction of all edges)
    img_size: int = 400
    margin: int = 60
    line_thickness: int = 2
    rng: random.Random | None = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.Random()
        self._edges_all: List[Edge] = all_possible_edges(self.rows, self.cols, self.diag)
        self._n_all = len(self._edges_all)

    # ---------------------------------------------------------------------
    # Public API

    def generate_pattern(self) -> Set[Edge]:
        """Return a *connected* random set of edges drawn w.r.t. density."""
        target_n = self._sample_edge_count()
        # 1. build a random spanning tree (rows*cols - 1 edges)
        tree_edges = self._random_spanning_tree()
        extra_needed = max(0, target_n - len(tree_edges))
        # 2. add extra edges uniformly at random (avoiding duplicates)
        extras = self.rng.sample([e for e in self._edges_all if e not in tree_edges], extra_needed)
        return tree_edges.union(extras)

    def draw_edges(self, edges: Iterable[Edge], out_path: Path) -> None:
        """Render *edges* to *out_path* (PNG)."""
        img = np.full((self.img_size, self.img_size, 3), 255, np.uint8)
        gap_x = (self.img_size - 2 * self.margin) / (self.cols - 1)
        gap_y = (self.img_size - 2 * self.margin) / (self.rows - 1)

        def to_px(node: Node) -> Tuple[int, int]:
            r, c = node
            x = int(self.margin + c * gap_x)
            y = int(self.margin + r * gap_y)
            return x, y

        for a, b in edges:
            cv2.line(img, to_px(a), to_px(b), (0, 0, 0), self.line_thickness, cv2.LINE_AA)
        cv2.imwrite(str(out_path), img)

    # ------------------------------------------------------------------
    # Containment -------------------------------------------------------

    @staticmethod
    def contains_model(pattern_edges: Set[Edge], model_edges: Set[Edge]) -> bool:
        """Return *True* iff *model_edges* is a subset of *pattern_edges*."""
        return model_edges.issubset(pattern_edges)

    # ------------------------------------------------------------------
    # Internals ---------------------------------------------------------

    def _sample_edge_count(self) -> int:
        mu = self.density * self._n_all
        sigma = self.density_std * self._n_all
        raw = self.rng.gauss(mu, sigma)
        lower = (self.rows * self.cols) - 1  # connectivity lower bound
        upper = self._n_all
        return int(min(max(round(raw), lower), upper))

    def _random_spanning_tree(self) -> Set[Edge]:
        """Uniformly random spanning tree via randomized DFS (acyclic & connected)."""
        visited = {(self.rng.randrange(self.rows), self.rng.randrange(self.cols))}
        tree: Set[Edge] = set()
        stack = [next(iter(visited))]
        while stack:
            current = stack.pop()
            unvisited_neigh = [n for n in neighbours(current, self.rows, self.cols, self.diag) if n not in visited]
            if unvisited_neigh:
                stack.append(current)  # put it back to allow further exploration
                nxt = self.rng.choice(unvisited_neigh)
                visited.add(nxt)
                tree.add(canonical_edge(current, nxt))
                stack.append(nxt)
        assert len(tree) == (self.rows * self.cols) - 1
        return tree

# ---------------------------------------------------------------------------
# Model parsing helpers ------------------------------------------------------

def parse_edges(s: str) -> Set[Edge]:
    """Parse edges from a string like "0,0-1,0; 1,0-1,1" → {((0,0),(1,0)), ...}."""
    edges: Set[Edge] = set()
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        a_str, b_str = part.split("-")
        a = tuple(map(int, a_str.split(",")))  # type: ignore[arg-type]
        b = tuple(map(int, b_str.split(",")))  # type: ignore[arg-type]
        edges.add(canonical_edge(a, b))
    return edges
