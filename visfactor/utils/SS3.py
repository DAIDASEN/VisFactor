from __future__ import annotations

"""Street‑route puzzle generator
================================

This module builds a *m × n* city grid, randomly blocks road segments with circular
"road‑blocks", scatters numbered buildings and then samples two perimeter
intersections (*START*, *END*) that admit **exactly one** shortest street route.
That route – together with the map itself – can be plotted in a style similar to
classic contest diagrams (black streets, hollow circles for blocks, grey
quarter‑square buildings and perimeter letters).

All path‑finding and graph operations use **NetworkX**, which makes the code
considerably simpler, faster and easier to reason about compared with manual
BFS/DFS bookkeeping.

Usage example
-------------
```python
if __name__ == "__main__":
    city = City(rows=6, cols=7, blocked_ratio=0.18, num_buildings=10, seed=42)

    print("START:", city.start_label, "END:", city.end_label)
    print("Shortest length:", len(city.path) - 1)
    print("Buildings crossed:", sorted(city.crossed_buildings))

    city.draw()
```

The constructor arguments are:
    rows, cols      : grid size in *cells* (rows × cols)  
    blocked_ratio   : fraction of street segments to block (0 ≤ r < 1)  
    num_buildings   : number of quarter‑square buildings to place  
    seed            : optional PRNG seed for reproducibility

You can also import the class and embed it in larger generators, web apps, etc.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import itertools
import math
import random
import string

import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _idx_to_letters(idx: int) -> str:
    """Spreadsheet‑style column encoding: 0 → 'A', 25 → 'Z', 26 → 'AA', ..."""
    out = []
    idx += 1  # Make it 1‑based
    while idx:
        idx -= 1
        idx, rem = divmod(idx, 26)
        out.append(chr(ord("A") + rem))
    return "".join(reversed(out))


@dataclass(frozen=True, slots=True)
class Building:
    """Quarter‑square building sitting in a cell quadrant."""

    id: int
    cell: Tuple[int, int]  # (x, y) lower‑left corner of the *cell*
    quadrant: str          # 'NE', 'NW', 'SW', 'SE'

    @property
    def touched_edges(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return the two undirected edges of the grid street that the building touches."""
        x, y = self.cell
        north = ((x, y + 1), (x + 1, y + 1))
        south = ((x, y), (x + 1, y))
        west = ((x, y), (x, y + 1))
        east = ((x + 1, y), (x + 1, y + 1))
        match self.quadrant:
            case "NE":
                return [north, east]
            case "NW":
                return [north, west]
            case "SW":
                return [south, west]
            case "SE":
                return [south, east]
        raise ValueError(f"Unknown quadrant: {self.quadrant}")


class City:
    """Random street‑route puzzle backed by a NetworkX graph."""

    def __init__(
        self,
        rows: int,
        cols: int,
        blocked_ratio: float = 0.15,
        num_buildings: int = 10,
        seed: int | None = None,
    ) -> None:
        self.rows = rows - 1
        self.cols = cols - 1
        self.rng = random.Random(seed)

        self.G = nx.Graph()
        self._build_full_grid()
        self._block_edges(blocked_ratio)
        self._place_buildings(num_buildings)
        self._label_perimeter()
        self._choose_terminals()
        self._extract_unique_path_and_buildings()

    # ---------------------------------------------------------------------
    # Grid construction & manipulation
    # ---------------------------------------------------------------------

    def _build_full_grid(self) -> None:
        """Create a (cols+1)×(rows+1) lattice of intersections with all streets present."""
        for x in range(self.cols + 1):
            for y in range(self.rows + 1):
                self.G.add_node((x, y))
                if x < self.cols:
                    self.G.add_edge((x, y), (x + 1, y))  # horizontal
                if y < self.rows:
                    self.G.add_edge((x, y), (x, y + 1))  # vertical

    def _block_edges(self, ratio: float) -> None:
        """Randomly mark a fraction of edges as blocked (but keep them for drawing)."""
        all_edges = list(self.G.edges)
        k = int(len(all_edges) * ratio)
        self.blocked_edges = set(self.rng.sample(all_edges, k))
        # Physically remove them from the graph (= cannot traverse)
        self.G.remove_edges_from(self.blocked_edges)

    # ---------------------------------------------------------------------
    # Buildings
    # ---------------------------------------------------------------------

    def _place_buildings(self, n: int) -> None:
        max_possible = self.rows * self.cols
        n = min(n, max_possible)
        cells = self.rng.sample(
            [(x, y) for x in range(self.cols) for y in range(self.rows)], n
        )
        quadrants = ["NE", "NW", "SW", "SE"]
        self.buildings: Dict[int, Building] = {}
        self.edge_to_buildings: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Set[int]] = defaultdict(set)
        for idx, (x, y) in enumerate(cells, start=1):
            q = self.rng.choice(quadrants)
            b = Building(idx, (x, y), q)
            self.buildings[idx] = b
            for e in b.touched_edges:
                self.edge_to_buildings[tuple(sorted(e))].add(idx)

    # ---------------------------------------------------------------------
    # Perimeter labelling & terminal sampling
    # ---------------------------------------------------------------------

    def _label_perimeter(self) -> None:
        top = [((x, self.rows)) for x in range(self.cols + 1)]
        right = [((self.cols, y)) for y in range(self.rows - 1, 0, -1)]
        bottom = [((x, 0)) for x in range(self.cols, -1, -1)]
        left = [((0, y)) for y in range(1, self.rows)]
        self.perimeter_nodes = top + right + bottom + left
        self.node_labels: Dict[Tuple[int, int], str] = {
            node: _idx_to_letters(i) for i, node in enumerate(self.perimeter_nodes)
        }

    def _choose_terminals(self) -> None:
        candidates = self.perimeter_nodes.copy()
        self.rng.shuffle(candidates)

        for start in candidates:
            for end in candidates:
                if start[0] == end[0] or start[1] == end[1]:
                    continue
                try:
                    length = nx.shortest_path_length(self.G, start, end)
                except nx.NetworkXNoPath:
                    continue
                # We need *exactly one* shortest path
                paths = list(nx.all_shortest_paths(self.G, start, end, length))
                if len(paths) == 1:
                    self.start = start
                    self.end = end
                    self.path = paths[0]
                    return
        raise RuntimeError("Failed to find a unique shortest path – try different seed or parameters")

    # ---------------------------------------------------------------------
    # Path & traversed buildings
    # ---------------------------------------------------------------------

    def _extract_unique_path_and_buildings(self) -> None:
        edges_on_path = [tuple(sorted(e)) for e in zip(self.path[:-1], self.path[1:])]
        crossed = set()
        for e in edges_on_path:
            crossed.update(self.edge_to_buildings.get(e, ()))
        self.crossed_buildings: Set[int] = crossed
        # Convenience aliases
        self.start_label = self.node_labels[self.start]
        self.end_label = self.node_labels[self.end]

    # ---------------------------------------------------------------------
    # Drawing
    # ---------------------------------------------------------------------

    def draw(self, savepath: str | None = None) -> None:
        fig, ax = plt.subplots(figsize=(self.cols * 0.9, self.rows * 0.9))

        # 1) Draw full grid (streets)
        for x in range(self.cols + 1):
            ax.plot([x, x], [0, self.rows], color="black", linewidth=1.2)
        for y in range(self.rows + 1):
            ax.plot([0, self.cols], [y, y], color="black", linewidth=1.2)

        # 2) Road‑blocks (circles)
        for u, v in self.blocked_edges:
            mx = (u[0] + v[0]) / 2
            my = (u[1] + v[1]) / 2
            ax.plot(mx, my, "o", ms=10, mfc="white", mec="black", mew=1.5)

        # 3) Buildings (light‑grey quarter squares with numbers)
        half = 0.5
        for b in self.buildings.values():
            x, y = b.cell
            dx = 0 if "W" in b.quadrant else half
            dy = 0 if "S" in b.quadrant else half
            rect = plt.Rectangle((x + dx, y + dy), half, half, fc="lightgrey", ec="black")
            ax.add_patch(rect)
            ax.text(
                x + dx + half / 2,
                y + dy + half / 2,
                str(b.id),
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

        # 4) Perimeter labels
        off = 0.28
        for node, label in self.node_labels.items():
            x, y = node
            if y == self.rows:  # top
                ax.text(x, y + off, label, ha="center", va="bottom", weight="bold")
            elif y == 0:  # bottom
                ax.text(x, y - off, label, ha="center", va="top", weight="bold")
            elif x == self.cols:  # right
                ax.text(x + off, y, label, ha="left", va="center", weight="bold")
            elif x == 0:  # left
                ax.text(x - off, y, label, ha="right", va="center", weight="bold")

        # Aesthetics
        ax.set_xlim(-1, self.cols + 1)
        ax.set_ylim(-1, self.rows + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

        # 5) Highlight shortest path
        xs, ys = zip(*self.path)
        ax.plot(xs, ys, "r-", lw=3, marker="o", ms=7, mfc="red")

        # Annotate start/end on the path for clarity
        ax.text(xs[0], ys[0], "S", color="white", ha="center", va="center", weight="bold")
        ax.text(xs[-1], ys[-1], "E", color="white", ha="center", va="center", weight="bold")
        plt.savefig(savepath.replace(".png", "_answer.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
