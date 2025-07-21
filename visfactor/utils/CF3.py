"""Grid random walker

Generate an m×n grid image with one point circled as the start, and a
corresponding image showing a random walk constrained to the grid.

Key features
------------
* Fully parameterised grid size, dot radius, random‑walk length range.
* Clean, object‑oriented API (`GridWalkGenerator`) separates geometry,
  drawing and walk logic.
* Deterministic output option via `--seed` for reproducibility.
* Generates **two** images per item:
    1. `<idx>_grid.png` – bare grid with the start point marked (sample ②).
    2. `<idx>_path.png` – grid plus random‑walk path (sample ①).
* Outputs:
    - `Images/ans.txt` list of end‑point coordinates.
    - `Images/questions_answers.json` dataset mapping questions→answers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------- #
#                           Configuration                                #
# ---------------------------------------------------------------------- #

@dataclass
class GridConfig:
    rows: int = 5
    cols: int = 5
    size: int = 600           # width & height (px)
    dot_radius: int = 15      # grid‑dot radius (px)
    line_thickness: int = 3   # path thickness (px)

    @property
    def cell_w(self) -> float:
        return self.size / self.cols

    @property
    def cell_h(self) -> float:
        return self.size / self.rows

# ---------------------------------------------------------------------- #
#                         Utility functions                              #
# ---------------------------------------------------------------------- #

def _cross(ax: int, ay: int, bx: int, by: int) -> int:
    """2‑D cross product (z‑component) for integer pairs."""
    return ax * by - ay * bx

# ---------------------------------------------------------------------- #
#                         Grid‑Walk Generator                            #
# ---------------------------------------------------------------------- #

class GridWalkGenerator:
    def __init__(
        self,
        grid: GridConfig,
        min_steps: int = 4,
        max_steps: int = 8,
        seed: int | None = None,
    ) -> None:
        if min_steps < 0 or max_steps < min_steps:
            raise ValueError("Invalid step range")
        self.cfg = grid
        self.min_steps = min_steps
        self.max_steps = max_steps
        if seed is not None:
            random.seed(seed)

        # cache coordinates (row‑major)
        self.pts: List[Tuple[int, int]] = [
            (
                int(self.cfg.cell_w * (c + 0.5)),
                int(self.cfg.cell_h * (r + 0.5)),
            )
            for r in range(self.cfg.rows)
            for c in range(self.cfg.cols)
        ]

    # ------------------------------------------------------------------ #
    #                     Collinearity checking                           #
    # ------------------------------------------------------------------ #
    def _is_collinear_with_path(self, curr: int, cand: int, path: List[int]) -> bool:
        """Return *True* iff segment (curr→cand) lies on same infinite line as **any**
        existing segment in *path*.
        """
        x0, y0 = self.pts[curr]
        xc, yc = self.pts[cand]
        vx, vy = xc - x0, yc - y0  # new segment vector

        for i in range(len(path) - 1):
            p, q = path[i], path[i + 1]
            x1, y1 = self.pts[p]
            x2, y2 = self.pts[q]
            wx, wy = x2 - x1, y2 - y1  # existing segment vector

            # 1) 方向平行 (cross = 0)
            if _cross(vx, vy, wx, wy) != 0:
                continue
            # 2) 新 segment 與此線在同一直線上：任取一點差向量也平行
            if _cross(x0 - x1, y0 - y1, wx, wy) == 0:
                return True
        return False

    # ------------------------------------------------------------------ #
    #                         Random walk                                 #
    # ------------------------------------------------------------------ #
    def _random_walk(self, start_idx: int) -> List[int]:
        max_len = random.randint(self.min_steps, self.max_steps)
        path = [start_idx]
        visited = {start_idx}

        for _ in range(max_len):
            remaining = [i for i in range(len(self.pts)) if i not in visited]
            # 過濾掉與既有線段共線的選項
            candidates = [
                i for i in remaining if not self._is_collinear_with_path(path[-1], i, path)
            ]
            if not candidates:
                break  # 無合法延伸→提早結束
            nxt = random.choice(candidates)
            path.append(nxt)
            visited.add(nxt)
        return path

    # ------------------------------------------------------------------ #
    #                            Drawing                                  #
    # ------------------------------------------------------------------ #
    def _draw_grid_points(self, img: np.ndarray) -> None:
        for x, y in self.pts:
            cv2.circle(img, (x, y), self.cfg.dot_radius // 2, (0, 0, 0), -1)

    def _draw_start_circle(self, img: np.ndarray, idx: int) -> None:
        x, y = self.pts[idx]
        cv2.circle(img, (x, y), self.cfg.dot_radius, (0, 0, 0), 2)
        cv2.circle(img, (x, y), self.cfg.dot_radius // 2, (0, 0, 0), -1)

    def _draw_path(self, img: np.ndarray, path: List[int]) -> None:
        for a, b in zip(path[:-1], path[1:]):
            cv2.line(img, self.pts[a], self.pts[b], (0, 0, 0), self.cfg.line_thickness)

    # ------------------------------------------------------------------ #
    #                       Public interface                              #
    # ------------------------------------------------------------------ #
    def generate_pair(self, idx: int, savepath: str) -> Tuple[str, str, Tuple[int, int]]:
        start_idx = random.randrange(len(self.pts))
        path = self._random_walk(start_idx)
        end_idx = path[-1]

        H = W = self.cfg.size
        grid_img = np.full((H, W, 3), 255, np.uint8)
        path_img = grid_img.copy()

        self._draw_grid_points(grid_img)
        self._draw_start_circle(grid_img, start_idx)

        self._draw_start_circle(path_img, start_idx)
        self._draw_path(path_img, path)

        cv2.imwrite(savepath.replace(".png", "-1.png"), grid_img)
        cv2.imwrite(savepath.replace(".png", "-0.png"), path_img)

        end_row = end_idx // self.cfg.cols + 1
        end_col = end_idx % self.cfg.cols + 1
        return (end_row, end_col)
