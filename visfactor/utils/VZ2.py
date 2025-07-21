"""Paper Folding Sequence Generator
--------------------------------
This script programmatically creates a series of figures that imitate the classic
fold‑and‑hole paper‑folding puzzles often found in IQ tests.

Given a square sheet of paper represented on an *n × n* grid, the script:
1. Randomly decides on **min_steps – max_steps** folds.
2. For each fold it randomly picks a fold axis that is either **horizontal**, **vertical**,
   or **45° diagonal** (the two main diagonals of the square).  The half‑plane containing
   the square's centre remains fixed; the opposite half is reflected over the axis.
3. After completing all folds, a single hole is punched at a random point strictly
   inside the (multi‑layer) folded shape.
4. The script then **unfolds** the paper step‑by‑step in reverse order, showing how the
   hole propagates.
5. Every intermediate state (each fold, the hole‑punch, and every unfold) is plotted
   as a separate PNG file.  Solid black lines trace the current outline; dashed lines
   show the original square.

Output
------
All images are stored in an automatically created "output" sub‑directory with
sequential filenames such as `step_00_initial.png`, `step_01_fold.png`, …, and so on.

Dependencies
------------
* **matplotlib** – for plotting
* **shapely**     – for geometric operations (splitting, reflection, union)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import split, unary_union

import os
import glob
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _reflect_point(pt: Tuple[float, float], axis: Dict) -> Tuple[float, float]:
    """Reflect a single point across the given fold axis."""
    x, y = pt
    kind = axis["type"]
    size = axis["size"]
    c = axis["offset"]

    if kind == "horizontal":
        return (x, 2 * c - y)

    if kind == "vertical":
        return (2 * c - x, y)

    if kind == "diag_pos":  # y = x + c
        return (y - c, x + c)

    if kind == "diag_neg":  # y = −x + c
        return (c - y, c - x)

    raise ValueError(f"Unknown axis type: {kind}")


def _reflect_linestring(ls: LineString, axis: Dict) -> LineString:
    """Return a reflected copy of *ls* across *axis*."""
    return LineString([_reflect_point(pt, axis) for pt in ls.coords])


def _reflect_polygon(poly: Polygon, axis: Dict) -> Polygon:
    """Reflect an entire polygon across the axis."""
    reflected_coords = [_reflect_point(pt, axis) for pt in poly.exterior.coords]
    return Polygon(reflected_coords)


def _make_intersection(ls: LineString, poly: Polygon) -> List:
    intersection = ls.intersection(poly)
    if not intersection.is_empty:
        if isinstance(intersection, LineString):
            return [intersection]
        elif isinstance(intersection, MultiLineString):
            return list(intersection.geoms)
    return []


def _generate_point(poly: Polygon, size: float, *, edges: List[LineString] | None = None, points: List[Point] | None = None, min_d: float = 0.2) -> Point:
    if edges == None:
        coords = list(poly.exterior.coords)
        edges = [LineString([coords[i], coords[i+1]]) for i in range(len(coords) - 1)]
    while True:
        candidate = Point(random.uniform(0, size), random.uniform(0, size))
        min_distance = min(e.distance(candidate) for e in edges)
        if points:
            min_distance_points = min(p.distance(candidate) for p in points)
            min_distance = min(min_distance, min_distance_points)
        if poly.contains(candidate) and min_distance >= min_d:
            return candidate


def _axis_geometry(axis: Dict) -> LineString:
    """Return a Shapely *LineString* representing the fold axis (long enough)."""
    kind = axis["type"]
    size = axis["size"]
    c = axis["offset"]

    if kind == "horizontal":
        return LineString([(0, c), (size, c)])

    if kind == "vertical":
        return LineString([(c, 0), (c, size)])

    if kind == "diag_pos":  # y = x + c
        return LineString([(-size, -size + c), (2 * size, 2 * size + c)])

    if kind == "diag_neg":  # y = −x + c
        return LineString([(-size, size + c), (2 * size, -2 * size + c)])

    raise ValueError(f"Unknown axis type: {kind}")


# ──────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ──────────────────────────────────────────────────────────────────────────────

def generate_sequence(
    n: int = 3,
    min_steps: int = 3,
    max_steps: int = 5,
    seed: int | None = None,
    *,
    save_dir: str | Path = "output",
) -> None:
    """Generate folding & unfolding sequence with internal edge rendering."""

    # ── setup ────────────────────────────────────────────────────────────────
    random.seed(seed)
    size: float = float(n)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    square = Polygon([(0, 0), (size, 0), (size, size), (0, size)])

    shapes: List[Polygon] = [square]
    edges_history: List[List[LineString]] = [[
        LineString([(0, 0), (size, 0)]),
        LineString([(size, 0), (size, size)]),
        LineString([(size, size), (0, size)]),
        LineString([(0, size), (0, 0)]),
    ]]
    axes: List[Dict] = []

    # ── choose folds ─────────────────────────────────────────────────────────
    num_folds = random.randint(min_steps, max_steps)
    grid_lines = [float(i) for i in range(1, n)]
    diag_pos_lines = [float(i) for i in range(1 - n, n)]
    diag_neg_lines = [float(i) for i in range(1, 2 * n)]

    centre_pt = Point(size / 2, size / 2)
    current_poly: Polygon = square
    current_edges: List[LineString] = edges_history[0]

    # ── folding loop ─────────────────────────────────────────────────────────
    while len(axes) < num_folds:
        axis_type = random.choice(["horizontal", "vertical", "diag_pos", "diag_neg",])

        if axis_type in ("horizontal", "vertical"):
            coord = random.choice(grid_lines)
            axis = {"type": axis_type, "offset": coord, "size": size}
        elif axis_type == "diag_pos":
            coord = random.choice(diag_pos_lines)
            axis = {"type": axis_type, "offset": coord, "size": size}
        elif axis_type == "diag_neg":
            coord = random.choice(diag_neg_lines)
            axis = {"type": axis_type, "offset": coord, "size": size}

        axis_line = _axis_geometry(axis)
        pieces = list(split(current_poly, axis_line).geoms)
        if len(pieces) != 2:
            continue  # degenerate split, retry

        # Identify stationary & moving halves
        if pieces[0].contains(centre_pt):
            stationary, to_fold = pieces[0], pieces[1]
        elif pieces[1].contains(centre_pt):
            stationary, to_fold = pieces[1], pieces[0]
        else:
            stationary, to_fold = (
                (pieces[0], pieces[1])
                if pieces[0].distance(centre_pt) <= pieces[1].distance(centre_pt)
                else (pieces[1], pieces[0])
            )
        new_poly = unary_union([stationary, _reflect_polygon(to_fold, axis)])

        # ── edge transformation ────────────────────────────────────────────
        new_edges: List[LineString] = []
        for e in current_edges:
            new_edges.extend(_make_intersection(e, new_poly))
            reflect_e = _reflect_linestring(e, axis)
            new_edges.extend(_make_intersection(reflect_e, new_poly))
        new_edges.extend(_make_intersection(axis_line, new_poly))

        # Update state
        current_edges = new_edges
        current_poly = new_poly
        shapes.append(current_poly)
        edges_history.append(current_edges)
        axes.append(axis)

    # ── rendering ────────────────────────────────────────────────────────────
    for i in range(1, len(shapes)):
        _plot_state(square, shapes[i], edges_history[i], [], save_path / f"step_{i:02d}_fold.png")

    # ── punch hole ───────────────────────────────────────────────────────────
    hole_point = _generate_point(current_poly, size, edges=edges_history[-1])
    _plot_state(square, shapes[-1], edges_history[-1], [hole_point], save_path / f"step_{len(shapes):02d}_hole.png",)

    # ── unfolding ────────────────────────────────────────────────────────────
    holes: List[Point] = [hole_point]
    wrong_holes: List[List[Point]] = []

    if num_folds == 1:
        candidate = _generate_point(shapes[1], size, points=holes)
        wrong_holes.append([i for i in holes] + [candidate])

        candidate = _generate_point(shapes[1], size, points=holes)
        wrong_holes.append([i for i in holes] + [candidate])

        candidate = _generate_point(shapes[1], size, points=holes)
        wrong_holes.append([candidate])

        candidate = _generate_point(shapes[1], size, points=holes)
        wrong_holes.append([candidate])

    for idx, axis in enumerate(reversed(axes), start=1):
        shape_to_plot = shapes[-(idx + 1)]
        edges_to_plot = edges_history[-(idx + 1)]

        new_holes: List[Point] = []
        seen: set[Tuple[float, float]] = set()
        for h in holes:
            if shape_to_plot.contains(h):
                key = (round(h.x, 6), round(h.y, 6))
                if key not in seen:
                    seen.add(key)
                    new_holes.append(h)
            h_ref = Point(*_reflect_point((h.x, h.y), axis))
            if shape_to_plot.contains(h_ref):
                key = (round(h_ref.x, 6), round(h_ref.y, 6))
                if key not in seen:
                    seen.add(key)
                    new_holes.append(h_ref)
        holes = new_holes
        _plot_state(square, shape_to_plot, edges_to_plot, holes, save_path / f"step_{len(shapes)+idx:02d}_unfold.png",)

        if idx == num_folds - 1:
            candidate = _generate_point(shapes[1], size, points=new_holes)
            wrong_holes.append([i for i in new_holes] + [candidate])

            chosen = random.randrange(len(new_holes))
            if len(new_holes) > 1:
                wrong_holes.append([new_holes[i] for i in range(len(new_holes)) if i != chosen])
            else:
                candidate = _generate_point(shapes[1], size, points=new_holes)
                wrong_holes.append([candidate])

            chosen = random.randrange(len(new_holes))
            candidate = _generate_point(shapes[1], size, points=new_holes)
            wrong_holes.append([new_holes[i] for i in range(len(new_holes)) if i != chosen] + [candidate])

            chosen = random.randrange(len(new_holes))
            candidate = _generate_point(shapes[1], size, points=new_holes)
            wrong_holes.append([new_holes[i] for i in range(len(new_holes)) if i != chosen] + [candidate])

        if idx == num_folds:
            for wrong_idx, wrong_hole in enumerate(wrong_holes):
                new_holes: List[Point] = []
                seen: set[Tuple[float, float]] = set()
                for h in wrong_hole:
                    if shape_to_plot.contains(h):
                        key = (round(h.x, 6), round(h.y, 6))
                        if key not in seen:
                            seen.add(key)
                            new_holes.append(h)
                    h_ref = Point(*_reflect_point((h.x, h.y), axis))
                    if shape_to_plot.contains(h_ref):
                        key = (round(h_ref.x, 6), round(h_ref.y, 6))
                        if key not in seen:
                            seen.add(key)
                            new_holes.append(h_ref)
                _plot_state(square, shape_to_plot, edges_to_plot, new_holes, save_path / f"wrong_choice_{wrong_idx}.png",)


def _concatenate_images_horizontally(image_paths):
    images = [Image.open(p) for p in image_paths]
    heights = [img.height for img in images]
    max_height = max(heights)
    total_width = sum(img.width for img in images)
    
    new_image = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return new_image


def process_images(dir_path, idx):
    idx_dir = os.path.join(dir_path, str(idx))

    # Question Image
    fold_images = sorted(glob.glob(os.path.join(idx_dir, "step_*_fold.png")))
    hole_image = glob.glob(os.path.join(idx_dir, "step_*_hole.png"))
    if hole_image:
        images_for_question = fold_images + [hole_image[0]]
        question_image = _concatenate_images_horizontally(images_for_question)
        question_image.save(os.path.join(dir_path, f"{idx}_question.png"))

    # Answer Image
    unfold_images = sorted(glob.glob(os.path.join(idx_dir, "step_*_unfold.png")))
    if unfold_images:
        answer_image = _concatenate_images_horizontally(unfold_images)
        answer_image.save(os.path.join(dir_path, f"{idx}_answer.png"))

        # Correct Choice Image (largest step)
        max_step_image = max(unfold_images, key=lambda x: int(os.path.basename(x).split('_')[1]))
        Image.open(max_step_image).save(os.path.join(dir_path, f"{idx}_correct_choice.png"))


# ──────────────────────────────────────────────────────────────────────────────
# Plotting utility
# ──────────────────────────────────────────────────────────────────────────────

def _plot_state(
    full_square: Polygon,
    poly: Polygon,
    edges: List[LineString],
    holes: List[Point],
    filename: Path,
) -> None:
    """Render one state with internal edges."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, full_square.bounds[2] + 0.1)
    ax.set_ylim(-0.1, full_square.bounds[3] + 0.1)
    ax.axis("off")

    # Original reference square (dashed)
    xs, ys = full_square.exterior.xy
    ax.plot(xs, ys, linestyle="--", linewidth=1, color="black")

    # Current outer outline (bold)
    xs, ys = poly.exterior.xy
    ax.plot(xs, ys, linewidth=1.8, color="black")

    # Internal edges & creases (thin)
    for e in edges:
        xs, ys = e.xy
        ax.plot(xs, ys, linewidth=1, color="black")

    # Holes
    for h in holes:
        ax.plot(h.x, h.y, "o", markersize=10, markerfacecolor="white", markeredgecolor="black")

    fig.tight_layout(pad=0)
    fig.savefig(filename, dpi=150)
    plt.close(fig)
