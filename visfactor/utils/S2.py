"""Cube generator
------------------
Generate a perspective‑correct 2‑D view of a cube whose visible faces (up / front / right)
carry arbitrary tokens – single letters, multi‑line text, or simple geometric primitives
(circle, square, triangle; solid or hollow). Each face can also be rotated in 90° steps.

API
----------
>>> img = generate_cube(
        face_tokens=["X", "A", "triangle_hollow"],
        face_rotations=[0, 0, 90],
        size=300,
        offset_ratio=0.35,
    )
cv2.imwrite("cube.png", img)

Cube equivalence checker

Utilities to decide whether two partial observations (views) of a
labeled cube could correspond to the *same* physical cube.

Each view consists of exactly three visible faces – the *Up*, *Front*, and
*Right* faces from the observer’s perspective – along with the rotation of
what is printed on each face.  A rotation is expressed in degrees clockwise
relative to the local Up direction of that face (0, 90, 180 or 270).

Hidden faces may carry arbitrary symbols, but **no symbol appears on more
than one face of a given cube**.  Some symbols are symmetric under certain
rotations and therefore look identical after the cube is turned.  We model
those symmetries explicitly so that the algorithm recognises that, e.g., a
solid square rotated by 90° still looks like itself.

The public entry‑point is :func:`is_same_cube`.

--------------------------------------------------------------------------
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from itertools import product
from typing import Dict, List, Sequence, Tuple

import random

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Public data‑classes & enumerations
# ─────────────────────────────────────────────────────────────────────────────
class FaceRotation(IntEnum):
    """Allowed clockwise rotations (multiples of 90°)."""

    R0 = 0
    R90 = 90
    R180 = 180
    R270 = 270


Rotation = FaceRotation  # legacy alias


@dataclass(frozen=True, slots=True)
class CubeView:
    """A partial observation consisting of the 3 visible faces (Up, Front, Right)."""

    chars: Tuple[str, str, str]
    rots: Tuple[Rotation, Rotation, Rotation]

    def __post_init__(self) -> None:
        if len(self.chars) != 3 or len(self.rots) != 3:
            raise ValueError("CubeView must describe exactly three faces (U, F, R)")
        if len(set(self.chars)) != 3:
            raise ValueError("Symbols inside one view must be distinct")
        if not all(r in Rotation for r in self.rots):
            raise ValueError("Rotations must be 0, 90, 180 or 270 degrees")


# ─────────────────────────────────────────────────────────────────────────────
#  Symbol helpers – symmetry & drawing primitives
# ─────────────────────────────────────────────────────────────────────────────
SYMBOL_POOL: List[str] = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in {"O", "I"}] +
    [str(d) for d in range(1, 8)] +
    ["+", "-"] +
    [
        "circle", "circle_hollow",
        "square", "square_hollow",
        "triangle", "triangle_hollow",
    ]
)

SYMMETRY_CLASS: Dict[str, int] = {
    **{s: 4 for s in {"square", "square_hollow", "circle", "circle_hollow", "+"}},
    **{s: 2 for s in {"S", "H", "Z", "N", "X", "-"}},
}

# ─────────────────────────────────────────────────────────────────────────────
#  Render section – only OpenCV lives here
# ─────────────────────────────────────────────────────────────────────────────
_SHAPES = {
    "circle",
    "circle_hollow",
    "square",
    "square_hollow",
    "triangle",
    "triangle_hollow",
}


@lru_cache(maxsize=None)
def _shape_mask(shape_token: str, size: int) -> np.ndarray:
    """Return *binary* mask of a primitive centred in a square canvas."""

    canvas = np.zeros((size, size), np.uint8)
    cx = cy = size // 2
    radius = size // 4

    hollow = shape_token.endswith("_hollow")

    if shape_token.startswith("circle"):
        thickness = -1 if not hollow else max(1, radius // 6)
        cv2.circle(canvas, (cx, cy), radius, 255, thickness, cv2.LINE_AA)

    elif shape_token.startswith("square"):
        thickness = -1 if not hollow else max(1, radius // 6)
        half = radius
        cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), 255, thickness, cv2.LINE_AA)

    elif shape_token.startswith("triangle"):
        # OpenCV does **not** allow negative thickness for polylines, therefore we
        # always draw the outline with a positive width and optionally fill later.
        outline = max(1, radius // 6)
        half = int(radius * 1.15)
        pts = np.int32([
            [cx, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ])
        cv2.polylines(canvas, [pts], True, 255, outline, cv2.LINE_AA)
        if not hollow:
            cv2.fillPoly(canvas, [pts], 255)

    else:
        raise ValueError(f"Unknown shape token: {shape_token!r}")
    return canvas


def _text_mask(token: str, size: int, font_scale: float, thickness: int) -> np.ndarray:
    canvas = np.zeros((size, size), np.uint8)
    lines = token.split("\n")
    line_sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for l in lines]
    full_h = sum(h for (_, h) in line_sizes) + (len(lines) - 1) * int(line_sizes[0][1] * 0.3)
    y = (size - full_h) // 2 + line_sizes[0][1]
    for (w, h), line in zip(line_sizes, lines):
        x = (size - w) // 2
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
        y += h + int(h * 0.3)
    return canvas


def _make_token_canvas(token: str, size: int, rotation: Rotation, colour: Tuple[int, int, int], *,
                       font_scale: float = 6.0, font_thickness: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    token_lower = token.lower()
    if token_lower in _SHAPES:
        mask = _shape_mask(token_lower, size).copy()
    else:
        mask = _text_mask(token, size, font_scale, font_thickness)

    if rotation == Rotation.R90:
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Rotation.R180:
        mask = cv2.rotate(mask, cv2.ROTATE_180)
    elif rotation == Rotation.R270:
        mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

    rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    rgb[mask > 0] = colour
    return rgb, mask

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry helpers – 3‑D → 2‑D projection
# ─────────────────────────────────────────────────────────────────────────────
Vec3 = Tuple[int, int, int]

_FACE_VEC: Dict[str, Vec3] = {
    "U": (0, 0, 1),
    "D": (0, 0, -1),
    "F": (0, 1, 0),
    "B": (0, -1, 0),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
}
_VEC_FACE = {v: k for k, v in _FACE_VEC.items()}


@lru_cache(maxsize=24)
def _orientations() -> Tuple[Tuple[Vec3, Vec3, Vec3], ...]:
    out = []
    for up, front in product(_FACE_VEC.values(), repeat=2):
        if up == front or any(a * b for a, b in zip(up, front)):
            continue
        right = (
            front[1] * up[2] - front[2] * up[1],
            front[2] * up[0] - front[0] * up[2],
            front[0] * up[1] - front[1] * up[0],
        )
        if right not in _VEC_FACE:
            continue
        out.append((up, front, right))
    assert len(out) == 24
    return tuple(out)

_BASE_AXES: Dict[str, Tuple[Vec3, Vec3]] = {
    "U": ((0, -1, 0), (1, 0, 0)),
    "D": ((0, 1, 0), (1, 0, 0)),
    "F": ((0, 0, 1), (1, 0, 0)),
    "B": ((0, 0, 1), (-1, 0, 0)),
    "R": ((0, 0, 1), (0, -1, 0)),
    "L": ((0, 0, 1), (0, 1, 0)),
}

_VIEW_POSITIONS: Tuple[str, ...] = ("U", "F", "R")


def _rotation_offset(base_up: Vec3, base_right: Vec3, curr_up: Vec3) -> Rotation:
    seq = [base_up, base_right, tuple(-x for x in base_up), tuple(-x for x in base_right)]
    return Rotation((seq.index(curr_up) * 90) % 360)  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
#  Cube equivalence – main algorithm
# ─────────────────────────────────────────────────────────────────────────────

def _rot_eq(tok: str, r1: Rotation, r2: Rotation) -> bool:
    diff = (r1 - r2) % 360
    sym = SYMMETRY_CLASS.get(tok, 1)
    return sym == 4 or (sym == 2 and diff % 180 == 0) or diff == 0


def is_same_cube(a: CubeView, b: CubeView) -> bool:
    baseline = {pos: (ch, rot) for pos, ch, rot in zip(_VIEW_POSITIONS, a.chars, a.rots)}
    sym_face = {ch: pos for pos, ch in zip(_VIEW_POSITIONS, a.chars)}

    for up, front, right in _orientations():
        fi = dict(baseline)
        cf = dict(sym_face)
        ok = True
        for pos, ch, rot in zip(_VIEW_POSITIONS, b.chars, b.rots):
            face = _VEC_FACE[{"U": up, "F": front, "R": right}[pos]]
            base_up, base_rt = _BASE_AXES[face]
            curr_up = (
                tuple(-f for f in front) if pos == "U" else up if pos == "F" else up
            )
            r_ref = Rotation((rot + _rotation_offset(base_up, base_rt, curr_up)) % 360)  # type: ignore[arg-type]
            if face in fi:
                ch0, r0 = fi[face]
                if ch0 != ch or not _rot_eq(ch0, r0, r_ref):
                    ok = False
                    break
            if ch in cf and cf[ch] != face:
                ok = False
                break
            fi[face] = (ch, r_ref)
            cf[ch] = face
        if ok:
            return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
#  Public: cube image renderer
# ─────────────────────────────────────────────────────────────────────────────

def generate_cube(
    view: CubeView,
    *,
    size: int = 200,
    depth_ratio: float = 0.4,
    background: Tuple[int, int, int] = (255, 255, 255),
    face_colour: Tuple[int, int, int] = (255, 255, 255),
    edge_colour: Tuple[int, int, int] = (0, 0, 0),
    token_colour: Tuple[int, int, int] = (0, 0, 0),
    edge_thickness: int = 2,
    font_scale: float = 6.0,
    font_thickness: int = 8,
) -> np.ndarray:
    up_tok, front_tok, right_tok = view.chars
    up_rot, front_rot, right_rot = view.rots

    depth = int(size * depth_ratio)
    pad = size // 2
    h = size + abs(depth) + pad * 2
    w = size + depth + pad * 2
    img = np.full((h, w, 3), background, np.uint8)

    p0 = (pad, pad + abs(depth))
    p1 = (p0[0] + size, p0[1])
    p2 = (p1[0], p1[1] + size)
    p3 = (p0[0], p0[1] + size)

    dx, dy = depth, -depth
    p4 = (p0[0] + dx, p0[1] + dy)
    p5 = (p1[0] + dx, p1[1] + dy)
    p6 = (p5[0], p5[1] + size)

    cv2.fillPoly(img, [np.int32([p4, p5, p1, p0])], face_colour)
    cv2.fillPoly(img, [np.int32([p1, p5, p6, p2])], face_colour)
    cv2.fillPoly(img, [np.int32([p0, p1, p2, p3])], face_colour)

    def warp(tok: str, rot: Rotation, quad: Sequence[Tuple[int, int]]):
        rgb, msk = _make_token_canvas(tok, size, rot, token_colour, font_scale=font_scale, font_thickness=font_thickness)
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [size, 0], [size, size], [0, size]]), np.float32(quad))
        wimg = cv2.warpPerspective(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        wmsk = cv2.warpPerspective(msk, M, (w, h), flags=cv2.INTER_NEAREST)
        img[wmsk > 0] = wimg[wmsk > 0]

    warp(front_tok, front_rot, [p0, p1, p2, p3])
    warp(up_tok, up_rot, [p4, p5, p1, p0])
    warp(right_tok, right_rot, [p1, p5, p6, p2])

    for a, b in [
        (p0, p1), (p1, p2), (p2, p3), (p3, p0),
        (p0, p4), (p1, p5), (p4, p5),
        (p1, p2), (p2, p6), (p5, p6), (p1, p5),
    ]:
        cv2.line(img, a, b, edge_colour, edge_thickness, cv2.LINE_AA)

    return img

# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────
_ANGLE_POOL: Tuple[Rotation, ...] = (Rotation.R0, Rotation.R90, Rotation.R180, Rotation.R270)

def _random_view() -> CubeView:
    return CubeView(tuple(random.sample(SYMBOL_POOL, 3)), tuple(random.choices(_ANGLE_POOL, k=3)))

def generate_cube_pairs() -> Tuple[CubeView, CubeView]:
    """Return two independent random CubeViews (useful for demos/tests)."""
    return _random_view(), _random_view()
