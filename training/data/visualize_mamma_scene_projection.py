#!/usr/bin/env python3
"""Visualize MAMMA BEDLAM .data.pyd projections on scene images."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = [
    (0, 230, 150, 220),
    (60, 190, 255, 220),
    (255, 220, 40, 220),
    (255, 70, 70, 220),
]


def project_points(intrinsics: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    uvw = (intrinsics @ points_3d.T).T
    return uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)


def load_landmark_indices(path: str | None, num_points: int) -> np.ndarray | None:
    if not path:
        return None
    pkl_path = Path(path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"landmark index file not found: {pkl_path}")
    try:
        import joblib

        matrix = joblib.load(pkl_path)
    except Exception:
        with pkl_path.open("rb") as f:
            matrix = pickle.load(f)
    if hasattr(matrix, "numpy"):
        matrix = matrix.numpy()
    matrix = np.asarray(matrix)
    if matrix.ndim == 2:
        indices = matrix.argmax(axis=-1)
    else:
        indices = matrix.astype(np.int64).reshape(-1)
    if indices.shape[0] >= num_points:
        return indices[:num_points]
    return indices


def draw_dot(draw: ImageDraw.ImageDraw, x: float, y: float, color, radius: int = 1) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def draw_ring(draw: ImageDraw.ImageDraw, x: float, y: float, color, radius: int = 3) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def crop_box_from_center_scale(
    center: np.ndarray,
    scale: float,
    image_size: tuple[int, int],
    crop_scale: float,
) -> tuple[int, int, int, int]:
    side = float(scale) * 200.0 * crop_scale
    left = int(np.floor(center[0] - side / 2))
    top = int(np.floor(center[1] - side / 2))
    right = int(np.ceil(center[0] + side / 2))
    bottom = int(np.ceil(center[1] + side / 2))
    left = max(0, left)
    top = max(0, top)
    right = min(image_size[0], right)
    bottom = min(image_size[1], bottom)
    return left, top, right, bottom


def evenly_sample(indices: np.ndarray, count: int) -> np.ndarray:
    if len(indices) <= count:
        return indices
    return indices[np.linspace(0, len(indices) - 1, count).astype(np.int64)]


def visualize(args: argparse.Namespace) -> dict:
    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    views = sorted(p for p in sequence_dir.iterdir() if p.is_dir() and p.name.startswith(args.view_prefix))
    landmark_indices = load_landmark_indices(args.verts_512, args.num_landmarks)

    summary = {
        "sequence_dir": str(sequence_dir),
        "frame": args.frame,
        "num_landmarks": args.num_landmarks,
        "verts_512": args.verts_512,
        "note": "MAMMA BEDLAM vertices3d are camera-space here; projection is cam_int @ vertices3d.",
        "views": [],
    }
    output_images = []

    for view in views[: args.num_views]:
        image_path = view / f"{args.frame}.jpg"
        mask_path = view / f"{args.frame}.mask.jpg"
        data_path = view / f"{args.frame}.data.pyd"
        if not image_path.exists() or not data_path.exists():
            continue

        image = Image.open(image_path).convert("RGB")
        overlay = image.convert("RGBA")
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L").resize(image.size)
            tint = Image.new("RGBA", image.size, (0, 190, 220, 55))
            overlay = Image.composite(tint, overlay, mask.point(lambda v: 180 if v > 0 else 0))
        else:
            mask = None

        draw = ImageDraw.Draw(overlay)
        with data_path.open("rb") as f:
            people = pickle.load(f)

        view_summary = {
            "view": view.name,
            "image": str(image_path),
            "mask": str(mask_path) if mask_path.exists() else None,
            "people": [],
        }

        for person_i, (person_id, person) in enumerate(sorted(people.items(), key=lambda item: int(item[0]))):
            color = COLORS[person_i % len(COLORS)]
            intrinsics = np.asarray(person["cam_int"], dtype=np.float64)
            vertices_2d = np.asarray(person["vertices2d"], dtype=np.float64)
            vertices_3d = np.asarray(person["vertices3d"], dtype=np.float64)
            visibility = np.asarray(person.get("vertex_visibility", np.ones((len(vertices_2d), 1)))).reshape(-1) > 0
            projected = project_points(intrinsics, vertices_3d)

            valid = visibility & np.isfinite(vertices_2d).all(axis=1) & np.isfinite(projected).all(axis=1)
            in_bounds = (
                valid
                & (vertices_2d[:, 0] >= 0)
                & (vertices_2d[:, 0] < image.width)
                & (vertices_2d[:, 1] >= 0)
                & (vertices_2d[:, 1] < image.height)
            )
            valid_indices = np.where(in_bounds)[0]

            for idx in evenly_sample(valid_indices, args.max_dense_points):
                draw_dot(draw, vertices_2d[idx, 0], vertices_2d[idx, 1], color, radius=1)

            if landmark_indices is not None:
                official_indices = landmark_indices[landmark_indices < len(vertices_2d)]
            else:
                official_indices = evenly_sample(valid_indices, args.num_landmarks)
            candidate_indices = official_indices[in_bounds[official_indices]]
            visible_candidate_indices = candidate_indices[visibility[candidate_indices]]

            for idx in visible_candidate_indices[: args.num_landmarks]:
                draw_ring(draw, projected[idx, 0], projected[idx, 1], (255, 0, 210, 235), radius=3)

            center = np.asarray(person["center"], dtype=np.float64)
            side = float(person["scale"]) * 200.0
            crop_box = crop_box_from_center_scale(center, float(person["scale"]), image.size, args.crop_scale)
            draw.rectangle(
                (center[0] - side / 2, center[1] - side / 2, center[0] + side / 2, center[1] + side / 2),
                outline=color,
                width=4,
            )

            crop = image.crop(crop_box).convert("RGBA")
            crop_draw = ImageDraw.Draw(crop)
            offset = np.array([crop_box[0], crop_box[1]], dtype=np.float64)
            hull_indices = evenly_sample(valid_indices, min(len(valid_indices), 1000))
            hull_points = vertices_2d[hull_indices] - offset
            hull_points = hull_points[
                (hull_points[:, 0] >= 0)
                & (hull_points[:, 0] < crop.size[0])
                & (hull_points[:, 1] >= 0)
                & (hull_points[:, 1] < crop.size[1])
            ]
            hull = convex_hull(hull_points)
            if len(hull) >= 3:
                person_layer = Image.new("RGBA", crop.size, (0, 0, 0, 0))
                person_draw = ImageDraw.Draw(person_layer)
                person_draw.polygon(hull, fill=(0, 190, 220, 65))
                crop = Image.alpha_composite(crop, person_layer)
                crop_draw = ImageDraw.Draw(crop)

            visible_crop_points = 0
            occluded_crop_points = 0
            for idx in official_indices[: args.num_landmarks]:
                point = projected[idx] - offset
                if point[0] < 0 or point[0] >= crop.size[0] or point[1] < 0 or point[1] >= crop.size[1]:
                    continue
                if visibility[idx] and in_bounds[idx]:
                    visible_crop_points += 1
                    draw_dot(crop_draw, point[0], point[1], color, radius=2)
                    draw_ring(crop_draw, point[0], point[1], (255, 0, 210, 245), radius=4)
                else:
                    occluded_crop_points += 1
                    draw_ring(crop_draw, point[0], point[1], (255, 145, 0, 220), radius=3)
            crop_draw.rectangle((0, 0, crop.size[0] - 1, crop.size[1] - 1), outline=color, width=4)
            crop_draw.rectangle((8, 8, min(crop.size[0] - 1, 520), 52), fill=(0, 0, 0, 150))
            crop_draw.text(
                (18, 18),
                f"{view.name}/{args.frame} person {person_id}: magenta visible, orange hidden",
                fill=(255, 255, 255, 255),
            )
            crop_output = output_dir / f"{view.name}_{args.frame}_person_{int(person_id):02d}_crop_projection.png"
            crop.convert("RGB").save(crop_output)

            error = np.linalg.norm(projected[valid_indices] - vertices_2d[valid_indices], axis=1)
            view_summary["people"].append(
                {
                    "person_id": int(person_id),
                    "vertices2d_shape": list(vertices_2d.shape),
                    "vertices3d_shape": list(vertices_3d.shape),
                    "visible_in_bounds": int(len(valid_indices)),
                    "official_landmarks_total": int(min(len(official_indices), args.num_landmarks)),
                    "official_visible_in_scene": int(min(len(visible_candidate_indices), args.num_landmarks)),
                    "official_visible_in_crop": int(visible_crop_points),
                    "official_hidden_in_crop": int(occluded_crop_points),
                    "crop_box_xyxy": list(crop_box),
                    "crop_output": str(crop_output),
                    "projection_error_px": {
                        "median": float(np.median(error)) if len(error) else None,
                        "mean": float(np.mean(error)) if len(error) else None,
                        "p95": float(np.percentile(error, 95)) if len(error) else None,
                    },
                }
            )

        draw.rectangle((8, 8, 800, 58), fill=(0, 0, 0, 150))
        draw.text(
            (18, 18),
            f"{view.name}/{args.frame}: color=vertices2d, magenta=projected sampled landmarks, blue=mask",
            fill=(255, 255, 255, 255),
        )
        out_image = output_dir / f"{view.name}_{args.frame}_scene_projection_checked.png"
        overlay.convert("RGB").save(out_image)
        view_summary["output"] = str(out_image)
        summary["views"].append(view_summary)
        output_images.append(out_image)

    if output_images:
        thumbs = []
        for path in output_images:
            image = Image.open(path).convert("RGB")
            scale = args.grid_width / image.width
            thumbs.append(image.resize((args.grid_width, int(image.height * scale))))
        grid = Image.new("RGB", (args.grid_width, sum(image.height for image in thumbs)), (0, 0, 0))
        y = 0
        for image in thumbs:
            grid.paste(image, (0, y))
            y += image.height
        grid.save(output_dir / "grid_scene_projection_checked.png")

    (output_dir / "summary_checked.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-dir",
        default="/mnt/train-data-4-hdd/yian/Mamma_dataset/tmp/bedlam_lab_20251031_191436/"
        "harmony4d_train_1_NC_200_00/png/be_HsuS3iLSSWWZ_seq_000000",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/train-data-4-hdd/yian/yian_vggt_smpl/debug/mamma_scene_projection",
    )
    parser.add_argument("--frame", default="0000")
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--view-prefix", default="IOI_")
    parser.add_argument("--verts-512", default=None)
    parser.add_argument("--num-landmarks", type=int, default=512)
    parser.add_argument("--max-dense-points", type=int, default=1800)
    parser.add_argument("--crop-scale", type=float, default=1.2)
    parser.add_argument("--grid-width", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    result = visualize(parse_args())
    print(json.dumps(result, indent=2))
