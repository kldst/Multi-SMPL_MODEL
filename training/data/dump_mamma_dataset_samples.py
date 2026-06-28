#!/usr/bin/env python3
"""Dump MammaDataset samples into per-sample debug folders."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import random
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw


TRAINING_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
REPO_ROOT = osp.abspath(osp.join(TRAINING_ROOT, ".."))
for path in (TRAINING_ROOT, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from data.datasets.mamma_dataset import MammaDataset  # noqa: E402


def make_common_conf(args):
    return SimpleNamespace(
        img_size=args.img_size,
        patch_size=args.patch_size,
        rescale=False,
        rescale_aug=False,
        landscape_check=False,
        debug=False,
        training=False,
        inside_random=False,
        allow_duplicate_img=False,
        augs=SimpleNamespace(scales=None),
    )


def save_rgb(path: str, image: np.ndarray) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path)


def draw_overlay(image_rgb: np.ndarray, mask: np.ndarray, landmarks: np.ndarray, weights: np.ndarray, title: str) -> np.ndarray:
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask_bool = np.asarray(mask) > 0.5
    if mask_bool.any():
        tint = image.copy()
        tint[mask_bool] = (0, 190, 220)
        image = np.where(mask_bool[..., None], (0.58 * image + 0.42 * tint).astype(np.uint8), image)

    weights = np.asarray(weights).reshape(-1)
    landmarks = np.asarray(landmarks, dtype=np.float32)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for idx, xy in enumerate(landmarks[:, :2]):
        x, y = float(xy[0]), float(xy[1])
        if not np.isfinite([x, y]).all():
            continue
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue
        visible = idx < len(weights) and weights[idx] > 0.5
        color = (255, 0, 210) if visible else (255, 145, 0)
        radius = 4 if visible else 3
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)

    draw.rectangle((0, 0, image.shape[1] - 1, image.shape[0] - 1), outline=(255, 255, 255), width=1)
    draw.rectangle((6, 6, min(image.shape[1] - 1, 740), 48), fill=(0, 0, 0))
    draw.text((14, 16), title, fill=(255, 255, 255))
    return np.asarray(pil)


def bbox_xyxy_from_center_size(center_size: np.ndarray, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    cx, cy, width, height = [float(v) for v in center_size]
    x1 = int(np.floor(cx - width * 0.5))
    y1 = int(np.floor(cy - height * 0.5))
    x2 = int(np.ceil(cx + width * 0.5))
    y2 = int(np.ceil(cy + height * 0.5))
    x1 = max(0, min(image_size[0] - 1, x1))
    y1 = max(0, min(image_size[1] - 1, y1))
    x2 = max(x1 + 1, min(image_size[0], x2))
    y2 = max(y1 + 1, min(image_size[1], y2))
    return x1, y1, x2, y2


def save_original_bbox_debug(
    image_path: str,
    center_size: np.ndarray,
    out_scene_path: str,
    out_raw_crop_path: str,
    title: str,
) -> tuple[int, int, int, int]:
    image = Image.open(image_path).convert("RGB")
    bbox = bbox_xyxy_from_center_size(center_size, image.size)
    scene = image.copy()
    draw = ImageDraw.Draw(scene)
    draw.rectangle(bbox, outline=(255, 0, 210), width=6)
    draw.rectangle((8, 8, min(scene.size[0] - 1, 720), 54), fill=(0, 0, 0))
    draw.text((18, 20), title, fill=(255, 255, 255))
    os.makedirs(osp.dirname(out_scene_path), exist_ok=True)
    scene.save(out_scene_path)
    image.crop(bbox).save(out_raw_crop_path)
    return bbox


def make_grid(images: list[np.ndarray], cols: int = 2, pad: int = 8) -> np.ndarray | None:
    if not images:
        return None
    cols = min(cols, len(images))
    h = max(image.shape[0] for image in images)
    w = max(image.shape[1] for image in images)
    rows = int(np.ceil(len(images) / cols))
    grid = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        y = row * (h + pad)
        x = col * (w + pad)
        grid[y : y + image.shape[0], x : x + image.shape[1]] = image
    return grid


def dump_sample(dataset: MammaDataset, sample_idx: int, args) -> dict:
    batch = dataset.get_data(seq_index=sample_idx, img_per_seq=args.num_views, aspect_ratio=args.aspect_ratio)
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in batch["seq_name"])
    sample_dir = osp.join(args.out_dir, f"sample_{sample_idx:05d}_{safe_name}")
    os.makedirs(sample_dir, exist_ok=True)

    overlays = []
    views = []
    for view_idx, image in enumerate(batch["images"]):
        camera_name = batch["camera_names"][view_idx]
        mask = np.asarray(batch["person_masks"][view_idx], dtype=np.float32)
        landmarks = np.asarray(batch["landmarks_512"][view_idx], dtype=np.float32)
        weights = np.asarray(batch["landmarks_512_weights"][view_idx], dtype=np.float32)
        valid_count = int((weights.reshape(-1) > 0.5).sum())
        title = f"{camera_name} visible={valid_count}/{len(landmarks)}"

        overlay = draw_overlay(image, mask, landmarks, weights, title)
        overlays.append(overlay)

        prefix = f"{view_idx:02d}_{camera_name}"
        save_rgb(osp.join(sample_dir, f"{prefix}_crop_rgb.png"), image)
        save_rgb(osp.join(sample_dir, f"{prefix}_crop_overlay.png"), overlay)
        Image.fromarray((mask * 255).astype(np.uint8)).save(osp.join(sample_dir, f"{prefix}_person_mask.png"))
        bbox_xyxy = save_original_bbox_debug(
            batch["image_paths"][view_idx],
            np.asarray(batch["crop_bboxes"][view_idx]),
            osp.join(sample_dir, f"{prefix}_scene_bbox.png"),
            osp.join(sample_dir, f"{prefix}_raw_bbox_crop.png"),
            f"{camera_name} original bbox -> resized to {image.shape[1]}x{image.shape[0]}",
        )

        views.append(
            {
                "view_idx": view_idx,
                "camera_name": camera_name,
                "image_path": batch["image_paths"][view_idx],
                "crop_image_shape": list(np.asarray(image).shape),
                "original_size_hw": np.asarray(batch["original_sizes"][view_idx]).astype(int).tolist(),
                "visible_landmarks": valid_count,
                "crop_bbox_center_xy_size_wh": np.asarray(batch["crop_bboxes"][view_idx]).astype(float).round(3).tolist(),
                "crop_bbox_xyxy_in_original": list(bbox_xyxy),
                "raw_bbox_crop_shape_hw": [bbox_xyxy[3] - bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0]],
                "mamma_crop_size_hw": np.asarray(batch.get("mamma_crop_sizes", [[-1, -1]])[view_idx]).astype(int).tolist()
                if "mamma_crop_sizes" in batch
                else None,
                "pad_offset_xy": np.asarray(batch.get("pad_offsets", [[0, 0]])[view_idx]).astype(float).round(3).tolist()
                if "pad_offsets" in batch
                else None,
                "crop_transform_2x3": np.asarray(batch["crop_transforms"][view_idx]).astype(float).round(6).tolist(),
                "crop_intrinsics": np.asarray(batch["intrinsics"][view_idx]).astype(float).round(4).tolist(),
                "full_intrinsics": np.asarray(batch["full_intrinsics"][view_idx]).astype(float).round(4).tolist(),
                "extrinsics_shape": list(np.asarray(batch["extrinsics"][view_idx]).shape),
            }
        )

    grid = make_grid(overlays, cols=2)
    if grid is not None:
        save_rgb(osp.join(sample_dir, "grid_crop_overlay.png"), grid)

    summary = {
        "sample_idx": sample_idx,
        "sample_dir": sample_dir,
        "seq_name": batch["seq_name"],
        "ids": np.asarray(batch["ids"]).astype(int).tolist(),
        "num_views": len(batch["images"]),
        "person_id": int(np.asarray(batch.get("person_id", -1))),
        "frame_id": int(np.asarray(batch.get("smplx_frame_id", -1))),
        "smplx_pose_shape": list(np.asarray(batch.get("smplx_pose", [])).shape),
        "smplx_betas_shape": list(np.asarray(batch.get("smplx_betas", [])).shape),
        "smplx_trans_shape": list(np.asarray(batch.get("smplx_trans", [])).shape),
        "smplx_gender": batch.get("smplx_gender", None),
        "views": views,
    }
    with open(osp.join(sample_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Dump per-sample MAMMA dataset debug folders.")
    parser.add_argument(
        "--dataset-dir",
        default="/mnt/train-data-4-hdd/yian/Mamma_dataset/tmp/bedlam_lab_20251031_191436",
    )
    parser.add_argument(
        "--verts-512",
        default="/mnt/train-data-4-hdd/yian/yian_vggt_smpl/data/body_models/downsampled_verts/verts_512.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/train-data-4-hdd/yian/yian_vggt_smpl/debug/mamma_dataset_samples",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--min-num-views", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--aspect-ratio", type=float, default=1.0)
    parser.add_argument("--crop-scale", type=float, default=1.0)
    parser.add_argument("--mamma-crop-width", type=int, default=384)
    parser.add_argument("--mamma-crop-height", type=int, default=512)
    parser.add_argument("--num-landmarks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = MammaDataset(
        common_conf=make_common_conf(args),
        split="test",
        MAMMA_DIR=args.dataset_dir,
        downsampled_verts_path=args.verts_512,
        min_num_views=args.min_num_views,
        len_test=max(args.start_index + args.num_samples, 1),
        crop_scale=args.crop_scale,
        mamma_crop_width=args.mamma_crop_width,
        mamma_crop_height=args.mamma_crop_height,
        pad_to_square=True,
        mask_from_landmarks=True,
        num_landmarks=args.num_landmarks,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    stop = min(args.start_index + args.num_samples, dataset.sequence_list_len)
    for sample_idx in range(args.start_index, stop):
        summaries.append(dump_sample(dataset, sample_idx, args))

    index = {
        "dataset_dir": args.dataset_dir,
        "verts_512": args.verts_512,
        "num_available_samples": dataset.sequence_list_len,
        "num_dumped_samples": len(summaries),
        "samples": summaries,
    }
    with open(osp.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(json.dumps(index, indent=2))
    print(f"\nSaved debug samples to: {args.out_dir}")


if __name__ == "__main__":
    main()
