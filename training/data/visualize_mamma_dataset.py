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


def draw_overlay(image_rgb, mask, landmarks, weights, title):
    image = image_rgb.copy()
    if mask is not None:
        mask_bool = mask > 0.5
        overlay = image.copy()
        overlay[mask_bool] = (0, 220, 180)
        image = np.where(mask_bool[..., None], (0.58 * image + 0.42 * overlay).astype(np.uint8), image)

    valid = np.ones((landmarks.shape[0],), dtype=bool)
    if weights is not None:
        valid &= weights.reshape(-1) > 0.5
    valid &= np.isfinite(landmarks[:, :2]).all(axis=1)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for idx, xy in enumerate(landmarks[:, :2]):
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            color = (40, 255, 70) if valid[idx] else (255, 80, 80)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)

    draw.rectangle((0, 0, image.shape[1] - 1, image.shape[0] - 1), outline=(255, 255, 255), width=1)
    draw.text((8, 8), title, fill=(0, 0, 0), stroke_width=3, stroke_fill=(0, 0, 0))
    draw.text((8, 8), title, fill=(255, 255, 255))
    return np.asarray(pil_image)


def save_rgb(path, image_rgb):
    os.makedirs(osp.dirname(path), exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def make_grid(images, cols=2, pad=8):
    if not images:
        return None
    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    rows = int(np.ceil(len(images) / cols))
    grid = np.zeros((rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        y = r * (h + pad)
        x = c * (w + pad)
        grid[y:y + img.shape[0], x:x + img.shape[1]] = img
    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize cropped MAMMA dataset samples.")
    parser.add_argument("--dataset-dir", default="/mnt/train-data-4-hdd/yian/Mamma_dataset")
    parser.add_argument("--out-dir", default="/mnt/train-data-4-hdd/yian/yian_vggt_smpl/debug/mamma_dataset_vis")
    parser.add_argument("--seq-index", type=int, default=0)
    parser.add_argument("--num-views", type=int, default=4)
    parser.add_argument("--person-id", type=int, default=0)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--aspect-ratio", type=float, default=1.0)
    parser.add_argument("--crop-scale", type=float, default=1.2)
    parser.add_argument("--num-landmarks", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = MammaDataset(
        common_conf=make_common_conf(args),
        split="test",
        MAMMA_DIR=args.dataset_dir,
        min_num_views=args.num_views,
        len_test=max(args.seq_index + 1, 1),
        person_id=args.person_id,
        frame_id=args.frame_id,
        crop_scale=args.crop_scale,
        mask_from_landmarks=True,
        num_landmarks=args.num_landmarks,
    )
    batch = dataset.get_data(seq_index=args.seq_index, img_per_seq=args.num_views, aspect_ratio=args.aspect_ratio)

    os.makedirs(args.out_dir, exist_ok=True)
    overlay_images = []
    view_summaries = []
    for view_idx, image in enumerate(batch["images"]):
        camera_name = batch["camera_names"][view_idx]
        mask = batch["person_masks"][view_idx]
        landmarks = batch["landmarks_512"][view_idx]
        weights = batch["landmarks_512_weights"][view_idx]
        valid_count = int(np.asarray(weights).reshape(-1).sum())
        title = f"{camera_name} valid={valid_count}/{len(landmarks)}"

        overlay = draw_overlay(image, mask, landmarks, weights, title)
        overlay_images.append(overlay)
        save_rgb(osp.join(args.out_dir, f"{view_idx:02d}_{camera_name}_crop_overlay.png"), overlay)
        save_rgb(osp.join(args.out_dir, f"{view_idx:02d}_{camera_name}_crop_rgb.png"), image)

        mask_u8 = (np.asarray(mask) * 255).astype(np.uint8)
        Image.fromarray(mask_u8).save(osp.join(args.out_dir, f"{view_idx:02d}_{camera_name}_mask.png"))

        view_summaries.append(
            {
                "view_idx": view_idx,
                "camera_name": camera_name,
                "image_path": batch["image_paths"][view_idx],
                "crop_image_shape": list(image.shape),
                "original_size_hw": np.asarray(batch["original_sizes"][view_idx]).astype(int).tolist(),
                "valid_landmarks": valid_count,
                "crop_bbox_center_xy_size_wh": np.asarray(batch["crop_bboxes"][view_idx]).astype(float).round(3).tolist(),
                "crop_intrinsics": np.asarray(batch["intrinsics"][view_idx]).astype(float).round(4).tolist(),
                "full_intrinsics": np.asarray(batch["full_intrinsics"][view_idx]).astype(float).round(4).tolist(),
                "extrinsics_shape": list(np.asarray(batch["extrinsics"][view_idx]).shape),
            }
        )

    grid = make_grid(overlay_images, cols=2)
    if grid is not None:
        save_rgb(osp.join(args.out_dir, "grid_crop_overlay.png"), grid)

    summary = {
        "seq_name": batch["seq_name"],
        "ids": np.asarray(batch["ids"]).astype(int).tolist(),
        "num_views": len(batch["images"]),
        "smplx_pose_shape": list(np.asarray(batch.get("smplx_pose", [])).shape),
        "smplx_betas_shape": list(np.asarray(batch.get("smplx_betas", [])).shape),
        "smplx_trans_shape": list(np.asarray(batch.get("smplx_trans", [])).shape),
        "smplx_gender": batch.get("smplx_gender", None),
        "views": view_summaries,
    }
    with open(osp.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved visualizations to: {args.out_dir}")


if __name__ == "__main__":
    main()
