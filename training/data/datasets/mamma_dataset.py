import glob
import json
import logging
import os.path as osp
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def read_image_pil(path: str) -> Optional[np.ndarray]:
    if not osp.isfile(path):
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase a bbox size (w, h) to match a target aspect (w, h)."""
    if target_aspect_ratio is None:
        return np.asarray(input_shape, dtype=np.float32)
    try:
        w, h = input_shape
        w_t, h_t = target_aspect_ratio
    except (TypeError, ValueError):
        return np.asarray(input_shape, dtype=np.float32)

    if h / max(w, 1e-6) < h_t / max(w_t, 1e-6):
        return np.array([w, max(w * h_t / w_t, h)], dtype=np.float32)
    return np.array([max(h * w_t / h_t, w), h], dtype=np.float32)


def rotate_2d(pt_2d: np.ndarray, rot_rad: float) -> np.ndarray:
    x, y = pt_2d
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([x * cs - y * sn, x * sn + y * cs], dtype=np.float32)


def get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_mamma_affine_transform(
    center: np.ndarray,
    scale: np.ndarray,
    pixel_std: float,
    output_size: Tuple[int, int],
    rot: float = 0.0,
) -> np.ndarray:
    """MAMMA/HRNet-style affine transform from original pixels to model crop pixels."""
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale], dtype=np.float32)
    scale_tmp = scale.astype(np.float32) * float(pixel_std)
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size
    rot_rad = np.pi * rot / 180.0
    src_dir = rotate_2d(np.array([0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0] = center.astype(np.float32)
    src[1] = center.astype(np.float32) + src_dir
    src[2] = get_3rd_point(src[0], src[1])
    dst[0] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst[1] = dst[0] + dst_dir
    dst[2] = get_3rd_point(dst[0], dst[1])

    src_h = np.concatenate([src, np.ones((3, 1), dtype=np.float32)], axis=1)
    return np.linalg.solve(src_h, dst).T.astype(np.float32)


def gen_trans_from_patch_cv(
    c_x: float,
    c_y: float,
    src_width: float,
    src_height: float,
    dst_width: float,
    dst_height: float,
    scale: float,
    rot: float,
) -> np.ndarray:
    """Affine transform from original image pixels to cropped patch pixels."""
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180.0

    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_center = np.array([dst_width * 0.5, dst_height * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_height * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_width * 0.5, 0], dtype=np.float32)

    src = np.stack([src_center, src_center + src_downdir, src_center + src_rightdir]).astype(np.float32)
    dst = np.stack([dst_center, dst_center + dst_downdir, dst_center + dst_rightdir]).astype(np.float32)
    src_h = np.concatenate([src, np.ones((3, 1), dtype=np.float32)], axis=1)
    return np.linalg.solve(src_h, dst).T.astype(np.float32)


def transform_points2d(points: np.ndarray, trans: np.ndarray) -> np.ndarray:
    if points is None:
        return points
    pts = np.asarray(points, dtype=np.float32)
    flat = pts.reshape(-1, pts.shape[-1])
    xy1 = np.concatenate([flat[:, :2], np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    out_xy = xy1 @ trans.T
    out = flat.copy()
    out[:, :2] = out_xy
    return out.reshape(pts.shape)


def affine_to_homogeneous(trans: np.ndarray) -> np.ndarray:
    out = np.eye(3, dtype=np.float32)
    out[:2, :] = trans.astype(np.float32)
    return out


def generate_image_patch_affine(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    c_x: float,
    c_y: float,
    bb_width: float,
    bb_height: float,
    patch_width: int,
    patch_height: int,
    scale: float,
    rot: float,
    border_value=0,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)
    trans_h = affine_to_homogeneous(trans)
    inv_trans = np.linalg.inv(trans_h)[:2].reshape(-1)
    img_patch = Image.fromarray(img).transform(
        (int(patch_width), int(patch_height)),
        Image.Transform.AFFINE,
        data=tuple(float(v) for v in inv_trans),
        resample=Image.Resampling.BILINEAR,
        fillcolor=border_value,
    )
    img_patch = np.asarray(img_patch)
    mask_patch = None
    if mask is not None:
        mask_patch = Image.fromarray(mask.astype(np.uint8)).transform(
            (int(patch_width), int(patch_height)),
            Image.Transform.AFFINE,
            data=tuple(float(v) for v in inv_trans),
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        mask_patch = np.asarray(mask_patch)
    return img_patch, mask_patch, trans


def generate_image_patch_with_trans(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    trans: np.ndarray,
    patch_width: int,
    patch_height: int,
    border_value=0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    trans_h = affine_to_homogeneous(trans)
    inv_trans = np.linalg.inv(trans_h)[:2].reshape(-1)
    img_patch = Image.fromarray(img).transform(
        (int(patch_width), int(patch_height)),
        Image.Transform.AFFINE,
        data=tuple(float(v) for v in inv_trans),
        resample=Image.Resampling.BILINEAR,
        fillcolor=border_value,
    )
    img_patch = np.asarray(img_patch)
    mask_patch = None
    if mask is not None:
        mask_patch = Image.fromarray(mask.astype(np.uint8)).transform(
            (int(patch_width), int(patch_height)),
            Image.Transform.AFFINE,
            data=tuple(float(v) for v in inv_trans),
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        mask_patch = np.asarray(mask_patch)
    return img_patch, mask_patch


def pad_to_canvas(
    image: np.ndarray,
    mask: np.ndarray,
    landmarks: np.ndarray,
    weights: np.ndarray,
    canvas_h: int,
    canvas_w: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    pad_x = int((canvas_w - w) // 2)
    pad_y = int((canvas_h - h) // 2)
    out_image = np.zeros((canvas_h, canvas_w, image.shape[2]), dtype=image.dtype)
    out_mask = np.zeros((canvas_h, canvas_w), dtype=mask.dtype)
    out_image[pad_y:pad_y + h, pad_x:pad_x + w] = image
    out_mask[pad_y:pad_y + h, pad_x:pad_x + w] = mask
    out_landmarks = landmarks.copy()
    out_landmarks[:, 0] += pad_x
    out_landmarks[:, 1] += pad_y
    out_weights = weights.copy()
    out_weights[:, 0] *= (
        np.isfinite(out_landmarks).all(axis=1)
        & (out_landmarks[:, 0] >= 0)
        & (out_landmarks[:, 0] < canvas_w)
        & (out_landmarks[:, 1] >= 0)
        & (out_landmarks[:, 1] < canvas_h)
    ).astype(np.float32)
    return out_image, out_mask, out_landmarks, out_weights, np.array([pad_x, pad_y], dtype=np.float32)


def first_array_like(value: Any, preferred_lengths: Tuple[int, ...] = (512,)) -> Optional[np.ndarray]:
    """Find a likely landmark/point array inside a nested object loaded from npz."""
    candidates: List[np.ndarray] = []

    def visit(obj: Any):
        if isinstance(obj, np.ndarray):
            if obj.dtype == object:
                if obj.shape == ():
                    visit(obj.item())
                else:
                    for item in obj.flat:
                        visit(item)
                return
            if obj.ndim >= 2 and obj.shape[-1] >= 2:
                candidates.append(obj)
            return
        if isinstance(obj, dict):
            preferred_keys = (
                "landmarks",
                "landmarks_2d",
                "joints2d",
                "joints_2d",
                "joints_2d_image",
                "keypoints",
                "keypoints_2d",
                "keypoints_2d_image",
                "vertices_2d",
                "vertices_2d_image",
                "points2d",
                "pts2d",
            )
            for key in preferred_keys:
                if key in obj:
                    visit(obj[key])
            for key, item in obj.items():
                if key not in preferred_keys:
                    visit(item)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)

    visit(value)
    if not candidates:
        return None

    def score(arr: np.ndarray) -> Tuple[int, int]:
        length = arr.shape[-2]
        preferred = 0 if length in preferred_lengths else 1
        return preferred, -length

    best = sorted(candidates, key=score)[0]
    if best.ndim > 2:
        best = best.reshape(-1, best.shape[-2], best.shape[-1])[0]
    return np.asarray(best[..., :2], dtype=np.float32)


def resolve_npz_key(data: Any, key: str) -> Optional[str]:
    candidates = [key]
    if key.endswith(".npy"):
        candidates.append(key[:-4])
    else:
        candidates.append(key + ".npy")
    for candidate in candidates:
        if candidate in data.files:
            return candidate
    return None


class MammaDataset:
    """Multi-view cropped-person dataset for MAMMA-style preprocessed samples.

    Expected minimal layout:
      root/
        runs_00000.npz
        runs_00000/
          IOI_01.jpg
          IOI_02.jpg
          ...

    The npz should contain camera keys like
    `cam_param_min/<cam>/intrinsics.K_flat9.npy`,
    `cam_param_min/<cam>/extrinsics.worldToCamera12.npy`, optional
    `reprojection_data/<cam>/people.npy`, and SMPL params under
    `out_param/person_XX/frame_YYYYY/smpl_params/`.
    """

    def __init__(
        self,
        common_conf,
        split: str = "train",
        MAMMA_DIR: str = "/mnt/train-data-4-hdd/yian/Mamma_dataset",
        downsampled_verts_path: str = "/mnt/train-data-4-hdd/yian/yian_vggt_smpl/data/body_models/downsampled_verts/verts_512.pkl",
        min_num_views: int = 4,
        len_train: int = 100000,
        len_test: int = 10000,
        person_id: int = 0,
        frame_id: int = 0,
        crop_scale: float = 1.2,
        mamma_crop_width: int = 384,
        mamma_crop_height: int = 512,
        pad_to_square: bool = True,
        mask_from_landmarks: bool = True,
        num_landmarks: int = 512,
        hand_weight: float = 1.0,
        full_scene: bool = True,
    ):
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size

        self.MAMMA_DIR = MAMMA_DIR
        self.downsampled_verts_path = downsampled_verts_path
        self.min_num_views = min_num_views
        self.person_id = person_id
        self.frame_id = frame_id
        self.crop_scale = crop_scale
        self.mamma_crop_width = mamma_crop_width
        self.mamma_crop_height = mamma_crop_height
        self.pad_to_square = pad_to_square
        self.mask_from_landmarks = mask_from_landmarks
        self.num_landmarks = num_landmarks
        self.hand_weight = hand_weight
        self.full_scene = full_scene
        self.landmark_indices = self._load_downsampled_vertex_indices(downsampled_verts_path, num_landmarks)

        parts_path = osp.join(osp.dirname(__file__), "_smplx_512_body_parts.json")
        if osp.isfile(parts_path):
            with open(parts_path) as _f:
                _raw = json.load(_f)
            self.body_parts_dict = {k: np.asarray(v, dtype=np.int64) for k, v in _raw.items()}
        else:
            logging.warning("MAMMA body parts JSON not found: %s", parts_path)
            self.body_parts_dict = None

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.bedlam_samples = self._build_bedlam_samples(self.MAMMA_DIR)
        if self.debug:
            self.bedlam_samples = self.bedlam_samples[:8]
        self.use_bedlam_pyd = len(self.bedlam_samples) > 0

        self.sequence_npz_list = []
        self.sequence_list = []
        if not self.use_bedlam_pyd:
            self.sequence_npz_list = sorted(glob.glob(osp.join(self.MAMMA_DIR, "*.npz")))
            if self.debug:
                self.sequence_npz_list = self.sequence_npz_list[:1]
            if not self.sequence_npz_list:
                raise FileNotFoundError(f"No .data.pyd or MAMMA npz files found under {self.MAMMA_DIR}")
            self.sequence_list = [osp.splitext(osp.basename(path))[0] for path in self.sequence_npz_list]

        self.sequence_list_len = len(self.bedlam_samples) if self.use_bedlam_pyd else len(self.sequence_list)
        logging.info(
            "%s: MAMMA %s data size: %d",
            "Training" if self.training else "Testing",
            "BEDLAM .data.pyd" if self.use_bedlam_pyd else "npz",
            self.sequence_list_len,
        )

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

    def get_target_shape(self, aspect_ratio):
        short_size = int(self.img_size * aspect_ratio)
        if short_size % self.patch_size != 0:
            short_size = (short_size // self.patch_size) * self.patch_size
        return np.array([short_size, self.img_size])

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.use_bedlam_pyd:
            return self._get_bedlam_data(
                seq_index=seq_index,
                img_per_seq=img_per_seq,
                ids=ids,
                aspect_ratio=aspect_ratio,
            )

        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        npz_path = osp.join(self.MAMMA_DIR, f"{seq_name}.npz")
        sample_dir = osp.join(self.MAMMA_DIR, seq_name)
        data = np.load(npz_path, allow_pickle=True)

        camera_names = self._camera_names(data, sample_dir)
        if len(camera_names) < self.min_num_views:
            raise ValueError(f"{seq_name} has only {len(camera_names)} views, expected at least {self.min_num_views}")

        if img_per_seq is None:
            img_per_seq = self.min_num_views
        img_per_seq = min(int(img_per_seq), len(camera_names))

        if ids is None:
            ids = np.random.choice(len(camera_names), img_per_seq, replace=self.allow_duplicate_img)
        ids = np.asarray(ids, dtype=np.int64)
        selected_cameras = [camera_names[int(i)] for i in ids]

        target_image_shape = self.get_target_shape(aspect_ratio)
        target_h, target_w = int(target_image_shape[0]), int(target_image_shape[1])

        smpl_params = self._load_smpl_params(data)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        person_masks = []
        extrinsics = []
        intrinsics = []
        full_intrinsics = []
        full_extrinsics = []
        crop_transforms = []
        crop_bboxes = []
        pad_offsets = []
        mamma_crop_sizes = []
        landmarks_512 = []
        landmarks_512_weights = []
        image_paths = []
        original_sizes = []

        for cam_name in selected_cameras:
            image_path = osp.join(sample_dir, f"{cam_name}.jpg")
            image = read_image_pil(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            original_h, original_w = image.shape[:2]

            K = self._load_intrinsic(data, cam_name)
            extri = self._load_extrinsic(data, cam_name)
            ldmks = self._load_landmarks(data, cam_name)
            bbox = self._bbox_from_landmarks(ldmks, original_w, original_h)
            bbox_size = expand_to_aspect_ratio(
                np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]], dtype=np.float32),
                target_aspect_ratio=np.array([target_w, target_h], dtype=np.float32),
            )
            bbox_size = np.maximum(bbox_size * self.crop_scale, 4.0)
            center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)

            mask = self._load_mask(sample_dir, cam_name)
            if mask is None and self.mask_from_landmarks and ldmks is not None:
                mask = self._mask_from_landmarks(ldmks, original_h, original_w)

            crop_img, crop_mask, trans = generate_image_patch_affine(
                image,
                mask,
                float(center[0]),
                float(center[1]),
                float(bbox_size[0]),
                float(bbox_size[1]),
                target_w,
                target_h,
                scale=1.0,
                rot=0.0,
            )
            crop_K = affine_to_homogeneous(trans) @ K
            crop_ldmks_raw = transform_points2d(ldmks, trans) if ldmks is not None else None
            crop_ldmks, crop_weights = self._format_landmarks(crop_ldmks_raw, target_w, target_h)

            if crop_mask is None:
                crop_mask = np.zeros((target_h, target_w), dtype=np.float32)
            crop_mask = (crop_mask > 0).astype(np.float32)

            images.append(crop_img)
            depths.append(np.zeros((target_h, target_w), dtype=np.float32))
            cam_points.append(np.zeros((target_h, target_w, 3), dtype=np.float32))
            world_points.append(np.zeros((target_h, target_w, 3), dtype=np.float32))
            point_masks.append(np.zeros((target_h, target_w), dtype=bool))
            person_masks.append(crop_mask)
            extrinsics.append(extri)
            intrinsics.append(crop_K.astype(np.float32))
            full_intrinsics.append(K.astype(np.float32))
            full_extrinsics.append(extri.astype(np.float32))
            crop_transforms.append(trans.astype(np.float32))
            crop_bboxes.append(np.array([center[0], center[1], bbox_size[0], bbox_size[1]], dtype=np.float32))
            landmarks_512.append(crop_ldmks.astype(np.float32))
            landmarks_512_weights.append(crop_weights.astype(np.float32))
            image_paths.append(image_path)
            original_sizes.append(np.array([original_h, original_w], dtype=np.int64))

        batch = {
            "seq_name": "mamma_" + seq_name,
            "ids": ids,
            "camera_names": selected_cameras,
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "person_masks": person_masks,
            "full_intrinsics": full_intrinsics,
            "full_extrinsics": full_extrinsics,
            "crop_transforms": crop_transforms,
            "crop_bboxes": crop_bboxes,
            "landmarks_512": landmarks_512,
            "landmarks_512_weights": landmarks_512_weights,
            "image_paths": image_paths,
            "original_sizes": original_sizes,
        }
        batch.update(smpl_params)
        return batch

    def _load_downsampled_vertex_indices(self, path: str, num_landmarks: int) -> Optional[np.ndarray]:
        if not path or not osp.isfile(path):
            logging.warning("MAMMA verts_512 file not found: %s", path)
            return None
        try:
            import joblib

            matrix = joblib.load(path)
        except Exception:
            matrix = load_pickle(path)
        if hasattr(matrix, "numpy"):
            matrix = matrix.numpy()
        matrix = np.asarray(matrix)
        if matrix.ndim == 2:
            # Bilateral split: take num_landmarks//2 from each body-side half,
            # matching the official BEDLAM_WD structure (left 0..half-1, right half..end).
            half_desired = num_landmarks // 2
            half_total = matrix.shape[0] // 2
            selected = np.concatenate([
                matrix[:half_desired],
                matrix[half_total : half_total + half_desired],
            ], axis=0)
            indices = selected.argmax(axis=-1)
        else:
            indices = matrix.reshape(-1).astype(np.int64)[:num_landmarks]
        return indices.astype(np.int64)

    def _build_bedlam_samples(self, root: str) -> List[Dict[str, Any]]:
        data_paths = sorted(glob.glob(osp.join(root, "**", "*.data.pyd"), recursive=True))
        grouped: Dict[Tuple[str, str], Dict[str, str]] = {}
        for path in data_paths:
            view_dir = osp.dirname(path)
            seq_dir = osp.dirname(view_dir)
            view_name = osp.basename(view_dir)
            frame = osp.splitext(osp.basename(path))[0].split(".")[0]
            grouped.setdefault((seq_dir, frame), {})[view_name] = path

        samples: List[Dict[str, Any]] = []
        for (seq_dir, frame), view_to_data in sorted(grouped.items()):
            views = sorted(view_to_data)
            if len(views) < self.min_num_views:
                continue
            people = load_pickle(view_to_data[views[0]])
            if not isinstance(people, dict):
                continue
            for person_id in sorted(people.keys(), key=lambda item: int(item)):
                samples.append(
                    {
                        "seq_dir": seq_dir,
                        "seq_name": osp.basename(seq_dir),
                        "frame": frame,
                        "person_id": int(person_id),
                        "views": views,
                    }
                )
        return samples

    def _get_bedlam_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = 0
        sample = self.bedlam_samples[int(seq_index) % len(self.bedlam_samples)]
        view_names = list(sample["views"])
        if img_per_seq is None:
            img_per_seq = self.min_num_views
        img_per_seq = min(int(img_per_seq), len(view_names))
        # Not every (frame, view) contains the target person (a person can leave a
        # camera's FOV). Keep only views whose .data.pyd actually has the person, and
        # cache the loaded person dict so the main loop doesn't reload it. A reduced
        # view count is fine — the fusion is view-count agnostic.
        pid = sample["person_id"]
        if ids is not None:
            order = [int(i) for i in np.asarray(ids).reshape(-1)]
        else:
            order = list(range(len(view_names)))
            if self.training:
                random.shuffle(order)
        selected_views, person_cache = [], {}
        for i in order:
            vn = view_names[int(i)]
            dp = osp.join(sample["seq_dir"], vn, f"{sample['frame']}.data.pyd")
            try:
                ppl = load_pickle(dp)
            except Exception:
                continue
            pr = ppl.get(pid, ppl.get(str(pid)))
            if pr is None:
                continue
            selected_views.append(vn)
            person_cache[vn] = pr
            if len(selected_views) >= img_per_seq:
                break
        if not selected_views:
            raise KeyError(f"Person {pid} not present in any view for frame {sample['frame']}")
        ids = np.asarray([view_names.index(v) for v in selected_views], dtype=np.int64)

        final_image_shape = self.get_target_shape(aspect_ratio)
        final_h, final_w = int(final_image_shape[0]), int(final_image_shape[1])
        crop_w = int(self.mamma_crop_width)
        crop_h = int(self.mamma_crop_height)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        person_masks = []
        extrinsics = []
        intrinsics = []
        full_intrinsics = []
        full_extrinsics = []
        crop_transforms = []
        crop_bboxes = []
        pad_offsets = []
        mamma_crop_sizes = []
        landmarks_512 = []
        landmarks_512_weights = []
        image_paths = []
        original_sizes = []
        first_person = None

        for view_name in selected_views:
            view_dir = osp.join(sample["seq_dir"], view_name)
            image_path = osp.join(view_dir, f"{sample['frame']}.jpg")
            data_path = osp.join(view_dir, f"{sample['frame']}.data.pyd")
            image = read_image_pil(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            person = person_cache[view_name]
            if first_person is None:
                first_person = person

            original_h, original_w = image.shape[:2]
            K = np.asarray(person["cam_int"], dtype=np.float32).reshape(3, 3)
            extri = np.asarray(person["cam_ext"], dtype=np.float32)
            vertices_2d = np.asarray(person["vertices2d"], dtype=np.float32)
            visibility = np.asarray(
                person.get("vertex_visibility", np.ones((len(vertices_2d), 1), dtype=np.float32))
            ).reshape(-1) > 0

            if self.landmark_indices is None:
                ldmk_indices = np.arange(min(self.num_landmarks, len(vertices_2d)), dtype=np.int64)
            else:
                ldmk_indices = self.landmark_indices[self.landmark_indices < len(vertices_2d)]
            ldmks = vertices_2d[ldmk_indices]
            landmark_visibility = visibility[ldmk_indices].astype(np.float32)

            if self.full_scene:
                # ---- Full-scene letterbox path (target-person mask, no per-person crop) ----
                # Resize the WHOLE scene image into the (final_h, final_w) square,
                # preserving aspect ratio (pad), so VGGT sees the multi-person scene
                # and the per-view mask selects the target person.  K follows the same
                # affine; extrinsics are unchanged.
                ltrans, new_h, new_w, pad_x, pad_y = self._letterbox_transform(
                    original_h, original_w, final_h, final_w
                )
                crop_img = self._warp_image(image, new_h, new_w, pad_x, pad_y, final_h, final_w)

                scene_mask = self._load_scene_person_mask(
                    view_dir, sample["frame"], sample["person_id"], original_h, original_w
                )
                if scene_mask is None:
                    # fallback: convex hull of visible projected vertices in full image
                    scene_mask = self._mask_from_landmarks(
                        vertices_2d[visibility], original_h, original_w
                    ) > 0
                crop_mask = self._warp_binary_mask(
                    scene_mask, new_h, new_w, pad_x, pad_y, final_h, final_w
                )

                crop_K = (affine_to_homogeneous(ltrans) @ K).astype(np.float32)
                ldmks_t = transform_points2d(ldmks, ltrans)
                crop_ldmks, crop_weights = self._format_landmarks(ldmks_t, final_w, final_h)
                count = min(len(landmark_visibility), self.num_landmarks)
                crop_weights[:count, 0] *= landmark_visibility[:count]

                final_trans = ltrans.astype(np.float32)
                pad_offset = np.array([pad_x, pad_y], dtype=np.float32)
                bbox_rec = np.array(
                    [original_w * 0.5, original_h * 0.5, float(original_w), float(original_h)],
                    dtype=np.float32,
                )
                crop_size_rec = np.array([final_h, final_w], dtype=np.int64)
                vh, vw = final_h, final_w
            else:
                # ---- Legacy per-person crop path ----
                center = np.asarray(person["center"], dtype=np.float32)
                mamma_scale = float(person["scale"]) / 1.2 * float(self.crop_scale)
                scale = np.array([mamma_scale, mamma_scale], dtype=np.float32)
                trans = get_mamma_affine_transform(center, scale, pixel_std=200.0, output_size=(crop_w, crop_h), rot=0.0)
                crop_img, _ = generate_image_patch_with_trans(image, None, trans, crop_w, crop_h, border_value=0)
                crop_ldmks_raw = transform_points2d(ldmks, trans)
                crop_ldmks, crop_weights = self._format_landmarks(crop_ldmks_raw, crop_w, crop_h)
                count = min(len(landmark_visibility), self.num_landmarks)
                crop_weights[:count, 0] *= landmark_visibility[:count]

                crop_vertices = transform_points2d(vertices_2d[visibility], trans)
                crop_mask = self._mask_from_landmarks(crop_vertices, crop_h, crop_w).astype(np.float32) / 255.0
                if self.pad_to_square:
                    crop_img, crop_mask, crop_ldmks, crop_weights, pad_offset = pad_to_canvas(
                        crop_img,
                        crop_mask,
                        crop_ldmks,
                        crop_weights,
                        final_h,
                        final_w,
                    )
                    pad_h = np.eye(3, dtype=np.float32)
                    pad_h[0, 2] = pad_offset[0]
                    pad_h[1, 2] = pad_offset[1]
                    crop_K = pad_h @ affine_to_homogeneous(trans) @ K
                    final_trans = (pad_h @ affine_to_homogeneous(trans))[:2].astype(np.float32)
                    vh, vw = final_h, final_w
                else:
                    pad_offset = np.array([0, 0], dtype=np.float32)
                    crop_K = affine_to_homogeneous(trans) @ K
                    final_trans = trans.astype(np.float32)
                    vh, vw = crop_h, crop_w
                crop_mask = (crop_mask > 0).astype(np.float32)
                source_side = float(scale[0] * 200.0)
                bbox_rec = np.array([center[0], center[1], source_side, source_side], dtype=np.float32)
                crop_size_rec = np.array([crop_h, crop_w], dtype=np.int64)

            images.append(crop_img)
            depths.append(np.zeros((vh, vw), dtype=np.float32))
            cam_points.append(np.zeros((vh, vw, 3), dtype=np.float32))
            world_points.append(np.zeros((vh, vw, 3), dtype=np.float32))
            point_masks.append(np.zeros((vh, vw), dtype=bool))
            person_masks.append(crop_mask.astype(np.float32))
            extrinsics.append(extri.astype(np.float32))
            intrinsics.append(crop_K.astype(np.float32))
            full_intrinsics.append(K.astype(np.float32))
            full_extrinsics.append(extri.astype(np.float32))
            crop_transforms.append(final_trans.astype(np.float32))
            crop_bboxes.append(bbox_rec)
            pad_offsets.append(pad_offset.astype(np.float32))
            mamma_crop_sizes.append(crop_size_rec)
            landmarks_512.append(crop_ldmks.astype(np.float32))
            landmarks_512_weights.append(crop_weights.astype(np.float32))
            image_paths.append(image_path)
            original_sizes.append(np.array([original_h, original_w], dtype=np.int64))

        smpl_params = self._format_bedlam_smpl_params(first_person, sample["person_id"], int(sample["frame"]))
        batch = {
            "seq_name": f"mamma_{sample['seq_name']}_frame_{sample['frame']}_person_{sample['person_id']:02d}",
            "ids": ids,
            "camera_names": selected_views,
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "person_masks": person_masks,
            "full_intrinsics": full_intrinsics,
            "full_extrinsics": full_extrinsics,
            "crop_transforms": crop_transforms,
            "crop_bboxes": crop_bboxes,
            "pad_offsets": pad_offsets,
            "mamma_crop_sizes": mamma_crop_sizes,
            "landmarks_512": landmarks_512,
            "landmarks_512_weights": landmarks_512_weights,
            "image_paths": image_paths,
            "original_sizes": original_sizes,
        }
        batch.update(smpl_params)
        return batch

    def _format_bedlam_smpl_params(self, person: Dict[str, Any], person_id: int, frame_id: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "person_id": np.asarray(person_id, dtype=np.int64),
            "smplx_frame_id": np.asarray(frame_id, dtype=np.int64),
        }
        if person is None:
            return out
        if "pose_world" in person:
            out["smplx_pose"] = np.asarray(person["pose_world"], dtype=np.float32).reshape(-1)
        elif "pose_cam" in person and person["pose_cam"] is not None:
            out["smplx_pose"] = np.asarray(person["pose_cam"], dtype=np.float32).reshape(-1)
        if "shape" in person:
            out["smplx_betas"] = np.asarray(person["shape"], dtype=np.float32).reshape(-1)
        if "trans_world" in person:
            out["smplx_trans"] = np.asarray(person["trans_world"], dtype=np.float32).reshape(-1)
        elif "trans_cam" in person and person["trans_cam"] is not None:
            out["smplx_trans"] = np.asarray(person["trans_cam"], dtype=np.float32).reshape(-1)
        if "gender" in person:
            out["smplx_gender"] = str(np.asarray(person["gender"]).item())
        return out

    def _camera_names(self, data: Any, sample_dir: str) -> List[str]:
        from_npz = sorted(
            {
                key.split("/")[1]
                for key in data.files
                if key.startswith("cam_param_min/")
                and (
                    key.endswith("/intrinsics.K_flat9.npy")
                    or key.endswith("/intrinsics.K_flat9")
                )
            }
        )
        from_images = sorted(osp.splitext(osp.basename(path))[0] for path in glob.glob(osp.join(sample_dir, "*.jpg")))
        if from_npz and from_images:
            image_set = set(from_images)
            return [name for name in from_npz if name in image_set]
        return from_npz or from_images

    def _load_intrinsic(self, data: Any, cam_name: str) -> np.ndarray:
        key = f"cam_param_min/{cam_name}/intrinsics.K_flat9.npy"
        key = resolve_npz_key(data, key)
        if key is None:
            raise KeyError(f"Missing intrinsics for {cam_name}")
        K = np.asarray(data[key], dtype=np.float32).reshape(3, 3)
        return K

    def _load_extrinsic(self, data: Any, cam_name: str) -> np.ndarray:
        key = f"cam_param_min/{cam_name}/extrinsics.worldToCamera12.npy"
        key = resolve_npz_key(data, key)
        if key is None:
            raise KeyError(f"Missing extrinsics for {cam_name}")
        extri = np.asarray(data[key], dtype=np.float32).reshape(3, 4)
        return extri

    def _load_smpl_params(self, data: Any) -> Dict[str, Any]:
        prefix = f"out_param/person_{self.person_id:02d}/frame_{self.frame_id:05d}/smpl_params"
        out: Dict[str, Any] = {}
        pose_key = f"{prefix}/poses.npy"
        beta_key = f"{prefix}/betas.npy"
        trans_key = f"{prefix}/trans.npy"
        gender_key = f"{prefix}/gender.npy"
        pose_key = resolve_npz_key(data, pose_key)
        beta_key = resolve_npz_key(data, beta_key)
        trans_key = resolve_npz_key(data, trans_key)
        gender_key = resolve_npz_key(data, gender_key)
        if pose_key is not None:
            out["smplx_pose"] = np.asarray(data[pose_key], dtype=np.float32).reshape(-1)
        if beta_key is not None:
            out["smplx_betas"] = np.asarray(data[beta_key], dtype=np.float32).reshape(-1)
        if trans_key is not None:
            out["smplx_trans"] = np.asarray(data[trans_key], dtype=np.float32).reshape(-1)
        if gender_key is not None:
            out["smplx_gender"] = str(np.asarray(data[gender_key]).item())
        out["person_id"] = np.asarray(self.person_id, dtype=np.int64)
        out["smplx_frame_id"] = np.asarray(self.frame_id, dtype=np.int64)
        return out

    def _load_landmarks(self, data: Any, cam_name: str) -> Optional[np.ndarray]:
        key = f"reprojection_data/{cam_name}/people.npy"
        key = resolve_npz_key(data, key)
        if key is None:
            return None
        people = data[key]
        if isinstance(people, np.ndarray) and people.shape == ():
            people = people.item()
        person_obj = self._select_person_object(people)
        if isinstance(person_obj, dict):
            for key in (
                "landmarks_2d",
                "joints_2d_image",
                "joints_2d",
                "joints2d",
                "keypoints_2d_image",
                "keypoints_2d",
                "vertices_2d_image",
                "vertices_2d",
                "points2d",
                "pts2d",
            ):
                if key in person_obj:
                    return np.asarray(person_obj[key], dtype=np.float32)[..., :2]
        return first_array_like(person_obj)

    def _select_person_object(self, people: Any) -> Any:
        if isinstance(people, dict):
            keys = [
                self.person_id,
                str(self.person_id),
                f"person_{self.person_id:02d}",
                f"{self.person_id:02d}",
            ]
            for key in keys:
                if key in people:
                    return people[key]
            if "people" in people:
                return self._select_person_object(people["people"])
            values = list(people.values())
            if values:
                return values[min(self.person_id, len(values) - 1)]
        if isinstance(people, (list, tuple)):
            if not people:
                return None
            return people[min(self.person_id, len(people) - 1)]
        return people

    def _bbox_from_landmarks(self, landmarks: Optional[np.ndarray], image_w: int, image_h: int) -> np.ndarray:
        if landmarks is None or landmarks.size == 0:
            return np.array([0, 0, image_w - 1, image_h - 1], dtype=np.float32)
        pts = np.asarray(landmarks, dtype=np.float32)[..., :2].reshape(-1, 2)
        valid = np.isfinite(pts).all(axis=1)
        valid &= pts[:, 0] >= 0
        valid &= pts[:, 1] >= 0
        if valid.sum() < 4:
            return np.array([0, 0, image_w - 1, image_h - 1], dtype=np.float32)
        pts = pts[valid]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        x1 = np.clip(x1, 0, image_w - 1)
        x2 = np.clip(x2, 0, image_w - 1)
        y1 = np.clip(y1, 0, image_h - 1)
        y2 = np.clip(y2, 0, image_h - 1)
        if x2 <= x1 or y2 <= y1:
            return np.array([0, 0, image_w - 1, image_h - 1], dtype=np.float32)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def _landmark_weights(self, landmarks: np.ndarray, image_w: int, image_h: int, has_landmarks: bool) -> np.ndarray:
        if not has_landmarks:
            return np.zeros((landmarks.shape[0], 1), dtype=np.float32)
        pts = landmarks[..., :2]
        finite = np.isfinite(pts).all(axis=-1)
        in_bounds = (
            finite
            & (pts[..., 0] >= 0)
            & (pts[..., 0] <= image_w - 1)
            & (pts[..., 1] >= 0)
            & (pts[..., 1] <= image_h - 1)
        )
        # Soft exponential falloff for outside-boundary points (matches official BEDLAM_WD).
        # Normalize coords to [-1, 1], compute L2 distance from centre; exp(-beta*|d-1|)
        # peaks at the image boundary (d=1) and decays outward.
        norm_x = np.where(finite, 2.0 * pts[..., 0] / max(image_w, 1) - 1.0, 0.0)
        norm_y = np.where(finite, 2.0 * pts[..., 1] / max(image_h, 1) - 1.0, 0.0)
        dist = np.sqrt(norm_x ** 2 + norm_y ** 2)
        soft = np.exp(-2.0 * np.abs(dist - 1.0))
        weights = np.where(in_bounds, 1.0, np.where(finite, soft, 0.0)).astype(np.float32)
        return weights[..., None]

    def _format_landmarks(
        self,
        landmarks: Optional[np.ndarray],
        image_w: int,
        image_h: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros((self.num_landmarks, 2), dtype=np.float32)
        weights = np.zeros((self.num_landmarks, 1), dtype=np.float32)
        if landmarks is None:
            return out, weights

        landmarks = np.asarray(landmarks, dtype=np.float32).reshape(-1, landmarks.shape[-1])[:, :2]
        count = min(len(landmarks), self.num_landmarks)
        if count == 0:
            return out, weights

        out[:count] = landmarks[:count]
        weights[:count] = self._landmark_weights(out[:count], image_w, image_h, True)

        # Body-part upweighting: hands/feet × 2, hands additionally × hand_weight
        # (matches official BEDLAM_WD target_weight logic)
        if self.body_parts_dict is not None:
            _upweight = {"left_hand", "right_hand", "left_feet", "right_feet"}
            _hand = {"left_hand", "right_hand"}
            for part, idx in self.body_parts_dict.items():
                if part not in _upweight:
                    continue
                valid_idx = idx[idx < count]
                if weights[valid_idx, 0].sum() > 0:
                    weights[valid_idx, 0] *= 2.0
                    if part in _hand:
                        weights[valid_idx, 0] *= self.hand_weight

        return out, weights

    def _load_mask(self, sample_dir: str, cam_name: str) -> Optional[np.ndarray]:
        candidates = [
            osp.join(sample_dir, "masks", f"{cam_name}.png"),
            osp.join(sample_dir, "masks", f"{cam_name}_{self.person_id:02d}.png"),
            osp.join(sample_dir, f"{cam_name}_mask.png"),
            osp.join(sample_dir, f"{cam_name}_{self.person_id:02d}_mask.png"),
        ]
        for path in candidates:
            if osp.isfile(path):
                with Image.open(path) as image:
                    return np.asarray(image.convert("L"))
        return None

    def _mask_from_landmarks(self, landmarks: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
        pts = np.asarray(landmarks, dtype=np.float32)[..., :2].reshape(-1, 2)
        valid = np.isfinite(pts).all(axis=1)
        valid &= pts[:, 0] >= 0
        valid &= pts[:, 0] < image_w
        valid &= pts[:, 1] >= 0
        valid &= pts[:, 1] < image_h
        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        if valid.sum() < 3:
            return mask
        hull = convex_hull(pts[valid])
        image = Image.fromarray(mask)
        draw = ImageDraw.Draw(image)
        draw.polygon([(float(x), float(y)) for x, y in hull], fill=255)
        mask = np.asarray(image)
        return mask

    # ------------------------------------------------------------------
    # Full-scene letterbox helpers (Phase 1)
    # ------------------------------------------------------------------
    def _letterbox_transform(self, src_h, src_w, dst_h, dst_w):
        """Affine (2x3, src->dst px) that resizes the full image to fit a
        dst_h x dst_w canvas preserving aspect ratio, centered with padding."""
        s = min(dst_w / float(src_w), dst_h / float(src_h))
        new_w = min(int(round(src_w * s)), dst_w)
        new_h = min(int(round(src_h * s)), dst_h)
        pad_x = (dst_w - new_w) // 2
        pad_y = (dst_h - new_h) // 2
        trans = np.array([[s, 0.0, pad_x], [0.0, s, pad_y]], dtype=np.float32)
        return trans, new_h, new_w, pad_x, pad_y

    def _warp_image(self, image, new_h, new_w, pad_x, pad_y, dst_h, dst_w):
        img_r = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
        out = np.zeros((dst_h, dst_w, image.shape[2]), dtype=image.dtype)
        out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_r
        return out

    def _warp_binary_mask(self, mask_bool, new_h, new_w, pad_x, pad_y, dst_h, dst_w):
        m = (np.asarray(mask_bool) > 0).astype(np.uint8) * 255
        m_r = np.asarray(Image.fromarray(m).resize((new_w, new_h), Image.NEAREST))
        out = np.zeros((dst_h, dst_w), dtype=np.float32)
        out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = (m_r > 127).astype(np.float32)
        return out

    def _load_scene_person_mask(self, view_dir, frame, person_id, src_h, src_w):
        """Read the multi-person scene mask ({frame}.mask.jpg, values {0,1,2,...})
        and return a boolean mask of the target person (value == person_id+1)."""
        path = osp.join(view_dir, f"{frame}.mask.jpg")
        if not osp.isfile(path):
            return None
        with Image.open(path) as im:
            m = np.asarray(im.convert("L"))
        if m.shape[:2] != (src_h, src_w):
            m = np.asarray(Image.fromarray(m).resize((src_w, src_h), Image.NEAREST))
        return np.abs(m.astype(np.float32) - float(person_id + 1)) < 0.5


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set((float(x), float(y)) for x, y in points[:, :2]))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float32)

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

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float32)
