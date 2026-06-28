"""Build a multi-view train/test split from a MAMMA *_contact dataset (e.g. harmony4d).

- keeps only seqs with >= min_views camera-tars (each tar = one IOI camera)
- extracts the selected seqs' tars into OUT/train and OUT/test (separate dirs, so the
  existing glob-everything MammaDataset gets a real split with zero code change)
- optional frame_stride subsamples frames CONSISTENTLY across all views
- validates each extracted seq: >= min_views IOI dirs, balanced jpg/mask/data.pyd
- writes train_seqs.txt / test_seqs.txt / summary.json and prints the MAMMA_DIRs

Usage (from training/):
  python data/prepare_harmony4d_split.py \
      --src /mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma/harmony4d_train_1_NC_200_00_contact \
      --out_dir /mnt/train-data-4-hdd/yian/Mamma_mv_split \
      --num_seqs 12 --train_ratio 0.8 --frame_stride 5 --min_views 4
"""
import argparse, glob, json, os, os.path as osp, random, shutil, subprocess, sys


def seq_camera_count(seq_dir):
    return len(glob.glob(osp.join(seq_dir, "*.tar")))


def extract_seq(seq_dir, out_root):
    """Extract all camera-tars of one seq into out_root (preserving internal paths)."""
    for tar in sorted(glob.glob(osp.join(seq_dir, "*.tar"))):
        subprocess.run(["tar", "-xf", tar, "-C", out_root],
                       stderr=subprocess.DEVNULL, check=False)


def find_extracted_seq_dir(out_root, seq_name):
    hits = glob.glob(osp.join(out_root, "**", seq_name), recursive=True)
    hits = [h for h in hits if osp.isdir(h)]
    return hits[0] if hits else None


def prune_frames(seq_extracted_dir, stride):
    if stride <= 1:
        return
    views = [d for d in glob.glob(osp.join(seq_extracted_dir, "*")) if osp.isdir(d)]
    # global frame stems (assume synchronized across views)
    stems = set()
    for v in views:
        for f in glob.glob(osp.join(v, "*.jpg")):
            b = osp.basename(f)
            if b.endswith(".mask.jpg"):
                continue
            stems.add(b[:-4])
    keep = set(sorted(stems)[::stride])
    for v in views:
        for f in glob.glob(osp.join(v, "*")):
            b = osp.basename(f)
            stem = b.split(".")[0]
            if stem not in keep:
                os.remove(f)


def validate_seq(seq_extracted_dir, min_views):
    views = [d for d in glob.glob(osp.join(seq_extracted_dir, "*")) if osp.isdir(d)]
    ok_views = 0
    for v in views:
        n_jpg = len([f for f in glob.glob(osp.join(v, "*.jpg")) if not f.endswith(".mask.jpg")])
        n_mask = len(glob.glob(osp.join(v, "*.mask.jpg")))
        n_pyd = len(glob.glob(osp.join(v, "*.data.pyd")))
        if n_jpg > 0 and n_jpg == n_mask == n_pyd:
            ok_views += 1
    return ok_views, len(views)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/train-data-4-hdd/yian/SMPL_multi_dataset/mamma/harmony4d_train_1_NC_200_00_contact")
    ap.add_argument("--out_dir", default="/mnt/train-data-4-hdd/yian/Mamma_mv_split")
    ap.add_argument("--num_seqs", type=int, default=12)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--frame_stride", type=int, default=5)
    ap.add_argument("--min_views", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    seq_dirs = sorted(d for d in glob.glob(osp.join(args.src, "be_*")) if osp.isdir(d))
    qualified = [d for d in seq_dirs if seq_camera_count(d) >= args.min_views]
    print(f"{len(seq_dirs)} seqs total, {len(qualified)} with >= {args.min_views} cameras")

    random.shuffle(qualified)
    chosen = qualified[: args.num_seqs]
    n_train = max(1, round(len(chosen) * args.train_ratio))
    splits = {"train": chosen[:n_train], "test": chosen[n_train:]}
    print(f"selected {len(chosen)} seqs -> train {len(splits['train'])} / test {len(splits['test'])}")

    summary = {"src": args.src, "frame_stride": args.frame_stride, "splits": {}}
    for split, seqs in splits.items():
        out_root = osp.join(args.out_dir, split)
        os.makedirs(out_root, exist_ok=True)
        valid = []
        for i, sd in enumerate(seqs):
            name = osp.basename(sd)
            ncam = seq_camera_count(sd)
            print(f"  [{split} {i+1}/{len(seqs)}] {name} ({ncam} cam-tars) extracting...", flush=True)
            extract_seq(sd, out_root)
            ext = find_extracted_seq_dir(out_root, name)
            if ext is None:
                print(f"     !! extraction produced no dir for {name}; skipping")
                continue
            prune_frames(ext, args.frame_stride)
            ok_views, tot_views = validate_seq(ext, args.min_views)
            status = "OK" if ok_views >= args.min_views else "DROP(<min_views)"
            print(f"     -> {ok_views}/{tot_views} complete views  [{status}]")
            if ok_views >= args.min_views:
                valid.append(name)
        with open(osp.join(args.out_dir, f"{split}_seqs.txt"), "w") as f:
            f.write("\n".join(valid) + ("\n" if valid else ""))
        # MAMMA_DIR = the level containing the seq dirs' parent tree; recursive glob works from out_root
        summary["splits"][split] = {"num_valid_seqs": len(valid), "MAMMA_DIR": out_root, "seqs": valid}

    with open(osp.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== DONE ====")
    for split in ("train", "test"):
        s = summary["splits"][split]
        print(f"  {split}: {s['num_valid_seqs']} valid seqs | MAMMA_DIR = {s['MAMMA_DIR']}")
    du = subprocess.run(["du", "-sh", args.out_dir], capture_output=True, text=True).stdout.strip()
    print(f"  disk: {du}")
    print(f"  summary -> {osp.join(args.out_dir, 'summary.json')}")
    print("\nSet in mamma_smplx.yaml:  train MAMMA_DIR -> .../train , val MAMMA_DIR -> .../test")


if __name__ == "__main__":
    main()
