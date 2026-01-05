#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal pipeline: radiomics voxel-feature extraction -> collect voxel pairs -> clustering -> save habitat masks.
All paths are configured via arguments or environment.
"""

import argparse
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

DEFAULT_FEATURE_YAML = "feature_extractor_config.yaml"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_image_for_case(struct_dir: Path, case_id: str):
    # try common patterns
    candidates = list(struct_dir.glob(f"{case_id}*"))
    return candidates[0] if candidates else None

def extract_entropy_images(seg_dir: Path, struct_dir: Path, entropy_dir: Path, extractor_yaml: str):
    extractor = featureextractor.RadiomicsFeatureExtractor(extractor_yaml)
    ensure_dir(entropy_dir)
    seg_files = sorted([p for p in seg_dir.iterdir() if p.is_file()])
    for seg_path in seg_files:
        case_id = seg_path.stem.split("_segm")[0]
        img_path = find_image_for_case(struct_dir, case_id)
        if img_path is None:
            continue
        try:
            res = extractor.execute(str(img_path), str(seg_path), voxelBased=True)
        except Exception:
            continue
        # find entropy key
        key = next((k for k in res.keys() if "entropy" in k.lower()), None)
        if key is None:
            continue
        vol = res[key]
        if isinstance(vol, sitk.Image):
            out_img = vol
        else:
            # convert numpy-like to SimpleITK image and reuse geometry from img_path
            arr = np.asarray(vol)
            out_img = sitk.GetImageFromArray(arr)
            src = sitk.ReadImage(str(img_path))
            out_img.SetOrigin(src.GetOrigin())
            out_img.SetDirection(src.GetDirection())
            out_img.SetSpacing(src.GetSpacing())
        out_path = entropy_dir / f"{case_id}_entropy.nrrd"
        sitk.WriteImage(out_img, str(out_path))

def collect_voxel_pairs(seg_dir: Path, struct_dir: Path, entropy_dir: Path):
    seg_files = sorted([p for p in seg_dir.iterdir() if p.is_file()])
    rows = []
    filecount = 0
    for seg_path in seg_files:
        case_id = seg_path.stem.split("_segm")[0]
        img_path = find_image_for_case(struct_dir, case_id)
        entropy_path = entropy_dir / f"{case_id}_entropy.nrrd"
        if not img_path or not entropy_path.exists():
            continue
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
        entropy = sitk.GetArrayFromImage(sitk.ReadImage(str(entropy_path)))
        valid = (mask == 1) & (~np.isnan(entropy))
        if not np.any(valid):
            filecount += 1
            continue
        coords = np.where(valid)
        linear_idx = np.ravel_multi_index(coords, img.shape)
        gray_vals = img[coords]
        entropy_vals = entropy[coords]
        filecount += 1
        df = pd.DataFrame({
            "gray": gray_vals,
            "entropy": entropy_vals,
            "filecount": filecount,
            "seg": seg_path.name,
            "index": linear_idx
        })
        rows.append(df)
    if rows:
        couples = pd.concat(rows, ignore_index=True)
    else:
        couples = pd.DataFrame(columns=["gray", "entropy", "filecount", "seg", "index"])
    return couples

def run_clustering(couples: pd.DataFrame, n_clusters: int, random_state: int = 0):
    scaler = MinMaxScaler()
    features = scaler.fit_transform(couples[["gray", "entropy"]].values)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(features)
    couples = couples.copy()
    couples["cluster"] = labels + 1
    return couples, kmeans, scaler

def save_habitat_masks(couples: pd.DataFrame, seg_dir: Path, out_dir: Path, n_clusters: int):
    ensure_dir(out_dir)
    seg_files = sorted([p for p in seg_dir.iterdir() if p.is_file()])
    # map seg name -> filecount occurrence(s)
    for seg_path in seg_files:
        seg_name = seg_path.name
        subset = couples[couples["seg"] == seg_name]
        if subset.empty:
            continue
        ori = sitk.ReadImage(str(seg_path))
        ori_arr = sitk.GetArrayFromImage(ori)
        shape = ori_arr.shape
        new_arr = np.zeros_like(ori_arr, dtype=np.int32)
        indices = subset["index"].astype(np.int64).to_numpy()
        labels = subset["cluster"].to_numpy().astype(np.int32)
        # If duplicates exist per index, last label wins
        flat = new_arr.ravel()
        flat[indices] = labels
        new_arr = flat.reshape(shape)
        out_img = sitk.GetImageFromArray(new_arr)
        out_img.SetOrigin(ori.GetOrigin())
        out_img.SetDirection(ori.GetDirection())
        out_img.SetSpacing(ori.GetSpacing())
        out_path = out_dir / f"{seg_path.stem}_habitat.nrrd"
        sitk.WriteImage(out_img, str(out_path))
        # optionally save separate binary masks
        folder = out_dir / seg_path.stem
        ensure_dir(folder)
        for c in range(1, n_clusters + 1):
            bin_arr = (new_arr == c).astype(np.uint8)
            if not bin_arr.any():
                continue
            bin_img = sitk.GetImageFromArray(bin_arr)
            bin_img.SetOrigin(ori.GetOrigin())
            bin_img.SetDirection(ori.GetDirection())
            bin_img.SetSpacing(ori.GetSpacing())
            sitk.WriteImage(bin_img, str(folder / f"{seg_path.stem}_cluster{c}.nrrd"))

def save_summary_table(couples: pd.DataFrame, out_path: Path):
    table = pd.crosstab(couples["seg"], couples["cluster"])
    table.to_excel(str(out_path))

def plot_scatter(couples: pd.DataFrame, out_path: Path, n_clusters: int):
    plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap("tab10")
    for c in range(1, n_clusters + 1):
        sel = couples["cluster"] == c
        plt.scatter(couples.loc[sel, "gray"], couples.loc[sel, "entropy"],
                    s=4, c=[cmap((c-1) % 10)], label=f"Cluster {c}", alpha=0.6)
    plt.xlabel("Normalized Gray")
    plt.ylabel("Normalized Entropy")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=300)
    plt.close()

def parse_args():
    p = argparse.ArgumentParser(description="Habitat pipeline")
    p.add_argument("--struct-dir", required=True, help="Structural images directory")
    p.add_argument("--seg-dir", required=True, help="Segmentation masks directory")
    p.add_argument("--entropy-dir", required=True, help="Entropy output directory")
    p.add_argument("--habitat-out", required=True, help="Habitat output directory")
    p.add_argument("--feature-yaml", default=DEFAULT_FEATURE_YAML, help="Radiomics extractor yaml")
    p.add_argument("--n-clusters", type=int, default=4, help="Number of clusters")
    p.add_argument("--skip-extract", action="store_true", help="Skip radiomics entropy extraction")
    p.add_argument("--plot-path", default="clustering_scatter.pdf", help="Scatter plot path")
    p.add_argument("--summary", default="segmentation_labels_count.xlsx", help="Summary table path")
    p.add_argument("--seed", type=int, default=99, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()
    struct_dir = Path(args.struct_dir)
    seg_dir = Path(args.seg_dir)
    entropy_dir = Path(args.entropy_dir)
    habitat_out = Path(args.habitat_out)
    ensure_dir(habitat_out)

    if not args.skip_extract:
        extract_entropy_images(seg_dir, struct_dir, entropy_dir, args.feature_yaml)

    couples = collect_voxel_pairs(seg_dir, struct_dir, entropy_dir)
    if couples.empty:
        sys.exit(0)
    couples, kmeans, scaler = run_clustering(couples, args.n_clusters, random_state=args.seed)
    save_habitat_masks(couples, seg_dir, habitat_out, args.n_clusters)
    save_summary_table(couples, Path(args.summary))
    plot_scatter(couples, Path(args.plot_path), args.n_clusters)

if __name__ == "__main__":
    main()