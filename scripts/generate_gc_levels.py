#!/usr/bin/env python
"""
Generate gc_levels.csv from germinal center annotations.
This measures the mean intensity of cells relative to the GC mask.
"""
import os
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from tifffile import imread
from skimage import measure
from tqdm import tqdm


def generate_gc_levels(
    seg_dir: str,
    gc_mask_dir: str,
    output_path: str
):
    """Generate GC levels from segmentation and GC mask
    
    Args:
        seg_dir: Directory with segmented cell labels
        gc_mask_dir: Directory with GC mask annotations
        output_path: Path to save gc_levels.csv
    """
    seg_files = sorted(glob(os.path.join(seg_dir, "*.tif")))
    gc_mask_files = sorted(glob(os.path.join(gc_mask_dir, "*.tif")))
    
    print(f"Found {len(seg_files)} segmentation files")
    print(f"Found {len(gc_mask_files)} GC mask files")
    
    gc_features = pd.DataFrame()
    
    for i in tqdm(range(min(len(seg_files), len(gc_mask_files))), desc="GC mask"):
        labelled_image = imread(seg_files[i])
        gc_mask_image = imread(gc_mask_files[i])
        
        props = measure.regionprops(labelled_image, gc_mask_image)
        features = []
        
        for prop in props:
            features.append({
                'label': prop.label,
                'int_mean': prop.mean_intensity,
                'int_max': prop.max_intensity,
                'int_min': prop.min_intensity
            })
        
        img_name = os.path.splitext(os.path.basename(seg_files[i]))[0]
        features_df = pd.DataFrame(features)
        features_df["image"] = img_name
        features_df["nuc_id"] = features_df["image"].astype(str) + "_" + features_df["label"].astype(str)
        
        gc_features = pd.concat([gc_features, features_df], ignore_index=True)
    
    # Save
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    gc_features.to_csv(output_path, index=False)
    print(f"Saved GC levels to {output_path}")
    print(f"  - {len(gc_features)} nuclei processed")
    print(f"  - {(gc_features['int_mean'] > 0).sum()} nuclei inside GC")
    
    return gc_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate gc_levels.csv from GC annotations")
    parser.add_argument("--seg-dir", required=True, help="Directory with segmented cell labels")
    parser.add_argument("--gc-mask-dir", required=True, help="Directory with GC mask annotations")
    parser.add_argument("--output", required=True, help="Output path for gc_levels.csv")
    
    args = parser.parse_args()
    
    generate_gc_levels(args.seg_dir, args.gc_mask_dir, args.output)

