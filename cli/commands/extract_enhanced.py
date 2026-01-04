"""Enhanced feature extraction command with advanced analysis modules."""
import logging
import os
import warnings
from glob import glob
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from tifffile import imread
from tqdm import tqdm

if TYPE_CHECKING:
    from cli.state import PipelineState

# Suppress FutureWarnings from nmco library
warnings.filterwarnings('ignore', category=FutureWarning, module='nmco')

logger = logging.getLogger(__name__)


def extract_enhanced_features(
    raw_images_dir: str,
    labels_dir: str,
    output_dir: str,
    protein_dirs: Optional[List[str]] = None,
    cell_segmentation: bool = False,
    dilation_radius: int = 10,
    extract_spatial: bool = True,
    gc_mask_dir: Optional[str] = None,
    state: Optional["PipelineState"] = None,
    resume: bool = False,
    # Enhanced feature options
    extract_multiscale: bool = True,
    extract_cell_cycle: bool = True,
    extract_spatial_graph: bool = True,
    extract_relative: bool = True,
    k_neighbors: int = 10,
    wavelet_levels: int = 3,
    density_radii: List[float] = None
):
    """Extract enhanced chrometric and spatial features from segmented images.
    
    This enhanced version includes:
    - Multi-scale wavelet and fractal features
    - Cell cycle state inference
    - Spatial graph-based features (centrality, Voronoi)
    - Relative and interaction features
    
    Args:
        raw_images_dir: Directory with raw DAPI images
        labels_dir: Directory with segmented label images
        output_dir: Output directory for features
        protein_dirs: List of directories with protein channel images
        cell_segmentation: Perform cell segmentation by dilation
        dilation_radius: Radius for cell boundary dilation
        extract_spatial: Extract spatial coordinates
        gc_mask_dir: Directory with germinal center mask annotations
        state: Pipeline state manager for tracking progress
        resume: Whether to skip already processed files
        extract_multiscale: Extract wavelet and fractal features
        extract_cell_cycle: Infer cell cycle state
        extract_spatial_graph: Extract graph-based spatial features
        extract_relative: Extract relative and interaction features
        k_neighbors: Number of neighbors for spatial analysis
        wavelet_levels: Number of wavelet decomposition levels
        density_radii: Radii for local density computation
    """
    # Import here to avoid slow startup
    from src.features.feature_extraction import run_nuclear_chromatin_feat_ext
    from src.features.multiscale_features import extract_all_multiscale_features
    from src.features.cell_cycle import infer_cell_cycle_state, compute_cell_cycle_features
    from src.features.relative_features import extract_all_relative_features
    from src.analysis.spatial_graph import extract_all_spatial_graph_features
    from skimage import measure, segmentation
    from PIL import Image
    
    if density_radii is None:
        density_radii = [25, 50, 100]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Normalize paths for cross-platform compatibility
    raw_images_dir = os.path.normpath(raw_images_dir)
    labels_dir = os.path.normpath(labels_dir)
    
    # Get image files
    raw_image_files = sorted(glob(os.path.join(raw_images_dir, "*.tif")))
    label_files = sorted(glob(os.path.join(labels_dir, "*.tif")))
    
    if not raw_image_files:
        raise FileNotFoundError(f"No .tif files found in {raw_images_dir}")
    if not label_files:
        raise FileNotFoundError(f"No .tif files found in {labels_dir}")
    
    if len(raw_image_files) != len(label_files):
        logger.warning(f"Mismatch: {len(raw_image_files)} raw images vs {len(label_files)} label images")
    
    # Create subdirectories
    features_dir = os.path.join(output_dir, "chrometric_features")
    spatial_dir = os.path.join(output_dir, "spatial_coordinates")
    consolidated_dir = os.path.join(output_dir, "consolidated_features")
    enhanced_dir = os.path.join(output_dir, "enhanced_features")
    
    for d in [features_dir, spatial_dir, consolidated_dir, enhanced_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting enhanced features from {len(raw_image_files)} images...")
    logger.info(f"  - Multi-scale features: {extract_multiscale}")
    logger.info(f"  - Cell cycle inference: {extract_cell_cycle}")
    logger.info(f"  - Spatial graph features: {extract_spatial_graph}")
    logger.info(f"  - Relative features: {extract_relative}")
    
    # Update state with total files
    if state:
        state.state['steps']['extract']['total_files'] = len(raw_image_files)
        state.save()
    
    # Determine which files need processing
    files_to_process = []
    for i in range(len(raw_image_files)):
        filename = os.path.basename(label_files[i])
        if resume and state and state.is_file_processed('extract', filename):
            continue
        files_to_process.append((i, raw_image_files[i], label_files[i]))
    
    if resume and len(files_to_process) < len(raw_image_files):
        logger.info(f"Resuming: {len(raw_image_files) - len(files_to_process)} images already processed")
    
    logger.info(f"Found {len(raw_image_files)} images, {len(files_to_process)} to process")
    
    # Load existing features if resuming
    all_features = pd.DataFrame()
    all_spatial = pd.DataFrame()
    all_enhanced = pd.DataFrame()
    
    existing_features_file = os.path.join(consolidated_dir, "nuc_features.csv")
    existing_enhanced_file = os.path.join(enhanced_dir, "enhanced_features.csv")
    
    if resume:
        if os.path.exists(existing_features_file):
            try:
                all_features = pd.read_csv(existing_features_file)
                logger.info(f"Loaded {len(all_features)} existing feature records")
            except Exception as e:
                logger.warning(f"Could not load existing features: {e}")
        
        if os.path.exists(existing_enhanced_file):
            try:
                all_enhanced = pd.read_csv(existing_enhanced_file)
                logger.info(f"Loaded {len(all_enhanced)} existing enhanced feature records")
            except Exception as e:
                logger.warning(f"Could not load existing enhanced features: {e}")
    
    failed_images = []
    
    for idx, (i, raw_file, label_file) in enumerate(tqdm(files_to_process, desc="Extracting features")):
        filename = os.path.basename(label_file)
        img_name = os.path.splitext(filename)[0]
        
        try:
            # ===== Standard chrometric features =====
            features = run_nuclear_chromatin_feat_ext(
                raw_file,
                label_file,
                features_dir,
                normalize=True,
                save_output=False,
            )
            
            if features.empty:
                logger.warning(f"No features extracted from {filename}")
                continue
            
            # Save individual feature file
            features.to_csv(os.path.join(features_dir, f"{img_name}.csv"), index=False)
            
            # Add image identifier
            features["image"] = img_name
            features["nuc_id"] = features["image"].astype(str) + "_" + features["label"].astype(str)
            
            # ===== Enhanced features =====
            enhanced_features = pd.DataFrame()
            
            # Read images for enhanced analysis
            raw_image = imread(raw_file)
            label_image = imread(label_file)
            props = measure.regionprops(label_image, raw_image)
            
            if extract_multiscale:
                logger.debug(f"Extracting multi-scale features for {img_name}")
                multiscale_list = []
                
                for prop in props:
                    try:
                        ms_feat = extract_all_multiscale_features(
                            prop.intensity_image,
                            prop.image,
                            wavelet_levels=wavelet_levels
                        )
                        ms_feat['label'] = prop.label
                        multiscale_list.append(ms_feat)
                    except Exception as e:
                        logger.debug(f"Multi-scale failed for nucleus {prop.label}: {e}")
                
                if multiscale_list:
                    multiscale_df = pd.concat(multiscale_list, ignore_index=True)
                    enhanced_features = pd.concat([enhanced_features, multiscale_df], axis=1)
            
            if extract_cell_cycle:
                logger.debug(f"Inferring cell cycle for {img_name}")
                cc_list = []
                
                for prop in props:
                    try:
                        cc_feat = compute_cell_cycle_features(
                            prop.intensity_image,
                            prop.image
                        )
                        cc_feat['label'] = prop.label
                        cc_list.append(cc_feat)
                    except Exception as e:
                        logger.debug(f"Cell cycle failed for nucleus {prop.label}: {e}")
                
                if cc_list:
                    cc_df = pd.concat(cc_list, ignore_index=True)
                    # Infer cell cycle state
                    combined_for_cc = pd.merge(features, cc_df, on='label', how='left')
                    cc_predictions = infer_cell_cycle_state(combined_for_cc)
                    
                    if not cc_df.empty:
                        enhanced_features = pd.concat([enhanced_features, cc_df.drop('label', axis=1, errors='ignore')], axis=1)
                    if not cc_predictions.empty:
                        enhanced_features = pd.concat([enhanced_features, cc_predictions], axis=1)
            
            # Add identifiers to enhanced features
            if not enhanced_features.empty:
                if 'label' in enhanced_features.columns:
                    enhanced_features = enhanced_features.drop('label', axis=1, errors='ignore')
                enhanced_features['image'] = img_name
                enhanced_features['nuc_id'] = features['nuc_id'].values
            
            # Accumulate features
            all_features = pd.concat([all_features, features], ignore_index=True)
            if not enhanced_features.empty:
                all_enhanced = pd.concat([all_enhanced, enhanced_features], ignore_index=True)
            
        except Exception as e:
            logger.warning(f"Failed to extract features from {filename}: {str(e)}")
            failed_images.append({'image': img_name, 'error': str(e)})
        
        # Mark file as processed and save periodically
        if state:
            state.mark_file_processed('extract', filename)
            if (idx + 1) % 2 == 0:
                all_features.to_csv(existing_features_file, index=False)
                if not all_enhanced.empty:
                    all_enhanced.to_csv(existing_enhanced_file, index=False)
                state.save()
    
    # Log failed images summary
    if failed_images:
        logger.warning(f"{len(failed_images)} images failed feature extraction:")
        for fail in failed_images:
            logger.warning(f"  - {fail['image']}: {fail['error']}")
        pd.DataFrame(failed_images).to_csv(os.path.join(consolidated_dir, "failed_images.csv"), index=False)
    
    # ===== Extract spatial coordinates =====
    if extract_spatial:
        logger.info("Extracting spatial coordinates...")
        for label_file in label_files:
            labelled_image = imread(label_file)
            img_name = os.path.splitext(os.path.basename(label_file))[0]
            
            spatial_data = pd.DataFrame(
                measure.regionprops_table(labelled_image, properties=("label", "centroid"))
            )
            spatial_data["image"] = img_name
            spatial_data["nuc_id"] = spatial_data["image"].astype(str) + "_" + spatial_data["label"].astype(str)
            all_spatial = pd.concat([all_spatial, spatial_data], ignore_index=True)
    
    # ===== Spatial graph and relative features (require all cells) =====
    if extract_spatial and not all_spatial.empty:
        logger.info("Computing spatial graph and relative features...")
        
        # Group by image for spatial analysis
        for img_name in all_spatial['image'].unique():
            img_mask = all_spatial['image'] == img_name
            img_spatial = all_spatial[img_mask].reset_index(drop=True)
            img_features = all_features[all_features['image'] == img_name].reset_index(drop=True)
            
            if len(img_spatial) < 3:
                continue
            
            try:
                if extract_spatial_graph:
                    logger.debug(f"Computing spatial graph features for {img_name}")
                    graph_feat = extract_all_spatial_graph_features(
                        img_spatial,
                        features=img_features,
                        k_neighbors=k_neighbors,
                        density_radii=density_radii,
                        compute_autocorrelation=True
                    )
                    if not graph_feat.empty:
                        graph_feat['image'] = img_name
                        graph_feat['nuc_id'] = img_spatial['nuc_id'].values
                        all_enhanced = pd.concat([all_enhanced, graph_feat], ignore_index=True)
                
                if extract_relative:
                    logger.debug(f"Computing relative features for {img_name}")
                    relative_feat = extract_all_relative_features(
                        img_features,
                        img_spatial,
                        k_neighbors=k_neighbors,
                        compute_gradients=True
                    )
                    if not relative_feat.empty:
                        relative_feat['image'] = img_name
                        relative_feat['nuc_id'] = img_spatial['nuc_id'].values
                        
                        # Merge with existing enhanced features
                        if 'nuc_id' in all_enhanced.columns:
                            all_enhanced = pd.merge(
                                all_enhanced, relative_feat,
                                on=['image', 'nuc_id'], how='outer'
                            )
                        else:
                            all_enhanced = pd.concat([all_enhanced, relative_feat], ignore_index=True)
                
            except Exception as e:
                logger.warning(f"Spatial analysis failed for {img_name}: {e}")
    
    # ===== Cell segmentation (optional) =====
    if cell_segmentation:
        cell_labels_dir = os.path.join(output_dir, "cell_labels")
        Path(cell_labels_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Performing cell segmentation (dilation radius={dilation_radius})...")
        for label_file in tqdm(label_files, desc="Cell segmentation"):
            labelled_image = imread(label_file)
            cell_boundaries = segmentation.expand_labels(labelled_image, distance=dilation_radius)
            
            output_path = os.path.join(cell_labels_dir, os.path.basename(label_file))
            im = Image.fromarray(cell_boundaries)
            im.save(output_path)
    
    # ===== Protein intensities (optional) =====
    if protein_dirs:
        from src.features.intensity_features import measure_intensity_features
        
        seg_dir = os.path.join(output_dir, "cell_labels") if cell_segmentation else labels_dir
        
        for protein_dir in protein_dirs:
            protein_dir = os.path.normpath(protein_dir)
            if not os.path.exists(protein_dir):
                logger.warning(f"Protein directory not found: {protein_dir}")
                continue
            
            protein_name = os.path.basename(protein_dir)
            protein_output_dir = os.path.join(output_dir, f"{protein_name}_levels")
            Path(protein_output_dir).mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Measuring {protein_name} intensities...")
            seg_files = sorted(glob(os.path.join(seg_dir, "*.tif")))
            prot_files = sorted(glob(os.path.join(protein_dir, "*.tif")))
            
            protein_features = pd.DataFrame()
            for j in tqdm(range(min(len(seg_files), len(prot_files))), desc=f"{protein_name}"):
                labelled_image = imread(seg_files[j])
                protein_image = imread(prot_files[j])
                
                props = measure.regionprops(labelled_image, protein_image)
                features_list = []
                
                for prop in props:
                    try:
                        intensity_feat = measure_intensity_features(
                            regionmask=prop.image,
                            intensity=prop.intensity_image,
                            measure_int_dist=True,
                            measure_hc_ec_ratios=False
                        )
                        intensity_feat['label'] = prop.label
                        features_list.append(intensity_feat)
                    except Exception as e:
                        logger.debug(f"Intensity extraction failed for label {prop.label}: {e}")
                        feat_row = pd.DataFrame([{
                            'label': prop.label,
                            'int_mean': prop.mean_intensity,
                            'int_max': prop.max_intensity,
                            'int_min': prop.min_intensity
                        }])
                        features_list.append(feat_row)
                
                if features_list:
                    feat_df = pd.concat(features_list, ignore_index=True)
                    img_name = os.path.splitext(os.path.basename(seg_files[j]))[0]
                    feat_df["image"] = img_name
                    feat_df["nuc_id"] = feat_df["image"].astype(str) + "_" + feat_df["label"].astype(str)
                    feat_df.to_csv(os.path.join(protein_output_dir, f"{img_name}.csv"), index=False)
                    protein_features = pd.concat([protein_features, feat_df], ignore_index=True)
            
            protein_features.to_csv(os.path.join(consolidated_dir, f"{protein_name}_levels.csv"), index=False)
    
    # ===== GC mask (optional) =====
    if gc_mask_dir and os.path.exists(gc_mask_dir):
        from src.features.intensity_features import measure_intensity_features
        
        gc_mask_dir = os.path.normpath(gc_mask_dir)
        gc_output_dir = os.path.join(output_dir, "gc_levels")
        Path(gc_output_dir).mkdir(parents=True, exist_ok=True)
        
        seg_dir = os.path.join(output_dir, "cell_labels") if cell_segmentation else labels_dir
        
        logger.info("Measuring germinal center mask intensities...")
        seg_files = sorted(glob(os.path.join(seg_dir, "*.tif")))
        gc_mask_files = sorted(glob(os.path.join(gc_mask_dir, "*.tif")))
        
        gc_features = pd.DataFrame()
        for j in tqdm(range(min(len(seg_files), len(gc_mask_files))), desc="GC mask"):
            labelled_image = imread(seg_files[j])
            gc_mask_image = imread(gc_mask_files[j])
            
            props = measure.regionprops(labelled_image, gc_mask_image)
            features_list = []
            
            for prop in props:
                try:
                    intensity_feat = measure_intensity_features(
                        regionmask=prop.image,
                        intensity=prop.intensity_image,
                        measure_int_dist=True,
                        measure_hc_ec_ratios=False
                    )
                    intensity_feat['label'] = prop.label
                    features_list.append(intensity_feat)
                except Exception as e:
                    logger.debug(f"GC intensity extraction failed for label {prop.label}: {e}")
                    feat_row = pd.DataFrame([{
                        'label': prop.label,
                        'int_mean': prop.mean_intensity,
                        'int_max': prop.max_intensity,
                        'int_min': prop.min_intensity
                    }])
                    features_list.append(feat_row)
            
            if features_list:
                feat_df = pd.concat(features_list, ignore_index=True)
                img_name = os.path.splitext(os.path.basename(seg_files[j]))[0]
                feat_df["image"] = img_name
                feat_df["nuc_id"] = feat_df["image"].astype(str) + "_" + feat_df["label"].astype(str)
                feat_df.to_csv(os.path.join(gc_output_dir, f"{img_name}.csv"), index=False)
                gc_features = pd.concat([gc_features, feat_df], ignore_index=True)
        
        gc_features.to_csv(os.path.join(consolidated_dir, "gc_levels.csv"), index=False)
        logger.info(f"GC mask intensities saved to {os.path.join(consolidated_dir, 'gc_levels.csv')}")
    
    # ===== Save consolidated outputs =====
    all_features.to_csv(os.path.join(consolidated_dir, "nuc_features.csv"), index=False)
    
    if extract_spatial and not all_spatial.empty:
        all_spatial.to_csv(os.path.join(consolidated_dir, "spatial_coordinates.csv"), index=False)
    
    if not all_enhanced.empty:
        # Remove duplicate columns
        all_enhanced = all_enhanced.loc[:, ~all_enhanced.columns.duplicated()]
        all_enhanced.to_csv(os.path.join(enhanced_dir, "enhanced_features.csv"), index=False)
        
        # Merge all features
        merged_features = pd.merge(all_features, all_enhanced, on=['image', 'nuc_id'], how='left')
        merged_features.to_csv(os.path.join(consolidated_dir, "all_features_merged.csv"), index=False)
        logger.info(f"Merged features saved to {os.path.join(consolidated_dir, 'all_features_merged.csv')}")
    
    # Save final state
    if state:
        state.save()
    
    logger.info(f"Enhanced feature extraction complete. Results saved to {consolidated_dir}")
    logger.info(f"  - {len(all_features)} nuclei processed")
    logger.info(f"  - {len(all_features.columns)} base features extracted")
    if not all_enhanced.empty:
        logger.info(f"  - {len(all_enhanced.columns)} enhanced features extracted")

