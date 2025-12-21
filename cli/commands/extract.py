"""Feature extraction command"""
import logging
import os
import warnings
from glob import glob
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import pandas as pd
from tifffile import imread
from tqdm import tqdm

if TYPE_CHECKING:
    from cli.state import PipelineState

# Suppress FutureWarnings from nmco library (scipy.stats.mode deprecation)
warnings.filterwarnings('ignore', category=FutureWarning, module='nmco')

logger = logging.getLogger(__name__)


def extract_features(
    raw_images_dir: str,
    labels_dir: str,
    output_dir: str,
    protein_dirs: Optional[List[str]] = None,
    cell_segmentation: bool = False,
    dilation_radius: int = 10,
    extract_spatial: bool = True,
    gc_mask_dir: Optional[str] = None,
    state: Optional["PipelineState"] = None,
    resume: bool = False
):
    """Extract chrometric and intensity features from segmented images
    
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
    """
    # Import here to avoid slow startup
    from src.features.feature_extraction import run_nuclear_chromatin_feat_ext
    from skimage import measure, segmentation
    from PIL import Image
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Normalize paths for cross-platform compatibility
    raw_images_dir = os.path.normpath(raw_images_dir)
    labels_dir = os.path.normpath(labels_dir)
    
    # Get image files using os.path.join for cross-platform glob
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
    Path(features_dir).mkdir(parents=True, exist_ok=True)
    Path(spatial_dir).mkdir(parents=True, exist_ok=True)
    Path(consolidated_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract nuclear chrometric features
    logger.info(f"Extracting chrometric features from {len(raw_image_files)} images...")
    
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
    existing_features_file = os.path.join(consolidated_dir, "nuc_features.csv")
    if resume and os.path.exists(existing_features_file):
        try:
            all_features = pd.read_csv(existing_features_file)
            logger.info(f"Loaded {len(all_features)} existing feature records")
        except Exception as e:
            logger.warning(f"Could not load existing features: {e}")
            all_features = pd.DataFrame()
    
    failed_images = []
    for idx, (i, raw_file, label_file) in enumerate(tqdm(files_to_process, desc="Extracting features")):
        filename = os.path.basename(label_file)
        img_name = os.path.splitext(filename)[0]
        
        try:
            features = run_nuclear_chromatin_feat_ext(
                raw_file,
                label_file,
                features_dir,
                normalize=True,
                save_output=False,  # We save consolidated features ourselves
            )
            
            # Save individual feature file
            features.to_csv(os.path.join(features_dir, f"{img_name}.csv"), index=False)
            
            # Add image identifier
            features["image"] = img_name
            all_features = pd.concat([all_features, features], ignore_index=True)
            
        except Exception as e:
            logger.warning(f"Failed to extract features from {filename}: {str(e)}")
            failed_images.append({'image': img_name, 'error': str(e)})
            # Continue with next image instead of crashing
        
        # Mark file as processed and save periodically
        if state:
            state.mark_file_processed('extract', filename)
            # Save progress every 2 files
            if (idx + 1) % 2 == 0:
                all_features.to_csv(existing_features_file, index=False)
                state.save()
    
    # Log failed images summary
    if failed_images:
        logger.warning(f"{len(failed_images)} images failed feature extraction:")
        for fail in failed_images:
            logger.warning(f"  - {fail['image']}: {fail['error']}")
        # Save failed images list
        pd.DataFrame(failed_images).to_csv(os.path.join(consolidated_dir, "failed_images.csv"), index=False)
    
    # Extract spatial coordinates
    all_spatial = pd.DataFrame()
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
    
    # Cell segmentation by dilation
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
    
    # Measure protein intensities
    if protein_dirs:
        from src.features.intensity_features import measure_intensity_features
        
        for protein_dir in protein_dirs:
            protein_dir = os.path.normpath(protein_dir)
            if not os.path.exists(protein_dir):
                logger.warning(f"Protein directory not found: {protein_dir}")
                continue
            
            protein_name = os.path.basename(protein_dir)
            protein_output_dir = os.path.join(output_dir, f"{protein_name}_levels")
            Path(protein_output_dir).mkdir(parents=True, exist_ok=True)
            
            # Use cell labels if available, otherwise nuclear labels
            seg_dir = cell_labels_dir if cell_segmentation else labels_dir
            seg_dir = os.path.normpath(seg_dir)
            
            logger.info(f"Measuring {protein_name} intensities...")
            seg_files = sorted(glob(os.path.join(seg_dir, "*.tif")))
            prot_files = sorted(glob(os.path.join(protein_dir, "*.tif")))
            
            protein_features = pd.DataFrame()
            for i in tqdm(range(min(len(seg_files), len(prot_files))), desc=f"{protein_name}"):
                labelled_image = imread(seg_files[i])
                protein_image = imread(prot_files[i])
                
                props = measure.regionprops(labelled_image, protein_image)
                features = pd.DataFrame()
                
                for prop in props:
                    try:
                        # Use full intensity feature extraction (like notebook)
                        intensity_feat = measure_intensity_features(
                            regionmask=prop.image,
                            intensity=prop.intensity_image,
                            measure_int_dist=True,
                            measure_hc_ec_ratios=False  # Don't need HC/EC ratios for protein levels
                        )
                        
                        # Add label
                        intensity_feat['label'] = prop.label
                        features = pd.concat([features, intensity_feat], ignore_index=True)
                    except Exception as e:
                        # Fallback to basic stats if full extraction fails
                        logger.debug(f"Full intensity extraction failed for label {prop.label}, using basic stats: {e}")
                        feat_row = pd.DataFrame([{
                            'label': prop.label,
                            'int_mean': prop.mean_intensity,
                            'int_max': prop.max_intensity,
                            'int_min': prop.min_intensity
                        }])
                        features = pd.concat([features, feat_row], ignore_index=True)
                
                img_name = os.path.splitext(os.path.basename(seg_files[i]))[0]
                features["image"] = img_name
                features["nuc_id"] = features["image"].astype(str) + "_" + features["label"].astype(str)
                features.to_csv(os.path.join(protein_output_dir, f"{img_name}.csv"), index=False)
                protein_features = pd.concat([protein_features, features], ignore_index=True)
            
            # Save consolidated protein features
            protein_features.to_csv(os.path.join(consolidated_dir, f"{protein_name}_levels.csv"), index=False)
    
    # Measure GC mask intensities (to determine cells inside/outside germinal center)
    if gc_mask_dir and os.path.exists(gc_mask_dir):
        from src.features.intensity_features import measure_intensity_features
        
        gc_mask_dir = os.path.normpath(gc_mask_dir)
        gc_output_dir = os.path.join(output_dir, "gc_levels")
        Path(gc_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use cell labels if available, otherwise nuclear labels
        seg_dir = cell_labels_dir if cell_segmentation else labels_dir
        seg_dir = os.path.normpath(seg_dir)
        
        logger.info("Measuring germinal center mask intensities...")
        seg_files = sorted(glob(os.path.join(seg_dir, "*.tif")))
        gc_mask_files = sorted(glob(os.path.join(gc_mask_dir, "*.tif")))
        
        gc_features = pd.DataFrame()
        for i in tqdm(range(min(len(seg_files), len(gc_mask_files))), desc="GC mask"):
            labelled_image = imread(seg_files[i])
            gc_mask_image = imread(gc_mask_files[i])
            
            props = measure.regionprops(labelled_image, gc_mask_image)
            features = pd.DataFrame()
            
            for prop in props:
                try:
                    # Use full intensity feature extraction (like notebook)
                    intensity_feat = measure_intensity_features(
                        regionmask=prop.image,
                        intensity=prop.intensity_image,
                        measure_int_dist=True,
                        measure_hc_ec_ratios=False  # Don't need HC/EC ratios for GC mask
                    )
                    
                    # Add label
                    intensity_feat['label'] = prop.label
                    features = pd.concat([features, intensity_feat], ignore_index=True)
                except Exception as e:
                    # Fallback to basic stats if full extraction fails
                    logger.debug(f"Full intensity extraction failed for label {prop.label}, using basic stats: {e}")
                    feat_row = pd.DataFrame([{
                        'label': prop.label,
                        'int_mean': prop.mean_intensity,
                        'int_max': prop.max_intensity,
                        'int_min': prop.min_intensity
                    }])
                    features = pd.concat([features, feat_row], ignore_index=True)
            
            img_name = os.path.splitext(os.path.basename(seg_files[i]))[0]
            features["image"] = img_name
            features["nuc_id"] = features["image"].astype(str) + "_" + features["label"].astype(str)
            features.to_csv(os.path.join(gc_output_dir, f"{img_name}.csv"), index=False)
            gc_features = pd.concat([gc_features, features], ignore_index=True)
        
        # Save consolidated GC features
        gc_features.to_csv(os.path.join(consolidated_dir, "gc_levels.csv"), index=False)
        logger.info(f"GC mask intensities saved to {os.path.join(consolidated_dir, 'gc_levels.csv')}")
    
    # Add nuc_id to features
    all_features["nuc_id"] = all_features["image"].astype(str) + "_" + all_features["label"].astype(str)
    
    # Save consolidated outputs
    all_features.to_csv(os.path.join(consolidated_dir, "nuc_features.csv"), index=False)
    if extract_spatial:
        all_spatial.to_csv(os.path.join(consolidated_dir, "spatial_coordinates.csv"), index=False)
    
    # Save final state
    if state:
        state.save()
    
    logger.info(f"Feature extraction complete. Results saved to {consolidated_dir}")
    logger.info(f"  - {len(all_features)} nuclei processed")
    logger.info(f"  - {len(all_features.columns)} features extracted")

