"""Nuclear segmentation command using StarDist"""
import gc
import logging
import os
from glob import glob
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import imageio as imio
from tifffile import imsave

if TYPE_CHECKING:
    from cli.state import PipelineState

logger = logging.getLogger(__name__)


def segment_nuclei(
    input_dir: str,
    output_labels_dir: str,
    output_rois_dir: Optional[str] = None,
    prob_thresh: float = 0.43,
    use_pretrained: bool = True,
    model_dir: str = "models",
    model_name: str = "DAPI_segmentation",
    normalize_quants: tuple = (1, 99.8),
    state: Optional["PipelineState"] = None,
    resume: bool = False
):
    """Segment nuclei using StarDist 2D model
    
    Args:
        input_dir: Directory with DAPI images
        output_labels_dir: Directory to save label images
        output_rois_dir: Optional directory to save ImageJ ROIs
        prob_thresh: Probability threshold for detection
        use_pretrained: Use pretrained model (2D_versatile_fluo)
        model_dir: Directory with custom model (if not using pretrained)
        model_name: Name of custom model
        normalize_quants: Quantiles for image normalization
        state: Pipeline state manager for tracking progress
        resume: Whether to skip already processed files
    """
    # Import here to avoid slow startup when not using this command
    from csbdeep.utils import normalize
    from stardist import export_imagej_rois
    from stardist.models import StarDist2D
    
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)
    if output_rois_dir:
        Path(output_rois_dir).mkdir(parents=True, exist_ok=True)
    
    # Normalize path for cross-platform compatibility
    input_dir = os.path.normpath(input_dir)
    
    # Load model
    if use_pretrained:
        logger.info("Loading pretrained StarDist model (2D_versatile_fluo)...")
        model = StarDist2D.from_pretrained("2D_versatile_fluo")
    else:
        logger.info(f"Loading custom model: {model_name} from {model_dir}...")
        model = StarDist2D(None, name=model_name, basedir=model_dir)
    
    # Get all images using os.path.join for cross-platform glob
    all_images = sorted(glob(os.path.join(input_dir, "*.tif")))
    if not all_images:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")
    
    # Update state with total files
    if state:
        state.state['steps']['segment']['total_files'] = len(all_images)
        state.save()
    
    # Filter already processed files if resuming
    images_to_process = all_images
    if resume and state:
        unprocessed_filenames = state.get_unprocessed_files('segment', [os.path.basename(f) for f in all_images])
        # Reconstruct full paths for unprocessed files
        images_to_process = [f for f in all_images if os.path.basename(f) in unprocessed_filenames]
        if len(images_to_process) < len(all_images):
            logger.info(f"Resuming: {len(all_images) - len(images_to_process)} images already processed")
    
    logger.info(f"Found {len(all_images)} images, {len(images_to_process)} to process")
    
    for i, image_path in enumerate(images_to_process):
        filename = os.path.basename(image_path)
        processed_count = len(all_images) - len(images_to_process) + i + 1
        logger.info(f"  [{processed_count}/{len(all_images)}] Segmenting {filename}...")
        
        # Read and normalize image
        X = imio.imread(image_path)
        X = normalize(X, normalize_quants[0], normalize_quants[1], axis=(0, 1))
        
        # Predict instances
        labels, polygons = model.predict_instances(
            X,
            n_tiles=model._guess_n_tiles(X),
            prob_thresh=prob_thresh
        )
        
        # Save label image
        output_path = os.path.join(output_labels_dir, filename)
        imsave(output_path, labels)
        
        # Save ImageJ ROIs if requested
        if output_rois_dir:
            roi_name = os.path.splitext(filename)[0] + ".zip"
            roi_path = os.path.join(output_rois_dir, roi_name)
            export_imagej_rois(roi_path, polygons["coord"])
        
        # Mark file as processed
        if state:
            state.mark_file_processed('segment', filename)
        
        # Clear memory
        del X, labels, polygons
        gc.collect()
    
    # Save final state
    if state:
        state.save()
    
    logger.info(f"Segmentation complete. Labels saved to {output_labels_dir}")

