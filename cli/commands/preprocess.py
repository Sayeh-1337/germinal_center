"""Image preprocessing command"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import imageio as imio
import numpy as np
from tifffile import imsave

logger = logging.getLogger(__name__)


def get_file_list(root_dir: str, file_type_filter: str = ".tif") -> List[str]:
    """Get sorted list of files in directory"""
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    files = []
    for f in os.listdir(root_dir):
        if file_type_filter is None or f.endswith(file_type_filter):
            files.append(os.path.join(root_dir, f))
    return sorted(files)


def extract_channel_save_image(image_dir: str, output_dir: str, channel: int):
    """Extract a specific channel from multi-channel images and save
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save extracted channel images
        channel: Channel index to extract (1-based, from end)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    img_locs = get_file_list(image_dir)
    
    for image_loc in img_locs:
        X = imio.imread(image_loc)
        X = X[:, :, -channel]  # Extract channel (1-based from end)
        output_path = os.path.join(output_dir, os.path.basename(image_loc))
        imsave(output_path, X)


def quantile_normalize_and_save_images(
    image_dir: str,
    output_dir: str,
    mask_dir: Optional[str] = None,
    quantiles: List[float] = None
):
    """Quantile normalize images and save
    
    Args:
        image_dir: Directory containing input images
        output_dir: Directory to save normalized images
        mask_dir: Optional directory with mask images
        quantiles: Quantile values [low, high] for normalization
    """
    if quantiles is None:
        quantiles = [0.01, 0.998]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_locs = get_file_list(image_dir)
    
    for loc in image_locs:
        img = imio.imread(loc)
        filename = os.path.basename(loc)
        
        if mask_dir is not None:
            mask_path = os.path.join(mask_dir, filename)
            if os.path.exists(mask_path):
                mask = imio.imread(mask_path)
                masked_img = np.ma.array(img, mask=~(mask > 0)).astype(float)
                low = np.quantile(masked_img.compressed(), quantiles[0])
                high = np.quantile(masked_img.compressed(), quantiles[1])
            else:
                low = np.quantile(img, quantiles[0])
                high = np.quantile(img, quantiles[1])
        else:
            low = np.quantile(img, quantiles[0])
            high = np.quantile(img, quantiles[1])
        
        scaled_img = (img.astype(float) - low) / (high - low)
        scaled_img = np.clip(scaled_img * 255, 0, 255).astype(np.uint8)
        imsave(os.path.join(output_dir, filename), scaled_img)


def preprocess_images(
    input_dir: str,
    output_dir: str,
    channels: Dict[str, int] = None,
    quantiles: List[float] = None,
    mask_dir: Optional[str] = None,
    skip_normalize: bool = False
):
    """Preprocess images: extract channels and normalize
    
    Args:
        input_dir: Input directory with raw merged images
        output_dir: Output directory for processed images
        channels: Dictionary mapping channel names to indices
        quantiles: Quantile values for normalization
        mask_dir: Optional mask directory for normalization
        skip_normalize: Skip quantile normalization step
    """
    if channels is None:
        channels = {'dapi': 1, 'cd3': 2, 'aicda': 3}
    
    if quantiles is None:
        quantiles = [0.01, 0.998]
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if input contains merged images or already separated channels
    sample_files = get_file_list(input_dir)
    if not sample_files:
        raise FileNotFoundError(f"No .tif files found in {input_dir}")
    
    sample_img = imio.imread(sample_files[0])
    is_multichannel = len(sample_img.shape) == 3 and sample_img.shape[2] >= max(channels.values())
    
    if is_multichannel:
        # Extract channels from merged images
        logger.info(f"Extracting {len(channels)} channels from merged images...")
        for name, channel_num in channels.items():
            channel_output = os.path.join(output_dir, name)
            extract_channel_save_image(input_dir, channel_output, channel_num)
            logger.info(f"  Extracted {name} channel (index {channel_num})")
    else:
        # Images are already single channel, copy to dapi folder
        logger.info("Input images are single channel, copying to output...")
        dapi_output = os.path.join(output_dir, 'dapi')
        Path(dapi_output).mkdir(parents=True, exist_ok=True)
        for src in sample_files:
            dst = os.path.join(dapi_output, os.path.basename(src))
            img = imio.imread(src)
            imsave(dst, img)
    
    # Quantile normalize DAPI
    if not skip_normalize:
        dapi_dir = os.path.join(output_dir, 'dapi')
        if os.path.exists(dapi_dir):
            dapi_scaled_dir = os.path.join(output_dir, 'dapi_scaled')
            logger.info(f"Quantile normalizing DAPI images (q={quantiles})...")
            quantile_normalize_and_save_images(
                image_dir=dapi_dir,
                output_dir=dapi_scaled_dir,
                mask_dir=mask_dir,
                quantiles=quantiles
            )
            logger.info(f"  Saved normalized images to {dapi_scaled_dir}")
    
    logger.info("Preprocessing complete")

