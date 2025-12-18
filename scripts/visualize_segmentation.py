#!/usr/bin/env python
"""
Visualize segmentation results overlaid on original images.

Usage:
    python scripts/visualize_segmentation.py --image-dir data/dataset1/processed/dapi_scaled --label-dir data/dataset1/processed/segmented_nucleus
    python scripts/visualize_segmentation.py --image-dir data/dataset1/processed/dapi_scaled --label-dir data/dataset1/processed/segmented_nucleus --output-dir outputs/segmentation_viz
    python scripts/visualize_segmentation.py --image-dir data/dataset1/processed/dapi_scaled --label-dir data/dataset1/processed/segmented_nucleus --interactive
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.segmentation import find_boundaries
from skimage.color import label2rgb
from tqdm import tqdm


def create_random_colormap(n_labels):
    """Create a random colormap for label visualization"""
    np.random.seed(42)
    colors = np.random.rand(n_labels, 3)
    colors[0] = [0, 0, 0]  # Background is black
    return ListedColormap(colors)


def overlay_segmentation(image, labels, alpha=0.3, boundary_color=[1, 0, 0], show_boundaries=True):
    """
    Create an overlay of segmentation on the original image.
    
    Args:
        image: 2D grayscale image
        labels: 2D label image (segmentation mask)
        alpha: Transparency of the overlay
        boundary_color: RGB color for boundaries
        show_boundaries: Whether to show cell boundaries
        
    Returns:
        RGB overlay image
    """
    # Normalize image to 0-1
    img_norm = image.astype(float)
    if img_norm.max() > 0:
        img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min())
    
    # Convert to RGB
    img_rgb = np.stack([img_norm, img_norm, img_norm], axis=-1)
    
    # Create colored labels
    label_rgb = label2rgb(labels, bg_label=0, bg_color=(0, 0, 0))
    
    # Blend
    overlay = (1 - alpha) * img_rgb + alpha * label_rgb
    
    # Add boundaries
    if show_boundaries:
        boundaries = find_boundaries(labels, mode='outer')
        overlay[boundaries] = boundary_color
    
    return np.clip(overlay, 0, 1)


def visualize_single_image(image_path, label_path, output_path=None, show=True, 
                           figsize=(16, 8), alpha=0.3):
    """
    Visualize a single image with its segmentation.
    
    Args:
        image_path: Path to the original image
        label_path: Path to the label/segmentation image
        output_path: Optional path to save the figure
        show: Whether to display the figure
        figsize: Figure size
        alpha: Overlay transparency
    """
    # Load images
    image = imread(image_path)
    labels = imread(label_path)
    
    # Handle multi-channel images
    if image.ndim == 3:
        image = image[0] if image.shape[0] < image.shape[-1] else image[:, :, 0]
    
    # Create overlay
    overlay = overlay_segmentation(image, labels, alpha=alpha)
    
    # Count nuclei
    n_nuclei = len(np.unique(labels)) - 1  # Exclude background
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f'Original DAPI\n{os.path.basename(image_path)}')
    axes[0].axis('off')
    
    # Labels only
    axes[1].imshow(label2rgb(labels, bg_label=0))
    axes[1].set_title(f'Segmentation Labels\n({n_nuclei:,} nuclei)')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (boundaries in red)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_directory(image_dir, label_dir, output_dir=None, show=False, 
                        max_images=None, alpha=0.3):
    """
    Visualize all images in a directory with their segmentations.
    
    Args:
        image_dir: Directory containing original images
        label_dir: Directory containing label images
        output_dir: Directory to save visualizations
        show: Whether to display each figure
        max_images: Maximum number of images to process
        alpha: Overlay transparency
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    # Find matching files
    image_files = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))
    
    if max_images:
        image_files = image_files[:max_images]
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(image_files)} images to visualize")
    
    for image_path in tqdm(image_files, desc="Creating visualizations"):
        # Find corresponding label file
        label_path = label_dir / image_path.name
        
        if not label_path.exists():
            # Try without extension matching
            label_candidates = list(label_dir.glob(f"{image_path.stem}*"))
            if label_candidates:
                label_path = label_candidates[0]
            else:
                print(f"Warning: No label found for {image_path.name}")
                continue
        
        output_path = None
        if output_dir:
            output_path = output_dir / f"{image_path.stem}_segmentation.png"
        
        try:
            visualize_single_image(
                str(image_path), 
                str(label_path), 
                output_path=str(output_path) if output_path else None,
                show=show,
                alpha=alpha
            )
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")


def interactive_viewer(image_dir, label_dir):
    """
    Launch an interactive viewer for browsing segmentations.
    
    Uses matplotlib's interactive mode to browse through images.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    image_files = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.tiff")))
    
    if not image_files:
        print("No images found!")
        return
    
    print(f"\nInteractive Segmentation Viewer")
    print(f"================================")
    print(f"Found {len(image_files)} images")
    print(f"\nControls:")
    print(f"  'n' or Right Arrow: Next image")
    print(f"  'p' or Left Arrow: Previous image")
    print(f"  'q': Quit")
    print(f"  '+'/'-': Adjust overlay transparency")
    print(f"  'b': Toggle boundaries")
    print(f"  's': Save current view")
    print()
    
    current_idx = [0]  # Use list to allow modification in nested function
    alpha = [0.3]
    show_boundaries = [True]
    
    def load_current():
        image_path = image_files[current_idx[0]]
        label_path = label_dir / image_path.name
        
        if not label_path.exists():
            label_candidates = list(label_dir.glob(f"{image_path.stem}*"))
            if label_candidates:
                label_path = label_candidates[0]
            else:
                return None, None, image_path.name
        
        image = imread(str(image_path))
        labels = imread(str(label_path))
        
        if image.ndim == 3:
            image = image[0] if image.shape[0] < image.shape[-1] else image[:, :, 0]
        
        return image, labels, image_path.name
    
    def update_display():
        image, labels, name = load_current()
        
        if image is None:
            ax.set_title(f"No segmentation for: {name}")
            return
        
        overlay = overlay_segmentation(
            image, labels, 
            alpha=alpha[0], 
            show_boundaries=show_boundaries[0]
        )
        
        n_nuclei = len(np.unique(labels)) - 1
        
        ax.clear()
        ax.imshow(overlay)
        ax.set_title(f"{name}\n{n_nuclei:,} nuclei | Image {current_idx[0]+1}/{len(image_files)} | Alpha: {alpha[0]:.1f}")
        ax.axis('off')
        fig.canvas.draw()
    
    def on_key(event):
        if event.key in ['n', 'right']:
            current_idx[0] = (current_idx[0] + 1) % len(image_files)
            update_display()
        elif event.key in ['p', 'left']:
            current_idx[0] = (current_idx[0] - 1) % len(image_files)
            update_display()
        elif event.key == '+' or event.key == '=':
            alpha[0] = min(1.0, alpha[0] + 0.1)
            update_display()
        elif event.key == '-':
            alpha[0] = max(0.0, alpha[0] - 0.1)
            update_display()
        elif event.key == 'b':
            show_boundaries[0] = not show_boundaries[0]
            update_display()
        elif event.key == 's':
            output_path = f"segmentation_{image_files[current_idx[0]].stem}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        elif event.key == 'q':
            plt.close()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_display()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize segmentation results overlaid on original images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all images and save to output directory
  python scripts/visualize_segmentation.py \\
      --image-dir data/dataset1/processed/dapi_scaled \\
      --label-dir data/dataset1/processed/segmented_nucleus \\
      --output-dir outputs/segmentation_viz

  # Interactive viewer
  python scripts/visualize_segmentation.py \\
      --image-dir data/dataset1/processed/dapi_scaled \\
      --label-dir data/dataset1/processed/segmented_nucleus \\
      --interactive

  # Visualize a single image
  python scripts/visualize_segmentation.py \\
      --image data/dataset1/processed/dapi_scaled/1.tif \\
      --labels data/dataset1/processed/segmented_nucleus/1.tif
        """
    )
    
    # Directory mode
    parser.add_argument('--image-dir', type=str, 
                        help='Directory containing original images')
    parser.add_argument('--label-dir', type=str,
                        help='Directory containing segmentation labels')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to save visualizations')
    
    # Single image mode
    parser.add_argument('--image', type=str,
                        help='Path to a single image file')
    parser.add_argument('--labels', type=str,
                        help='Path to corresponding label file')
    
    # Options
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Launch interactive viewer')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Overlay transparency (0-1, default: 0.3)')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process')
    parser.add_argument('--show', action='store_true',
                        help='Display each image (not recommended for many images)')
    
    args = parser.parse_args()
    
    # Single image mode
    if args.image and args.labels:
        visualize_single_image(
            args.image, 
            args.labels, 
            show=True,
            alpha=args.alpha
        )
        return
    
    # Directory mode
    if not args.image_dir or not args.label_dir:
        parser.print_help()
        print("\nError: Please provide --image-dir and --label-dir, or --image and --labels")
        sys.exit(1)
    
    if args.interactive:
        interactive_viewer(args.image_dir, args.label_dir)
    else:
        if not args.output_dir and not args.show:
            print("Note: No --output-dir specified. Use --show to display or --output-dir to save.")
            args.show = True
        
        visualize_directory(
            args.image_dir,
            args.label_dir,
            output_dir=args.output_dir,
            show=args.show,
            max_images=args.max_images,
            alpha=args.alpha
        )


if __name__ == "__main__":
    main()

