#!/usr/bin/env python
"""
Germinal Center Analysis CLI
Main entry point for the command-line interface
"""
import argparse
import sys
import logging
from pathlib import Path

from cli.commands.preprocess import preprocess_images
from cli.commands.segment import segment_nuclei
from cli.commands.extract import extract_features
from cli.commands.extract_enhanced import extract_enhanced_features
from cli.commands.analyze import run_analysis
from cli.commands.pipeline import run_full_pipeline
from cli.config import load_config, save_config, get_default_config
from cli.state import get_state

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_parser():
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog='gc-pipeline',
        description="Germinal Center Chromatin Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with config file
  gc-pipeline pipeline --config configs/dataset1_config.yaml
  
  # Run specific steps only
  gc-pipeline pipeline --config configs/dataset1_config.yaml --steps preprocess segment
  
  # Run individual commands
  gc-pipeline preprocess --input data/images/raw --output data/images/processed
  gc-pipeline segment --input data/images/processed/dapi_scaled --output-labels data/images/segmented
  gc-pipeline extract --raw-images data/images/processed/dapi_scaled --labels data/images/segmented --output data/features
  gc-pipeline analyze --features data/features --output data/analysis
  
  # Generate default config file
  gc-pipeline init --name my_dataset --output configs/my_config.yaml
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands')
    
    # Init command - generate config file
    init_parser = subparsers.add_parser('init', help='Generate default configuration file')
    init_parser.add_argument(
        '--name', '-n',
        type=str,
        default='dataset1',
        help='Dataset name (default: dataset1)'
    )
    init_parser.add_argument(
        '--output', '-o',
        type=str,
        default='configs/config.yaml',
        help='Output path for config file'
    )
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline')
    pipeline_parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    pipeline_parser.add_argument(
        '--steps',
        nargs='+',
        choices=['preprocess', 'segment', 'extract', 'analyze'],
        help='Specific steps to run (default: all)'
    )
    pipeline_parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory from config'
    )
    pipeline_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last interrupted state (skips completed steps and files)'
    )
    pipeline_parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset pipeline state and start fresh (use with --resume to clear state)'
    )
    pipeline_parser.add_argument(
        '--status',
        action='store_true',
        help='Show pipeline progress status and exit'
    )
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess images')
    preprocess_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory with raw images'
    )
    preprocess_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for processed images'
    )
    preprocess_parser.add_argument(
        '--channels',
        nargs='+',
        type=str,
        default=['dapi:1', 'cd3:2', 'aicda:3'],
        help='Channels to extract as name:index pairs (default: dapi:1 cd3:2 aicda:3)'
    )
    preprocess_parser.add_argument(
        '--quantiles',
        nargs=2,
        type=float,
        default=[0.01, 0.998],
        help='Quantiles for normalization (default: 0.01 0.998)'
    )
    preprocess_parser.add_argument(
        '--mask-dir',
        type=str,
        help='Directory with mask images for normalization'
    )
    preprocess_parser.add_argument(
        '--skip-normalize',
        action='store_true',
        help='Skip quantile normalization step'
    )
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment nuclei using StarDist')
    segment_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input directory with DAPI images'
    )
    segment_parser.add_argument(
        '--output-labels', '-ol',
        type=str,
        required=True,
        help='Output directory for label images'
    )
    segment_parser.add_argument(
        '--output-rois', '-or',
        type=str,
        help='Output directory for ImageJ ROIs (optional)'
    )
    segment_parser.add_argument(
        '--prob-thresh',
        type=float,
        default=0.43,
        help='Probability threshold for StarDist (default: 0.43)'
    )
    segment_parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Use custom model instead of pretrained'
    )
    segment_parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing custom model (when --no-pretrained is set)'
    )
    segment_parser.add_argument(
        '--model-name',
        type=str,
        default='DAPI_segmentation',
        help='Name of custom model (when --no-pretrained is set)'
    )
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract chrometric features')
    extract_parser.add_argument(
        '--raw-images', '-r',
        type=str,
        required=True,
        help='Directory with raw DAPI images'
    )
    extract_parser.add_argument(
        '--labels', '-l',
        type=str,
        required=True,
        help='Directory with segmented label images'
    )
    extract_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for features'
    )
    extract_parser.add_argument(
        '--proteins',
        nargs='+',
        type=str,
        help='Directories with protein images for intensity measurement (e.g., path/to/cd3 path/to/aicda)'
    )
    extract_parser.add_argument(
        '--cell-segmentation',
        action='store_true',
        help='Perform cell segmentation by nuclear boundary dilation'
    )
    extract_parser.add_argument(
        '--dilation-radius',
        type=int,
        default=10,
        help='Radius for cell segmentation dilation in pixels (default: 10)'
    )
    extract_parser.add_argument(
        '--extract-spatial',
        action='store_true',
        default=True,
        help='Extract spatial coordinates (default: True)'
    )
    
    # Extract-enhanced command (includes advanced features)
    extract_enhanced_parser = subparsers.add_parser(
        'extract-enhanced', 
        help='Extract enhanced chrometric features (includes multi-scale, cell cycle, spatial graph)'
    )
    extract_enhanced_parser.add_argument(
        '--raw-images', '-r',
        type=str,
        required=True,
        help='Directory with raw DAPI images'
    )
    extract_enhanced_parser.add_argument(
        '--labels', '-l',
        type=str,
        required=True,
        help='Directory with segmented label images'
    )
    extract_enhanced_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for features'
    )
    extract_enhanced_parser.add_argument(
        '--proteins',
        nargs='+',
        type=str,
        help='Directories with protein images for intensity measurement'
    )
    extract_enhanced_parser.add_argument(
        '--cell-segmentation',
        action='store_true',
        help='Perform cell segmentation by nuclear boundary dilation'
    )
    extract_enhanced_parser.add_argument(
        '--dilation-radius',
        type=int,
        default=10,
        help='Radius for cell segmentation dilation in pixels (default: 10)'
    )
    # Enhanced feature options
    extract_enhanced_parser.add_argument(
        '--no-multiscale',
        action='store_true',
        help='Skip multi-scale wavelet and fractal features'
    )
    extract_enhanced_parser.add_argument(
        '--no-cell-cycle',
        action='store_true',
        help='Skip cell cycle state inference'
    )
    extract_enhanced_parser.add_argument(
        '--no-spatial-graph',
        action='store_true',
        help='Skip spatial graph-based features (centrality, Voronoi)'
    )
    extract_enhanced_parser.add_argument(
        '--no-relative',
        action='store_true',
        help='Skip relative and interaction features'
    )
    extract_enhanced_parser.add_argument(
        '--k-neighbors',
        type=int,
        default=10,
        help='Number of neighbors for spatial analysis (default: 10)'
    )
    extract_enhanced_parser.add_argument(
        '--wavelet-levels',
        type=int,
        default=3,
        help='Number of wavelet decomposition levels (default: 3)'
    )
    extract_enhanced_parser.add_argument(
        '--density-radii',
        nargs='+',
        type=float,
        default=[25, 50, 100],
        help='Radii for local density computation in pixels (default: 25 50 100)'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run analysis on extracted features')
    analyze_parser.add_argument(
        '--features', '-f',
        type=str,
        required=True,
        help='Directory with feature CSV files'
    )
    analyze_parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for analysis results'
    )
    analyze_parser.add_argument(
        '--analysis-type',
        choices=[
            'cell_type_detection', 'cell_type', 'classification',
            'tcell_interaction', 'tcell', 'boundary', 'dz_lz_boundary',
            'correlation', 'umap', 'visualization', 'markers', 'differential',
            'enhanced_visualization', 'enhanced_features', 'all'
        ],
        nargs='+',
        default=['cell_type'],
        help='Types of analysis to run (default: cell_type). Options: '
             'cell_type_detection (GMM marker detection), '
             'cell_type/classification (DZ vs LZ classification), '
             'tcell_interaction/tcell (T-cell influence zones), '
             'boundary/dz_lz_boundary (DZ/LZ boundary distance), '
             'correlation (feature-metadata correlation), '
             'umap/visualization (UMAP embedding & clustering), '
             'markers/differential (marker feature detection), '
             'enhanced_visualization/enhanced_features (enhanced features visualization), '
             'all (run all analyses)'
    )
    analyze_parser.add_argument(
        '--metadata',
        type=str,
        help='Path to metadata CSV file (for correlation analysis)'
    )
    analyze_parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.8,
        help='Pearson correlation threshold for feature filtering (default: 0.8)'
    )
    analyze_parser.add_argument(
        '--random-seed',
        type=int,
        default=1234,
        help='Random seed for reproducibility (default: 1234)'
    )
    # Spatial analysis parameters
    analyze_parser.add_argument(
        '--pixel-size',
        type=float,
        default=0.3225,
        help='Pixel size in microns for spatial analyses (default: 0.3225)'
    )
    analyze_parser.add_argument(
        '--contact-radius',
        type=float,
        default=15.0,
        help='T-cell physical contact radius in microns (default: 15.0)'
    )
    analyze_parser.add_argument(
        '--signaling-radius',
        type=float,
        default=30.0,
        help='T-cell signaling radius in microns (default: 30.0)'
    )
    analyze_parser.add_argument(
        '--border-threshold',
        type=float,
        default=0.4,
        help='Threshold for DZ/LZ border proximity classification (default: 0.4)'
    )
    # Visualization and statistical options
    analyze_parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating visualization plots'
    )
    analyze_parser.add_argument(
        '--n-permutations',
        type=int,
        default=10000,
        help='Number of permutations for statistical tests (default: 10000)'
    )
    
    return parser


def parse_channels(channel_args):
    """Parse channel arguments from command line
    
    Args:
        channel_args: List of 'name:index' strings
        
    Returns:
        Dictionary mapping channel names to indices
    """
    channels = {}
    for ch in channel_args:
        if ':' in ch:
            name, idx = ch.split(':')
            channels[name] = int(idx)
        else:
            raise ValueError(f"Invalid channel format: {ch}. Use 'name:index' format.")
    return channels


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'init':
            logger.info(f"Generating default config for '{args.name}'...")
            config = get_default_config(args.name)
            save_config(config, args.output)
            logger.info(f"Config saved to: {args.output}")
        
        elif args.command == 'pipeline':
            config = load_config(args.config)
            if args.output_dir:
                config['output_dir'] = args.output_dir
            
            output_dir = config['output_dir']
            state = get_state(output_dir)
            
            # Handle --status flag
            if args.status:
                print(state.get_progress_summary())
                sys.exit(0)
            
            # Handle --reset flag
            if args.reset:
                state.reset()
                logger.info("Pipeline state has been reset")
                if not args.resume:
                    # Just reset and exit if not also resuming
                    print("Pipeline state reset. Run again without --reset to start fresh.")
                    sys.exit(0)
            
            steps = args.steps if args.steps else ['preprocess', 'segment', 'extract', 'analyze']
            
            # Handle --resume flag
            if args.resume:
                logger.info("Resume mode enabled - checking for previous progress...")
                print(state.get_progress_summary())
                resume_steps = state.get_resume_steps(steps)
                if not resume_steps:
                    logger.info("All requested steps are already completed!")
                    print("\nAll requested steps are already completed. Use --reset to start fresh.")
                    sys.exit(0)
                logger.info(f"Resuming with steps: {', '.join(resume_steps)}")
                steps = resume_steps
            
            run_full_pipeline(config, steps, state=state, resume=args.resume)
        
        elif args.command == 'preprocess':
            channels = parse_channels(args.channels)
            preprocess_images(
                input_dir=args.input,
                output_dir=args.output,
                channels=channels,
                quantiles=args.quantiles,
                mask_dir=args.mask_dir,
                skip_normalize=args.skip_normalize
            )
        
        elif args.command == 'segment':
            segment_nuclei(
                input_dir=args.input,
                output_labels_dir=args.output_labels,
                output_rois_dir=args.output_rois,
                prob_thresh=args.prob_thresh,
                use_pretrained=not args.no_pretrained,
                model_dir=args.model_dir,
                model_name=args.model_name
            )
        
        elif args.command == 'extract':
            extract_features(
                raw_images_dir=args.raw_images,
                labels_dir=args.labels,
                output_dir=args.output,
                protein_dirs=args.proteins,
                cell_segmentation=args.cell_segmentation,
                dilation_radius=args.dilation_radius,
                extract_spatial=args.extract_spatial
            )
        
        elif args.command == 'extract-enhanced':
            extract_enhanced_features(
                raw_images_dir=args.raw_images,
                labels_dir=args.labels,
                output_dir=args.output,
                protein_dirs=args.proteins,
                cell_segmentation=args.cell_segmentation,
                dilation_radius=args.dilation_radius,
                extract_spatial=True,
                extract_multiscale=not args.no_multiscale,
                extract_cell_cycle=not args.no_cell_cycle,
                extract_spatial_graph=not args.no_spatial_graph,
                extract_relative=not args.no_relative,
                k_neighbors=args.k_neighbors,
                wavelet_levels=args.wavelet_levels,
                density_radii=args.density_radii
            )
        
        elif args.command == 'analyze':
            analysis_types = args.analysis_type
            if 'all' in analysis_types:
                analysis_types = [
                    'cell_type_detection', 'cell_type', 'tcell_interaction',
                    'boundary', 'correlation', 'umap', 'markers'
                ]
            
            run_analysis(
                features_dir=args.features,
                output_dir=args.output,
                analysis_types=analysis_types,
                metadata_path=args.metadata,
                correlation_threshold=args.correlation_threshold,
                random_seed=args.random_seed,
                pixel_size=args.pixel_size,
                contact_radius=args.contact_radius,
                signaling_radius=args.signaling_radius,
                border_threshold=args.border_threshold,
                generate_plots=not args.no_plots,
                n_permutations=args.n_permutations
            )
        
        print(f"\n[OK] {args.command} completed successfully")
        
    except Exception as e:
        print(f"\n[ERROR] {args.command}: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

