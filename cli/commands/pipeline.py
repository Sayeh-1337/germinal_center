"""Full pipeline execution"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from cli.commands.preprocess import preprocess_images
from cli.commands.segment import segment_nuclei
from cli.commands.extract import extract_features
from cli.commands.analyze import run_analysis
from cli.state import PipelineState

logger = logging.getLogger(__name__)


def run_full_pipeline(
    config: Dict[str, Any], 
    steps: List[str] = None,
    state: Optional[PipelineState] = None,
    resume: bool = False
):
    """Run the full analysis pipeline
    
    Args:
        config: Configuration dictionary
        steps: List of steps to run (default: all)
        state: Pipeline state manager for tracking progress
        resume: Whether we're resuming from a previous run
    """
    if steps is None:
        steps = ['preprocess', 'segment', 'extract', 'analyze']
    
    logger.info(f"=" * 60)
    logger.info(f"Starting Germinal Center Analysis Pipeline")
    logger.info(f"Dataset: {config.get('dataset_name', 'unknown')}")
    logger.info(f"Steps: {', '.join(steps)}")
    if resume:
        logger.info(f"Mode: RESUME (skipping completed files)")
    logger.info(f"=" * 60)
    
    output_dir = config['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create state manager if not provided
    if state is None:
        from cli.state import get_state
        state = get_state(output_dir)
    
    # Step 1: Preprocessing
    if 'preprocess' in steps:
        logger.info("\n" + "=" * 40)
        logger.info("Step 1/4: Preprocessing images")
        logger.info("=" * 40)
        
        try:
            state.start_step('preprocess')
            preprocess_config = config.get('preprocessing', {})
            channels = preprocess_config.get('channels', {'dapi': 1, 'cd3': 2, 'aicda': 3})
            quant_config = preprocess_config.get('quantile_normalization', {})
            
            preprocess_images(
                input_dir=config.get('input_dir', preprocess_config.get('merged_image_dir', '')),
                output_dir=output_dir,
                channels=channels,
                quantiles=quant_config.get('quantiles', [0.01, 0.998]),
                mask_dir=quant_config.get('mask_dir'),
                skip_normalize=not quant_config.get('enabled', True)
            )
            state.complete_step('preprocess')
        except Exception as e:
            state.fail_step('preprocess', str(e))
            raise
    
    # Step 2: Segmentation
    if 'segment' in steps:
        logger.info("\n" + "=" * 40)
        logger.info("Step 2/4: Segmenting nuclei")
        logger.info("=" * 40)
        
        try:
            state.start_step('segment')
            seg_config = config.get('segmentation', {})
            
            # Determine input directory
            dapi_scaled_dir = os.path.join(output_dir, 'dapi_scaled')
            dapi_dir = os.path.join(output_dir, 'dapi')
            
            if os.path.exists(dapi_scaled_dir):
                seg_input_dir = dapi_scaled_dir
            elif os.path.exists(dapi_dir):
                seg_input_dir = dapi_dir
            else:
                seg_input_dir = seg_config.get('input_dir') or config['input_dir']
            
            # Handle None values from config (YAML null)
            seg_output_labels = seg_config.get('output_labels_dir') or os.path.join(output_dir, 'segmented_nucleus')
            seg_output_rois = seg_config.get('output_rois_dir') or os.path.join(output_dir, 'segmented_nuclei_rois')
            
            segment_nuclei(
                input_dir=seg_input_dir,
                output_labels_dir=seg_output_labels,
                output_rois_dir=seg_output_rois,
                prob_thresh=seg_config.get('prob_thresh') or 0.43,
                use_pretrained=seg_config.get('use_pretrained', True),
                state=state,
                resume=resume
            )
            state.complete_step('segment')
        except Exception as e:
            state.fail_step('segment', str(e))
            raise
    
    # Step 3: Feature Extraction
    if 'extract' in steps:
        logger.info("\n" + "=" * 40)
        logger.info("Step 3/4: Extracting features")
        logger.info("=" * 40)
        
        try:
            state.start_step('extract')
            feat_config = config.get('feature_extraction', {})
            
            # Determine directories (handle None values from YAML null)
            dapi_scaled_dir = os.path.join(output_dir, 'dapi_scaled')
            dapi_dir = os.path.join(output_dir, 'dapi')
            
            if os.path.exists(dapi_scaled_dir):
                raw_images_dir = dapi_scaled_dir
            elif os.path.exists(dapi_dir):
                raw_images_dir = dapi_dir
            else:
                raw_images_dir = feat_config.get('raw_image_dir') or config['input_dir']
            
            labels_dir = feat_config.get('labels_dir') or os.path.join(output_dir, 'segmented_nucleus')
            
            # Get protein directories if specified
            protein_dirs = None
            protein_config = feat_config.get('protein_measurement', {})
            if protein_config.get('enabled', False):
                protein_dirs = []
                for protein in protein_config.get('proteins', []):
                    protein_image_dir = protein.get('image_dir', '')
                    if protein_image_dir and os.path.exists(protein_image_dir):
                        protein_dirs.append(protein_image_dir)
                    else:
                        # Check in output directory
                        protein_name = protein.get('name', '')
                        if protein_name:
                            protein_output = os.path.join(output_dir, protein_name)
                            if os.path.exists(protein_output):
                                protein_dirs.append(protein_output)
            
            cell_seg_config = feat_config.get('cell_segmentation', {})
            feat_output_dir = feat_config.get('output_dir') or os.path.join(output_dir, 'features')
            
            extract_features(
                raw_images_dir=raw_images_dir,
                labels_dir=labels_dir,
                output_dir=feat_output_dir,
                protein_dirs=protein_dirs if protein_dirs else None,
                cell_segmentation=cell_seg_config.get('enabled', False),
                dilation_radius=cell_seg_config.get('dilation_radius') or 10,
                extract_spatial=feat_config.get('extract_spatial', True),
                state=state,
                resume=resume
            )
            state.complete_step('extract')
        except Exception as e:
            state.fail_step('extract', str(e))
            raise
    
    # Step 4: Analysis
    if 'analyze' in steps:
        logger.info("\n" + "=" * 40)
        logger.info("Step 4/4: Running analysis")
        logger.info("=" * 40)
        
        try:
            state.start_step('analyze')
            analysis_config = config.get('analysis', {})
            
            # Determine features directory (handle None values)
            features_dir = os.path.join(output_dir, 'features', 'consolidated_features')
            if not os.path.exists(features_dir):
                features_dir = analysis_config.get('features_dir') or os.path.join(output_dir, 'consolidated_features')
            
            # Get enabled analyses
            analysis_types = []
            for analysis in analysis_config.get('analyses', []):
                if analysis.get('enabled', False):
                    analysis_types.append(analysis['type'])
            
            if not analysis_types:
                analysis_types = ['cell_type']  # Default
            
            analysis_output_dir = analysis_config.get('output_dir') or os.path.join(output_dir, 'analysis')
            
            # Get spatial and visualization parameters
            spatial_config = analysis_config.get('spatial_parameters', {})
            
            run_analysis(
                features_dir=features_dir,
                output_dir=analysis_output_dir,
                analysis_types=analysis_types,
                metadata_path=analysis_config.get('metadata'),
                correlation_threshold=analysis_config.get('correlation_threshold', 0.8),
                random_seed=analysis_config.get('random_seed', 1234),
                pixel_size=spatial_config.get('pixel_size', 0.3225),
                contact_radius=spatial_config.get('contact_radius', 15.0),
                signaling_radius=spatial_config.get('signaling_radius', 30.0),
                border_threshold=spatial_config.get('border_threshold', 0.4),
                generate_plots=analysis_config.get('generate_plots', True),
                n_permutations=analysis_config.get('n_permutations', 10000)
            )
            state.complete_step('analyze')
        except Exception as e:
            state.fail_step('analyze', str(e))
            raise
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)
    
    # Save final state
    state.save()

