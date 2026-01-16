#!/usr/bin/env python
"""
Fix enhanced_features.csv by merging duplicate rows for the same nuc_id.

The issue: Spatial graph features were concatenated instead of merged,
creating duplicate rows where one row has spatial features and another
has cell cycle features for the same nuc_id.

This script merges these duplicate rows into a single row per nuc_id.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_enhanced_features_merge(csv_path: str, output_path: str = None):
    """
    Fix duplicate rows in enhanced_features.csv by merging on nuc_id.
    
    Args:
        csv_path: Path to enhanced_features.csv
        output_path: Output path (default: overwrites input)
    """
    logger.info(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Unique nuc_ids: {df['nuc_id'].nunique()}")
    logger.info(f"Total rows: {len(df)}")
    
    # Check for duplicates
    duplicates = df['nuc_id'].value_counts()
    n_duplicates = (duplicates > 1).sum()
    
    if n_duplicates == 0:
        logger.info("No duplicate nuc_ids found. File is already correct.")
        return df
    
    logger.info(f"Found {n_duplicates} nuc_ids with duplicate rows")
    logger.info(f"Max duplicates per nuc_id: {duplicates.max()}")
    
    # Group by nuc_id and merge rows
    logger.info("Merging duplicate rows...")
    
    # Strategy: For each nuc_id, combine all non-null values from all rows
    # Use a more efficient approach with groupby and aggregate
    merged_rows = []
    
    for nuc_id, group in df.groupby('nuc_id'):
        if len(group) == 1:
            merged_rows.append(group.iloc[0])
        else:
            # Start with first row
            merged = group.iloc[0].copy()
            
            # For each subsequent row, fill in missing values
            for idx in range(1, len(group)):
                row = group.iloc[idx]
                # Fill NaN values in merged with values from this row
                mask = merged.isna() & row.notna()
                merged[mask] = row[mask]
            
            merged_rows.append(merged)
    
    # Create DataFrame from merged rows
    merged_df = pd.DataFrame(merged_rows).reset_index(drop=True)
    
    logger.info(f"Merged shape: {merged_df.shape}")
    logger.info(f"Unique nuc_ids after merge: {merged_df['nuc_id'].nunique()}")
    
    # Verify merge worked
    if merged_df['nuc_id'].nunique() != len(merged_df):
        logger.warning("Still have duplicate nuc_ids after merge!")
    else:
        logger.info("✓ All duplicates merged successfully")
    
    # Check that we have both spatial and cell cycle features in same rows
    if 'degree_centrality' in merged_df.columns and 'cell_cycle_phase' in merged_df.columns:
        has_both = (merged_df['degree_centrality'].notna() & merged_df['cell_cycle_phase'].notna()).sum()
        logger.info(f"Rows with both spatial and cell cycle features: {has_both}/{len(merged_df)}")
        
        if has_both == 0:
            logger.warning("WARNING: Still no rows with both spatial and cell cycle features!")
        else:
            logger.info("✓ Merge successful - spatial and cell cycle features are now in same rows")
    
    # Save fixed file
    if output_path is None:
        output_path = csv_path
    
    # Try to save, if permission error, save to temp file first
    try:
        logger.info(f"Saving fixed file to {output_path}...")
        merged_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved fixed file: {output_path}")
    except PermissionError:
        # File might be open in another program, save to temp location
        import tempfile
        temp_path = output_path.replace('.csv', '_fixed_temp.csv')
        logger.warning(f"Permission denied. Saving to temporary file: {temp_path}")
        merged_df.to_csv(temp_path, index=False)
        logger.info(f"✓ Saved to temporary file: {temp_path}")
        logger.info(f"Please close the file if it's open, then manually replace:")
        logger.info(f"  {csv_path}")
        logger.info(f"  with")
        logger.info(f"  {temp_path}")
        logger.info(f"Or run this script again when the file is closed.")
    
    return merged_df


if __name__ == '__main__':
    import sys
    
    # Default paths
    csv_path = 'data/dataset1/processed/features_enhanced/enhanced_features/enhanced_features.csv'
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    output_path = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Create backup (only if it doesn't exist)
    backup_path = csv_path.replace('.csv', '_backup.csv')
    import shutil
    if Path(backup_path).exists():
        logger.info(f"Backup already exists: {backup_path}")
        logger.info("Skipping backup creation to preserve existing backup")
    else:
        logger.info(f"Creating backup: {backup_path}")
        shutil.copy2(csv_path, backup_path)
        logger.info(f"✓ Backup created")
    
    # Fix the file
    fixed_df = fix_enhanced_features_merge(csv_path, output_path)
    
    logger.info("\n" + "="*60)
    logger.info("Fix complete!")
    logger.info("="*60)
