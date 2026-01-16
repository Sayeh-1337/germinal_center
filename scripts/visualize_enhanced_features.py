#!/usr/bin/env python
"""
Visualization and analysis script for enhanced features CSV.

This script provides common visualizations and analyses for understanding
enhanced chromatin features extracted from germinal center nuclei.

Usage:
    python scripts/visualize_enhanced_features.py \
        --features data/features/enhanced_features.csv \
        --output figures/enhanced_features_analysis/
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")


def load_features(csv_path: str) -> pd.DataFrame:
    """Load enhanced features CSV."""
    logger.info(f"Loading features from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} cells with {len(df.columns)} features")
    return df


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_groups: dict,
    output_dir: Path,
    group_col: Optional[str] = None
):
    """
    Plot distributions of features grouped by category.
    
    Args:
        df: DataFrame with features
        feature_groups: Dict mapping category name to list of feature names
        output_dir: Output directory for figures
        group_col: Optional column to stratify by (e.g., 'cell_cycle_phase')
    """
    for category, features in feature_groups.items():
        # Filter to existing features
        valid_features = [f for f in features if f in df.columns]
        if not valid_features:
            logger.warning(f"No valid features for category: {category}")
            # Create an empty plot with a message instead of skipping
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No {category} features found in data', 
                   ha='center', va='center', fontsize=14, 
                   transform=ax.transAxes)
            ax.set_title(f'{category.capitalize()} Features Distribution')
            output_path = output_dir / f"distributions_{category}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved empty plot with message: {output_path}")
            continue
        
        n_features = len(valid_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes = axes.flatten()
        
        for idx, feature in enumerate(valid_features):
            ax = axes[idx]
            
            if group_col and group_col in df.columns:
                # Stratified by group
                groups = df[group_col].dropna().unique()
                # Use distinct colors and lower alpha to reduce overlap
                colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
                for group_idx, group in enumerate(groups):
                    data = df[df[group_col] == group][feature].dropna()
                    if len(data) > 0:
                        ax.hist(data, alpha=0.5, label=str(group), bins=30, 
                               color=colors[group_idx], edgecolor='black', linewidth=0.5)
                if len(groups) > 0:  # Only add legend if there are groups
                    ax.legend(loc='best', fontsize=8)
                ax.set_title(f"{feature}\n(stratified by {group_col})", fontsize=10)
            else:
                # Single distribution
                data = df[feature].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.8, edgecolor='black', linewidth=0.5, color='steelblue')
                    ax.set_title(feature, fontsize=10)
                    mean_val = data.mean()
                    median_val = data.median()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                              label=f'Mean: {mean_val:.3f}')
                    ax.axvline(median_val, color='blue', linestyle='--', linewidth=1.5,
                              label=f'Median: {median_val:.3f}')
                    ax.legend(fontsize=8, loc='best')
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(feature)
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        
        # Hide extra subplots
        for idx in range(len(valid_features), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"distributions_{category}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {output_path}")


def plot_cell_cycle_analysis(
    df: pd.DataFrame,
    output_dir: Path
):
    """Plot cell cycle phase distributions and features by phase."""
    if 'cell_cycle_phase' not in df.columns:
        logger.warning("cell_cycle_phase not found, skipping cell cycle plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Phase distribution
    phase_counts = df['cell_cycle_phase'].value_counts()
    axes[0, 0].pie(phase_counts.values, labels=phase_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Cell Cycle Phase Distribution')
    
    # 2. Proliferating vs quiescent
    if 'is_proliferating' in df.columns:
        proliferating_counts = df['is_proliferating'].value_counts()
        axes[0, 1].bar(['Quiescent (G0/G1)', 'Proliferating (S/G2/M)'], 
                      [proliferating_counts.get(False, 0), 
                       proliferating_counts.get(True, 0)])
        axes[0, 1].set_title('Proliferation Status')
        axes[0, 1].set_ylabel('Count')
    
    # 3. Key features by phase
    key_features = ['fractal_dimension', 'chromatin_condensation_score', 
                   'n_bright_foci', 'wavelet_fine_coarse_ratio']
    key_features = [f for f in key_features if f in df.columns]
    
    if key_features:
        phase_data = []
        for phase in df['cell_cycle_phase'].dropna().unique():
            phase_df = df[df['cell_cycle_phase'] == phase]
            for feature in key_features:
                feature_data = phase_df[feature].dropna()
                if len(feature_data) > 0:
                    phase_data.append({
                        'phase': phase,
                        'feature': feature,
                        'value': feature_data.median()
                    })
        
        if phase_data:
            phase_df_plot = pd.DataFrame(phase_data)
            pivot = phase_df_plot.pivot(index='feature', columns='phase', values='value')
            pivot.plot(kind='bar', ax=axes[1, 0], rot=45, legend=True)
            axes[1, 0].set_title('Median Feature Values by Cell Cycle Phase')
            axes[1, 0].set_ylabel('Median Value')
            axes[1, 0].legend(title='Phase')
    
    # 4. Probability distributions
    prob_cols = [c for c in df.columns if 'cell_cycle_prob' in c]
    if prob_cols:
        prob_data = df[prob_cols].mean()
        axes[1, 1].bar(range(len(prob_data)), prob_data.values)
        axes[1, 1].set_xticks(range(len(prob_data)))
        axes[1, 1].set_xticklabels([c.replace('cell_cycle_prob_', '') for c in prob_data.index], 
                                   rotation=45)
        axes[1, 1].set_title('Average Cell Cycle Probabilities')
        axes[1, 1].set_ylabel('Probability')
    
    plt.tight_layout()
    output_path = output_dir / "cell_cycle_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_wavelet_analysis(
    df: pd.DataFrame,
    output_dir: Path
):
    """Plot wavelet feature patterns across scales."""
    wavelet_cols = [c for c in df.columns if 'wavelet' in c]
    if not wavelet_cols:
        logger.warning("No wavelet features found")
        return
    
    # Plot energy by level
    energy_cols = [c for c in wavelet_cols if 'energy' in c and '_l' in c]
    if energy_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, level in enumerate(['l1', 'l2', 'l3']):
            level_cols = [c for c in energy_cols if f'_{level}' in c]
            if level_cols:
                level_data = df[level_cols].mean()
                axes[idx].barh(range(len(level_data)), level_data.values)
                axes[idx].set_yticks(range(len(level_data)))
                axes[idx].set_yticklabels([c.replace(f'wavelet_', '').replace(f'_{level}', '') 
                                          for c in level_data.index], fontsize=8)
                axes[idx].set_title(f'Wavelet Energy - {level.upper()}')
                axes[idx].set_xlabel('Mean Energy')
        
        plt.tight_layout()
        output_path = output_dir / "wavelet_energy_by_level.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {output_path}")
    
    # Plot fine-coarse ratio distribution
    if 'wavelet_fine_coarse_ratio' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = df['wavelet_fine_coarse_ratio'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(data.mean(), color='red', linestyle='--', 
                      label=f'Mean: {data.mean():.3f}')
            ax.axvline(data.median(), color='blue', linestyle='--', 
                      label=f'Median: {data.median():.3f}')
            ax.set_xlabel('Fine-Coarse Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title('Wavelet Fine-Coarse Ratio Distribution')
            ax.legend()
            plt.tight_layout()
            output_path = output_dir / "wavelet_fine_coarse_ratio.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved {output_path}")


def plot_dz_lz_comparison(
    df: pd.DataFrame,
    output_dir: Path,
    dz_label_col: Optional[str] = None
):
    """
    Compare features between DZ and LZ cells.
    
    Args:
        df: DataFrame with features
        output_dir: Output directory
        dz_label_col: Column with DZ/LZ labels (e.g., 'predicted', 'zone')
    """
    if dz_label_col is None or dz_label_col not in df.columns:
        logger.warning(f"DZ/LZ label column '{dz_label_col}' not found, skipping comparison")
        return
    
    # Key features for DZ/LZ discrimination
    comparison_features = [
        'wavelet_fine_coarse_ratio',
        'is_proliferating',
        'n_bright_foci',
        'fractal_dimension',
        'voronoi_area',
        'domain_area_fraction',
        'radial_intensity_gradient'
    ]
    comparison_features = [f for f in comparison_features if f in df.columns]
    
    if not comparison_features:
        logger.warning("No comparison features found")
        return
    
    n_features = len(comparison_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    zones = df[dz_label_col].dropna().unique()
    
    for idx, feature in enumerate(comparison_features):
        ax = axes[idx]
        
        data_to_plot = []
        labels_to_plot = []
        for zone in zones:
            zone_data = df[df[dz_label_col] == zone][feature].dropna()
            if len(zone_data) > 0:
                data_to_plot.append(zone_data.values)
                labels_to_plot.append(str(zone))
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
            # Use distinct colors with better contrast
            colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']  # Red, Blue, Green, Orange, Purple
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)  # Slightly higher alpha for better visibility
                patch.set_edgecolor('black')
                patch.set_linewidth(1.2)
            
            ax.set_title(feature)
            ax.set_ylabel('Value')
            
            # Statistical test
            if len(data_to_plot) == 2:
                try:
                    stat, pval = stats.mannwhitneyu(data_to_plot[0], data_to_plot[1])
                    ax.text(0.5, 0.95, f'p={pval:.2e}', transform=ax.transAxes,
                           ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat'))
                except Exception as e:
                    logger.warning(f"Could not compute statistics for {feature}: {e}")
    
    # Hide extra subplots
    for idx in range(len(comparison_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / "dz_lz_feature_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_feature_correlations(
    df: pd.DataFrame,
    output_dir: Path,
    feature_subset: Optional[List[str]] = None,
    max_features: int = 30
):
    """Plot correlation matrix of features."""
    if feature_subset is None:
        # Select numeric features, excluding IDs
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ['label', 'nuc_id', 'image']
        feature_subset = [c for c in numeric_cols if not any(e in str(c).lower() for e in exclude)]
    
    # Limit to max_features most variable features
    if len(feature_subset) > max_features:
        variances = df[feature_subset].var().sort_values(ascending=False)
        feature_subset = variances.head(max_features).index.tolist()
    
    feature_subset = [f for f in feature_subset if f in df.columns]
    
    if len(feature_subset) < 2:
        logger.warning("Not enough features for correlation plot")
        return
    
    corr_matrix = df[feature_subset].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
               square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
               xticklabels=False, yticklabels=False, ax=ax)
    ax.set_title(f'Feature Correlation Matrix (top {len(feature_subset)} features)')
    
    plt.tight_layout()
    output_path = output_dir / "feature_correlations.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {output_path}")


def generate_summary_statistics(
    df: pd.DataFrame,
    output_dir: Path
):
    """Generate summary statistics table."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        logger.warning("No numeric columns found for summary statistics")
        return
    
    summary_stats = df[numeric_cols].describe()
    
    # Add additional statistics
    summary_stats.loc['skewness'] = df[numeric_cols].skew()
    summary_stats.loc['kurtosis'] = df[numeric_cols].kurtosis()
    summary_stats.loc['missing'] = df[numeric_cols].isna().sum()
    summary_stats.loc['missing_pct'] = (df[numeric_cols].isna().sum() / len(df)) * 100
    
    output_path = output_dir / "summary_statistics.csv"
    summary_stats.to_csv(output_path)
    logger.info(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and analyze enhanced features CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/visualize_enhanced_features.py \\
      --features data/features/enhanced_features.csv \\
      --output figures/enhanced_analysis/

  # With DZ/LZ comparison
  python scripts/visualize_enhanced_features.py \\
      --features data/features/enhanced_features.csv \\
      --output figures/enhanced_analysis/ \\
      --dz-label-col predicted

  # Group by cell cycle phase
  python scripts/visualize_enhanced_features.py \\
      --features data/features/enhanced_features.csv \\
      --output figures/enhanced_analysis/ \\
      --group-by cell_cycle_phase
        """
    )
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to enhanced_features.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for figures'
    )
    parser.add_argument(
        '--dz-label-col',
        type=str,
        default=None,
        help='Column name with DZ/LZ labels (optional, e.g., "predicted" or "zone")'
    )
    parser.add_argument(
        '--group-by',
        type=str,
        default='cell_cycle_phase',
        help='Column to group by for stratified plots (default: cell_cycle_phase)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_features(args.features)
    
    # Define feature groups
    feature_groups = {
        'wavelet': [c for c in df.columns if 'wavelet' in c],
        'fractal': [c for c in df.columns if 'fractal' in c],
        'domain': [c for c in df.columns if 'domain' in c],
        'radial': [c for c in df.columns if 'radial' in c],
        'cell_cycle': [c for c in df.columns if 'cell_cycle' in c or 'bright_foci' in c or 'condensation' in c],
        'spatial': [c for c in df.columns if any(keyword in c.lower() for keyword in [
            'voronoi', 'centrality', 'density', 'degree', 'clustering', 
            'betweenness', 'pagerank', 'morans', 'neighbor', 'gradient'
        ])]
    }
    
    # Generate plots
    logger.info("Generating feature distribution plots...")
    plot_feature_distributions(
        df, feature_groups, output_dir, 
        group_col=args.group_by if args.group_by in df.columns else None
    )
    
    logger.info("Generating cell cycle analysis...")
    plot_cell_cycle_analysis(df, output_dir)
    
    logger.info("Generating wavelet analysis...")
    plot_wavelet_analysis(df, output_dir)
    
    if args.dz_label_col:
        logger.info("Generating DZ/LZ comparison...")
        plot_dz_lz_comparison(df, output_dir, args.dz_label_col)
    
    logger.info("Generating correlation matrix...")
    plot_feature_correlations(df, output_dir)
    
    logger.info("Generating summary statistics...")
    generate_summary_statistics(df, output_dir)
    
    logger.info(f"Analysis complete! Figures saved to {output_dir}")


if __name__ == '__main__':
    main()
