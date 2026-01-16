"""Visualization functions for germinal center analysis"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)

# Default color palettes
CELL_TYPE_COLORS = {
    'DZ B-cells': '#E74C3C',  # Red
    'LZ B-cells': '#3498DB',  # Blue
    'T-cells': '#2ECC71',     # Green
}

TCELL_INFLUENCE_COLORS = {
    'T-cell interactors': '#E74C3C',
    'potential T-cell interactors': '#F39C12',
    'Non-T-cell interactors': '#3498DB',
}

FEATURE_CATEGORY_COLORS = {
    'morphology': '#3498DB',
    'intensity': '#2ECC71',
    'boundary': '#E74C3C',
    'texture': '#9B59B6',
    'chromatin condensation': '#F39C12',
    'moments': '#1ABC9C',
}


def compute_umap_embedding(
    data: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """Compute UMAP embedding for feature data
    
    Args:
        data: DataFrame with numeric features
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        n_components: Number of UMAP dimensions
        random_state: Random seed
        
    Returns:
        DataFrame with UMAP coordinates
    """
    try:
        from umap import UMAP
    except ImportError:
        logger.error("umap-learn not installed. Install with: pip install umap-learn")
        raise
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    
    # Compute UMAP
    logger.info(f"Computing UMAP embedding (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state
    )
    embedding = reducer.fit_transform(scaled_data)
    
    # Create result DataFrame
    result = pd.DataFrame(
        embedding,
        index=data.index,
        columns=[f'umap_{i}' for i in range(n_components)]
    )
    
    return result


def cluster_umap_embedding(
    embedding: pd.DataFrame,
    min_cluster_size: int = 100,
    min_samples: int = 10
) -> np.ndarray:
    """Cluster UMAP embedding using HDBSCAN
    
    Args:
        embedding: DataFrame with UMAP coordinates
        min_cluster_size: HDBSCAN min_cluster_size
        min_samples: HDBSCAN min_samples
        
    Returns:
        Array of cluster labels (-1 for noise)
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        logger.error("hdbscan not installed. Install with: pip install hdbscan")
        raise
    
    logger.info(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(embedding.values)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"  Found {n_clusters} clusters, {n_noise} noise points")
    
    return labels


def plot_umap(
    embedding: pd.DataFrame,
    labels: Optional[Union[np.ndarray, pd.Series]] = None,
    output_path: Optional[str] = None,
    title: str = "UMAP Embedding",
    figsize: Tuple[int, int] = (10, 8),
    point_size: float = 5,
    alpha: float = 0.7,
    cmap: str = 'Set1'
) -> None:
    """Plot UMAP embedding with optional labels
    
    Args:
        embedding: DataFrame with UMAP coordinates
        labels: Optional labels for coloring
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Point size
        alpha: Point alpha
        cmap: Colormap for discrete labels
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        logger.error("matplotlib not installed")
        raise
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        scatter = ax.scatter(
            embedding['umap_0'],
            embedding['umap_1'],
            c=pd.Categorical(labels).codes if not np.issubdtype(type(labels[0]), np.number) else labels,
            cmap=cmap,
            s=point_size,
            alpha=alpha
        )
        if len(np.unique(labels)) <= 20:
            plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        ax.scatter(
            embedding['umap_0'],
            embedding['umap_1'],
            s=point_size,
            alpha=alpha
        )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved UMAP plot to {output_path}")
    
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_labels: List[str],
    output_path: Optional[str] = None,
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (8, 6)
) -> Dict:
    """Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels
        y_scores: Predicted probabilities (n_samples, n_classes)
        class_labels: Class label names
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Dictionary with AUC values
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
    except ImportError:
        logger.error("matplotlib or sklearn not installed")
        raise
    
    # Binarize labels - handle both string and integer labels
    n_classes = len(class_labels)
    if len(y_true) > 0 and isinstance(y_true[0], str):
        y_bin = label_binarize(y_true, classes=class_labels)
    else:
        y_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, label in enumerate(class_labels):
        if y_scores.ndim == 1:
            fpr[label], tpr[label], _ = roc_curve(y_bin[:, i], y_scores)
        else:
            fpr[label], tpr[label], _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    for (label, color) in zip(class_labels, colors):
        ax.plot(
            fpr[label], tpr[label],
            color=color,
            lw=2,
            label=f'{label} (AUC = {roc_auc[label]:.2f})'
        )
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ROC plot to {output_path}")
    
    plt.close()
    
    return roc_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True
) -> pd.DataFrame:
    """Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: Class label names
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize by row
        
    Returns:
        Confusion matrix DataFrame
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    # Use class_labels directly if labels are strings, otherwise use integers
    if len(y_true) > 0 and isinstance(y_true[0], str):
        cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    else:
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_labels)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {output_path}")
    
    plt.close()
    
    return cm_df


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Feature Importance",
    n_features: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> pd.DataFrame:
    """Plot feature importance bar chart
    
    Args:
        importance: Array of importance values
        feature_names: List of feature names
        output_path: Path to save figure
        title: Plot title
        n_features: Number of top features to show
        figsize: Figure size
        
    Returns:
        DataFrame with sorted feature importance
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.error("matplotlib not installed")
        raise
    
    # Create DataFrame and sort
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    top_features = imp_df.head(n_features)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        range(len(top_features)),
        top_features['importance'],
        color='steelblue'
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")
    
    plt.close()
    
    return imp_df


def plot_violin_comparison(
    data: pd.DataFrame,
    feature: str,
    group_col: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    palette: Optional[List[str]] = None
) -> None:
    """Plot violin plot comparing groups
    
    Args:
        data: DataFrame with feature and group columns
        feature: Column name of feature to plot
        group_col: Column name of grouping variable
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        palette: Color palette
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib or seaborn not installed")
        raise
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.violinplot(
        data=data,
        x=group_col,
        y=feature,
        ax=ax,
        palette=palette
    )
    
    ax.set_xlabel(group_col)
    ax.set_ylabel(feature)
    ax.set_title(title or f'{feature} by {group_col}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved violin plot to {output_path}")
    
    plt.close()


def plot_spatial_scatter(
    data: pd.DataFrame,
    x_col: str = 'spat_centroid_x',
    y_col: str = 'spat_centroid_y',
    color_col: Optional[str] = None,
    output_path: Optional[str] = None,
    title: str = "Spatial Distribution",
    figsize: Tuple[int, int] = (10, 10),
    point_size: float = 10,
    alpha: float = 0.7,
    palette: Optional[Union[str, List[str]]] = None
) -> None:
    """Plot spatial scatter plot
    
    Args:
        data: DataFrame with spatial coordinates
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        color_col: Column for coloring points
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Point size
        alpha: Point alpha
        palette: Color palette
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib or seaborn not installed")
        raise
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_col and color_col in data.columns:
        sns.scatterplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=color_col,
            ax=ax,
            s=point_size,
            alpha=alpha,
            palette=palette
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(
            data[x_col],
            data[y_col],
            s=point_size,
            alpha=alpha
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved spatial plot to {output_path}")
    
    plt.close()


def plot_cell_type_distribution(
    data: pd.DataFrame,
    cell_type_col: str = 'cell_type',
    hue_col: Optional[str] = None,
    output_path: Optional[str] = None,
    title: str = "Cell Type Distribution",
    figsize: Tuple[int, int] = (10, 6),
    palette: Optional[Dict] = None
) -> None:
    """Plot bar chart of cell type distribution
    
    Args:
        data: DataFrame with cell type column
        cell_type_col: Column with cell type labels
        hue_col: Optional column for grouping (e.g., 'gc_status', 'tcell_influence')
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        palette: Color palette dict
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib or seaborn not installed")
        raise
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if palette is None:
        palette = CELL_TYPE_COLORS if hue_col is None else None
    
    if hue_col and hue_col in data.columns:
        sns.countplot(
            data=data,
            x=cell_type_col,
            hue=hue_col,
            ax=ax,
            palette=palette
        )
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Get order and colors
        order = data[cell_type_col].value_counts().index.tolist()
        colors = [CELL_TYPE_COLORS.get(ct, '#95A5A6') for ct in order]
        
        sns.countplot(
            data=data,
            x=cell_type_col,
            ax=ax,
            order=order,
            hue=cell_type_col,
            palette=colors,
            legend=False
        )
    
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add count labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cell type distribution to {output_path}")
    
    plt.close()


def plot_tcell_influence_distribution(
    data: pd.DataFrame,
    cell_type_col: str = 'cell_type',
    influence_col: str = 'tcell_influence',
    output_path: Optional[str] = None,
    title: str = "T-cell Influence by Cell Type",
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot T-cell influence distribution by cell type
    
    Args:
        data: DataFrame with cell type and influence columns
        cell_type_col: Column with cell type labels
        influence_col: Column with T-cell influence labels
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib or seaborn not installed")
        raise
    
    fig, ax = plt.subplots(figsize=figsize)
    
    influence_order = ['T-cell interactors', 'potential T-cell interactors', 'Non-T-cell interactors']
    influence_order = [o for o in influence_order if o in data[influence_col].unique()]
    
    sns.countplot(
        data=data,
        x=cell_type_col,
        hue=influence_col,
        ax=ax,
        hue_order=influence_order,
        palette=TCELL_INFLUENCE_COLORS
    )
    
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(title='T-cell Influence', bbox_to_anchor=(0.5, -0.15), 
              loc='upper center', ncol=3, frameon=False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved T-cell influence distribution to {output_path}")
    
    plt.close()


def plot_roc_binary(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    pos_label: str,
    output_path: Optional[str] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6)
) -> Dict:
    """Plot ROC curve for binary classification
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities for positive class
        pos_label: Name of positive class
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        
    Returns:
        Dictionary with AUC and threshold info
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from sklearn.metrics import roc_curve, auc
    except ImportError:
        logger.error("matplotlib or sklearn not installed")
        raise
    
    # Handle pos_label - if y_true is numeric/boolean, pos_label should be numeric too
    y_true_array = np.array(y_true)
    if y_true_array.dtype == bool or (y_true_array.dtype in [np.int64, np.int32] and set(np.unique(y_true_array)).issubset({0, 1})):
        # y_true is already encoded, use pos_label=1 (or convert pos_label to numeric)
        if isinstance(pos_label, str):
            # Try to infer: if pos_label is a string but y_true is numeric, assume pos_label=1
            pos_label_num = 1
        else:
            pos_label_num = pos_label
        fpr, tpr, thresholds = roc_curve(y_true_array, y_scores, pos_label=pos_label_num)
    else:
        # y_true contains string labels, use pos_label as-is
        fpr, tpr, thresholds = roc_curve(y_true_array, y_scores, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='#E74C3C', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    
    # Fill area under curve
    ax.fill_between(fpr, tpr, alpha=0.2, color='#E74C3C')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {output_path}")
    
    plt.close()
    
    return {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr}


def plot_violin_with_stats(
    data: pd.DataFrame,
    feature: str,
    group_col: str,
    groups: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    palette: Optional[Dict] = None,
    show_pvalue: bool = True
) -> Dict:
    """Plot violin plot with statistical annotation
    
    Args:
        data: DataFrame with feature and group columns
        feature: Column name of feature to plot
        group_col: Column name of grouping variable
        groups: List of groups to compare (default: all unique values)
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        palette: Color palette
        show_pvalue: Whether to show p-value annotation
        
    Returns:
        Dictionary with statistical test results
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from scipy import stats
    except ImportError:
        logger.error("matplotlib, seaborn or scipy not installed")
        raise
    
    if groups is None:
        groups = data[group_col].unique().tolist()
    
    plot_data = data[data[group_col].isin(groups)].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if palette is None:
        palette = CELL_TYPE_COLORS
    
    sns.violinplot(
        data=plot_data,
        x=group_col,
        y=feature,
        ax=ax,
        order=groups,
        palette=palette,
        inner='box'
    )
    
    # Compute statistics
    result = {}
    if len(groups) == 2:
        group1_data = plot_data[plot_data[group_col] == groups[0]][feature].dropna()
        group2_data = plot_data[plot_data[group_col] == groups[1]][feature].dropna()
        
        stat, pvalue = stats.ttest_ind(group1_data, group2_data, equal_var=False)
        result = {'statistic': stat, 'pvalue': pvalue, 'test': 'Welch t-test'}
        
        if show_pvalue:
            # Add p-value annotation
            y_max = plot_data[feature].max()
            y_range = plot_data[feature].max() - plot_data[feature].min()
            
            # Format p-value
            if pvalue < 0.001:
                pval_text = 'p < 0.001'
            elif pvalue < 0.01:
                pval_text = f'p = {pvalue:.3f}'
            else:
                pval_text = f'p = {pvalue:.2f}'
            
            # Draw bracket and p-value
            ax.plot([0, 0, 1, 1], 
                    [y_max + 0.05*y_range, y_max + 0.1*y_range, 
                     y_max + 0.1*y_range, y_max + 0.05*y_range], 
                    'k-', lw=1)
            ax.text(0.5, y_max + 0.12*y_range, pval_text, 
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel(group_col)
    ax.set_ylabel(feature)
    ax.set_title(title or f'{feature} by {group_col}')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved violin plot to {output_path}")
    
    plt.close()
    
    return result


def plot_spatial_on_image(
    data: pd.DataFrame,
    image_path: str,
    x_col: str = 'centroid-1',
    y_col: str = 'centroid-0',
    color_col: Optional[str] = None,
    output_path: Optional[str] = None,
    title: str = "Spatial Distribution",
    figsize: Tuple[int, int] = (16, 6),
    point_size: float = 10,
    alpha: float = 0.7,
    palette: Optional[Dict] = None,
    block_channel: Optional[int] = None
) -> None:
    """Plot spatial distribution overlaid on raw image
    
    Args:
        data: DataFrame with spatial coordinates
        image_path: Path to raw image
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        color_col: Column for coloring points
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Point size
        alpha: Point alpha
        palette: Color palette dict
        block_channel: Channel to zero out (e.g., 0 for DAPI)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from tifffile import imread
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    # Load image
    image = imread(image_path)
    if block_channel is not None and image.ndim == 3:
        image = image.copy()
        if image.shape[0] < image.shape[-1]:  # CHW format
            image[block_channel] = 0
        else:  # HWC format
            image[:, :, block_channel] = 0
    
    # Transpose if needed for display
    if image.ndim == 3 and image.shape[0] < image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Raw image
    axes[0].imshow(image)
    axes[0].set_title('Raw Image')
    axes[0].axis('off')
    
    # Cell type scatter
    if color_col and color_col in data.columns:
        if palette is None:
            palette = CELL_TYPE_COLORS
        
        for label in data[color_col].unique():
            mask = data[color_col] == label
            color = palette.get(label, '#95A5A6')
            axes[1].scatter(
                data.loc[mask, x_col],
                data.loc[mask, y_col],
                c=color,
                s=point_size,
                alpha=alpha,
                label=label
            )
        axes[1].legend(loc='upper right', fontsize=8)
    else:
        axes[1].scatter(
            data[x_col],
            data[y_col],
            s=point_size,
            alpha=alpha
        )
    
    axes[1].set_xlim([0, image.shape[1]])
    axes[1].set_ylim([image.shape[0], 0])  # Invert y-axis
    axes[1].set_title('Cell Positions')
    axes[1].set_aspect('equal')
    
    # Overlay on image
    axes[2].imshow(image)
    if color_col and color_col in data.columns:
        for label in data[color_col].unique():
            mask = data[color_col] == label
            color = palette.get(label, '#95A5A6')
            axes[2].scatter(
                data.loc[mask, x_col],
                data.loc[mask, y_col],
                c=color,
                s=point_size,
                alpha=alpha,
                label=label
            )
    else:
        axes[2].scatter(
            data[x_col],
            data[y_col],
            s=point_size,
            alpha=alpha,
            c='yellow'
        )
    
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved spatial overlay to {output_path}")
    
    plt.close()


def plot_tcell_interaction_zones(
    data: pd.DataFrame,
    image_path: str,
    x_col: str = 'centroid-1',
    y_col: str = 'centroid-0',
    cell_type_col: str = 'cell_type',
    influence_col: str = 'tcell_influence',
    tcell_label: str = 'T-cells',
    output_path: Optional[str] = None,
    title: str = "T-cell Interaction Zones",
    figsize: Tuple[int, int] = (20, 6),
    point_size: float = 10,
    alpha: float = 0.7
) -> None:
    """Plot T-cell interaction zones on image
    
    Args:
        data: DataFrame with cell data and spatial coordinates
        image_path: Path to raw image
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        cell_type_col: Column with cell type labels
        influence_col: Column with T-cell influence labels
        tcell_label: Label for T-cells
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Point size
        alpha: Point alpha
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from tifffile import imread
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    # Load image
    image = imread(image_path)
    if image.ndim == 3 and image.shape[0] < image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # 1. Raw image
    axes[0].imshow(image)
    axes[0].set_title('Raw Image')
    axes[0].axis('off')
    
    # 2. T-cell vs non-T-cell
    is_tcell = data[cell_type_col] == tcell_label
    axes[1].scatter(
        data.loc[~is_tcell, x_col],
        data.loc[~is_tcell, y_col],
        c='#3498DB',
        s=point_size,
        alpha=alpha,
        label='Non-T-cell'
    )
    axes[1].scatter(
        data.loc[is_tcell, x_col],
        data.loc[is_tcell, y_col],
        c='#E74C3C',
        s=point_size,
        alpha=alpha,
        label='T-cell'
    )
    axes[1].set_xlim([0, image.shape[1]])
    axes[1].set_ylim([image.shape[0], 0])
    axes[1].set_title('T-cells vs Non-T-cells')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_aspect('equal')
    
    # 3. T-cell influence zones
    if influence_col in data.columns:
        for influence, color in TCELL_INFLUENCE_COLORS.items():
            if influence in data[influence_col].values:
                mask = data[influence_col] == influence
                axes[2].scatter(
                    data.loc[mask, x_col],
                    data.loc[mask, y_col],
                    c=color,
                    s=point_size,
                    alpha=alpha,
                    label=influence
                )
        axes[2].set_xlim([0, image.shape[1]])
        axes[2].set_ylim([image.shape[0], 0])
        axes[2].set_title('T-cell Influence Zones')
        axes[2].legend(loc='upper right', fontsize=7)
        axes[2].set_aspect('equal')
    
    # 4. Cell types
    for cell_type, color in CELL_TYPE_COLORS.items():
        if cell_type in data[cell_type_col].values:
            mask = data[cell_type_col] == cell_type
            axes[3].scatter(
                data.loc[mask, x_col],
                data.loc[mask, y_col],
                c=color,
                s=point_size,
                alpha=alpha,
                label=cell_type
            )
    axes[3].set_xlim([0, image.shape[1]])
    axes[3].set_ylim([image.shape[0], 0])
    axes[3].set_title('Cell Types')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved T-cell interaction zones to {output_path}")
    
    plt.close()


def plot_prediction_map(
    data: pd.DataFrame,
    image_path: str,
    pred_col: str = 'predicted',
    prob_col: Optional[str] = None,
    true_col: Optional[str] = None,
    x_col: str = 'centroid-1',
    y_col: str = 'centroid-0',
    output_path: Optional[str] = None,
    title: str = "Prediction Map",
    figsize: Tuple[int, int] = (20, 6),
    point_size: float = 10,
    alpha: float = 0.7
) -> None:
    """Plot prediction results on image
    
    Args:
        data: DataFrame with predictions and coordinates
        image_path: Path to raw image
        pred_col: Column with predicted labels
        prob_col: Column with prediction probabilities
        true_col: Column with true labels
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Point size
        alpha: Point alpha
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        from tifffile import imread
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    image = imread(image_path)
    if image.ndim == 3 and image.shape[0] < image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))
    
    n_plots = 2 + (1 if true_col else 0) + (1 if prob_col else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    ax_idx = 0
    
    # 1. Raw image
    axes[ax_idx].imshow(image)
    axes[ax_idx].set_title('Raw Image')
    axes[ax_idx].axis('off')
    ax_idx += 1
    
    # 2. True labels (if provided)
    if true_col and true_col in data.columns:
        for label in data[true_col].unique():
            mask = data[true_col] == label
            color = CELL_TYPE_COLORS.get(label, '#95A5A6')
            axes[ax_idx].scatter(
                data.loc[mask, x_col],
                data.loc[mask, y_col],
                c=color,
                s=point_size,
                alpha=alpha,
                label=label
            )
        axes[ax_idx].set_xlim([0, image.shape[1]])
        axes[ax_idx].set_ylim([image.shape[0], 0])
        axes[ax_idx].set_title('True Labels')
        axes[ax_idx].legend(loc='upper right', fontsize=8)
        axes[ax_idx].set_aspect('equal')
        ax_idx += 1
    
    # 3. Predictions
    for label in data[pred_col].unique():
        mask = data[pred_col] == label
        color = CELL_TYPE_COLORS.get(label, '#95A5A6')
        axes[ax_idx].scatter(
            data.loc[mask, x_col],
            data.loc[mask, y_col],
            c=color,
            s=point_size,
            alpha=alpha,
            label=label
        )
    axes[ax_idx].set_xlim([0, image.shape[1]])
    axes[ax_idx].set_ylim([image.shape[0], 0])
    axes[ax_idx].set_title('Predictions')
    axes[ax_idx].legend(loc='upper right', fontsize=8)
    axes[ax_idx].set_aspect('equal')
    ax_idx += 1
    
    # 4. Probability heatmap (if provided)
    if prob_col and prob_col in data.columns:
        scatter = axes[ax_idx].scatter(
            data[x_col],
            data[y_col],
            c=data[prob_col],
            cmap='inferno',
            s=point_size,
            alpha=alpha
        )
        axes[ax_idx].set_xlim([0, image.shape[1]])
        axes[ax_idx].set_ylim([image.shape[0], 0])
        axes[ax_idx].set_title('Prediction Probability')
        axes[ax_idx].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[ax_idx], label='Probability')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved prediction map to {output_path}")
    
    plt.close()


def plot_feature_importance_colored(
    importance: np.ndarray,
    feature_names: List[str],
    feature_categories: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    title: str = "Feature Importance",
    n_features: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> pd.DataFrame:
    """Plot feature importance with category-based coloring
    
    Args:
        importance: Array of importance values
        feature_names: List of feature names
        feature_categories: Dict mapping feature names to categories
        output_path: Path to save figure
        title: Plot title
        n_features: Number of top features to show
        figsize: Figure size
        
    Returns:
        DataFrame with sorted feature importance
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.error("matplotlib not installed")
        raise
    
    # Create DataFrame and sort
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Add categories if provided
    if feature_categories:
        imp_df['category'] = imp_df['feature'].map(
            lambda x: feature_categories.get(x, 'other')
        )
    else:
        imp_df['category'] = 'other'
    
    # Plot top features
    top_features = imp_df.head(n_features)
    
    # Get colors
    colors = [FEATURE_CATEGORY_COLORS.get(cat, '#95A5A6') 
              for cat in top_features['category']]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(
        range(len(top_features)),
        top_features['importance'],
        color=colors
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Add legend for categories
    if feature_categories:
        unique_cats = top_features['category'].unique()
        legend_patches = [plt.Rectangle((0,0), 1, 1, 
                          color=FEATURE_CATEGORY_COLORS.get(cat, '#95A5A6'),
                          label=cat)
                         for cat in unique_cats if cat != 'other']
        if legend_patches:
            ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")
    
    plt.close()
    
    return imp_df


def plot_marker_comparison(
    data: pd.DataFrame,
    markers: pd.DataFrame,
    group_col: str,
    n_markers: int = 5,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """Plot violin plots for top marker features
    
    Args:
        data: DataFrame with features and group column
        markers: DataFrame with marker analysis results (feature, pvalue, etc.)
        group_col: Column with group labels
        n_markers: Number of top markers to plot
        output_path: Path to save figure
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib or seaborn not installed")
        raise
    
    # Get top markers
    top_markers = markers.nsmallest(n_markers, 'pvalue')['feature'].tolist()
    
    n_cols = min(3, len(top_markers))
    n_rows = (len(top_markers) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    
    for idx, feature in enumerate(top_markers):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        
        if feature in data.columns:
            plot_data = data[[feature, group_col]].dropna()
            
            sns.violinplot(
                data=plot_data,
                x=group_col,
                y=feature,
                ax=ax,
                palette=CELL_TYPE_COLORS,
                inner='box'
            )
            
            # Get p-value from markers
            pval = markers[markers['feature'] == feature]['pvalue'].values
            if len(pval) > 0:
                pval_text = f'p={pval[0]:.2e}' if pval[0] < 0.01 else f'p={pval[0]:.3f}'
                ax.set_title(f'{feature}\n{pval_text}', fontsize=10)
            else:
                ax.set_title(feature, fontsize=10)
            
            ax.set_xlabel('')
    
    # Hide empty subplots
    for idx in range(len(top_markers), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.suptitle(f'Top {n_markers} Marker Features', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved marker comparison to {output_path}")
    
    plt.close()


def plot_dz_probability_violin(
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "DZ B-cell Prediction Probability by T-cell Interaction",
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot violin plot of DZ prediction probability by cell type and T-cell influence
    
    Args:
        data: DataFrame with dz_probability, true_cell_type, and tcell_influence columns
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from scipy import stats
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create combined category for x-axis
    data = data.copy()
    
    influence_palette = {
        'T-cell interactors': '#E74C3C',
        'Non-T-cell interactors': '#3498DB'
    }
    
    sns.violinplot(
        data=data,
        x='true_cell_type',
        y='dz_probability',
        hue='tcell_influence',
        hue_order=['T-cell interactors', 'Non-T-cell interactors'],
        palette=influence_palette,
        ax=ax,
        inner='box',
        split=False
    )
    
    ax.set_xlabel('Cell Type')
    ax.set_ylabel('DZ B-cell Prediction Probability')
    ax.set_title(title)
    ax.legend(title='T-cell Interaction', loc='lower left')
    
    # Add statistical annotations
    for i, cell_type in enumerate(['DZ B-cells', 'LZ B-cells']):
        if cell_type not in data['true_cell_type'].values:
            continue
            
        ct_data = data[data['true_cell_type'] == cell_type]
        interactors = ct_data[ct_data['tcell_influence'] == 'T-cell interactors']['dz_probability']
        non_interactors = ct_data[ct_data['tcell_influence'] == 'Non-T-cell interactors']['dz_probability']
        
        if len(interactors) > 0 and len(non_interactors) > 0:
            stat, pval = stats.ttest_ind(interactors, non_interactors, equal_var=False)
            
            # Format p-value
            if pval < 0.0001:
                pval_text = '****'
            elif pval < 0.001:
                pval_text = '***'
            elif pval < 0.01:
                pval_text = '**'
            elif pval < 0.05:
                pval_text = '*'
            else:
                pval_text = 'ns'
            
            y_max = data['dz_probability'].max()
            y_height = y_max + 0.05 + (i * 0.1)
            
            # Draw bracket
            x_pos = i
            ax.plot([x_pos - 0.2, x_pos - 0.2, x_pos + 0.2, x_pos + 0.2],
                    [y_height, y_height + 0.02, y_height + 0.02, y_height],
                    'k-', lw=1)
            ax.text(x_pos, y_height + 0.03, pval_text, ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved DZ probability violin plot to {output_path}")
    
    plt.close()


def plot_tcell_fraction_comparison(
    stats_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title: str = "Fraction of T-cell Interacting B-cells",
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """Plot bar comparison of T-cell fraction in DZ vs LZ with statistical annotation
    
    Args:
        stats_df: DataFrame with per-image statistics including:
            - freq_tcell_interactors_in_dz
            - freq_tcell_interactors_in_lz
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from scipy import stats
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Fraction of T-cell interactors within each zone
    dz_fractions = stats_df['freq_tcell_interactors_in_dz'].values
    lz_fractions = stats_df['freq_tcell_interactors_in_lz'].values
    
    plot_data1 = pd.DataFrame({
        'Fraction': np.concatenate([dz_fractions, lz_fractions]),
        'Zone': ['DZ'] * len(dz_fractions) + ['LZ'] * len(lz_fractions)
    })
    
                sns.barplot(data=plot_data1, x='Zone', y='Fraction', hue='Zone', ax=axes[0], 
                palette=['#E74C3C', '#3498DB'], capsize=0.1, errorbar='sd', 
                err_kws={'linewidth': 1.5}, legend=False)
    
    # Add statistical annotation
    stat, pval = stats.ttest_ind(dz_fractions, lz_fractions, equal_var=False)
    
    if pval < 0.0001:
        pval_text = '****'
    elif pval < 0.001:
        pval_text = '***'
    elif pval < 0.01:
        pval_text = '**'
    elif pval < 0.05:
        pval_text = '*'
    else:
        pval_text = 'ns'
    
    y_max = max(plot_data1['Fraction'].max(), 0.5)
    axes[0].plot([0, 0, 1, 1], [y_max*1.05, y_max*1.1, y_max*1.1, y_max*1.05], 'k-', lw=1)
    axes[0].text(0.5, y_max*1.12, pval_text, ha='center', fontsize=12)
    
    axes[0].set_xlabel('B-cell Zone')
    axes[0].set_ylabel('Fraction of T-cell Interactors')
    axes[0].set_title('T-cell Interactors per Zone')
    
    # Plot 2: Distribution of DZ vs LZ among T-cell interactors
    if 'freq_dz_of_tcell_interactors' in stats_df.columns:
        dz_of_interactors = stats_df['freq_dz_of_tcell_interactors'].values
        lz_of_interactors = stats_df['freq_lz_of_tcell_interactors'].values
        
        plot_data2 = pd.DataFrame({
            'Fraction': np.concatenate([dz_of_interactors, lz_of_interactors]),
            'Zone': ['DZ'] * len(dz_of_interactors) + ['LZ'] * len(lz_of_interactors)
        })
        
        sns.barplot(data=plot_data2, x='Zone', y='Fraction', hue='Zone', ax=axes[1],
                    palette=['#E74C3C', '#3498DB'], capsize=0.1, errorbar='sd',
                    err_kws={'linewidth': 1.5}, legend=False)
        
        # Statistical test
        stat2, pval2 = stats.ttest_ind(dz_of_interactors, lz_of_interactors, equal_var=False)
        
        if pval2 < 0.0001:
            pval_text2 = '****'
        elif pval2 < 0.001:
            pval_text2 = '***'
        elif pval2 < 0.01:
            pval_text2 = '**'
        elif pval2 < 0.05:
            pval_text2 = '*'
        else:
            pval_text2 = 'ns'
        
        y_max2 = max(plot_data2['Fraction'].max(), 0.5)
        axes[1].plot([0, 0, 1, 1], [y_max2*1.05, y_max2*1.1, y_max2*1.1, y_max2*1.05], 'k-', lw=1)
        axes[1].text(0.5, y_max2*1.12, pval_text2, ha='center', fontsize=12)
        
        axes[1].set_xlabel('B-cell Zone')
        axes[1].set_ylabel('Fraction')
        axes[1].set_title('Zone Distribution among T-cell Interactors')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved T-cell fraction comparison to {output_path}")
    
    plt.close()


def plot_violin_with_stats(
    data: pd.DataFrame,
    features: List[str],
    group_col: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    n_cols: int = 3
) -> Dict:
    """Plot multiple violin plots with statistical annotations
    
    Args:
        data: DataFrame with features and group column
        features: List of feature columns to plot
        group_col: Column with group labels
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        n_cols: Number of columns in subplot grid
        
    Returns:
        Dictionary with statistical test results for each feature
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from scipy import stats
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    # Filter to features that exist in data
    features = [f for f in features if f in data.columns]
    if len(features) == 0:
        logger.warning("No valid features to plot")
        return {}
    
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    results = {}
    groups = data[group_col].unique().tolist()
    
    for idx, feature in enumerate(features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        plot_data = data[[feature, group_col]].dropna()
        
        sns.violinplot(
            data=plot_data,
            x=group_col,
            y=feature,
            ax=ax,
            inner='box'
        )
        
        # Compute statistics for pairs
        if len(groups) == 2:
            g1_data = plot_data[plot_data[group_col] == groups[0]][feature]
            g2_data = plot_data[plot_data[group_col] == groups[1]][feature]
            
            stat, pval = stats.ttest_ind(g1_data, g2_data, equal_var=False)
            results[feature] = {'statistic': stat, 'pvalue': pval}
            
            # Add p-value annotation
            if pval < 0.0001:
                pval_text = '****'
            elif pval < 0.001:
                pval_text = '***'
            elif pval < 0.01:
                pval_text = '**'
            elif pval < 0.05:
                pval_text = '*'
            else:
                pval_text = 'ns'
            
            y_max = plot_data[feature].max()
            y_range = plot_data[feature].max() - plot_data[feature].min()
            
            ax.plot([0, 0, 1, 1], 
                    [y_max + 0.05*y_range, y_max + 0.1*y_range, 
                     y_max + 0.1*y_range, y_max + 0.05*y_range], 
                    'k-', lw=1)
            ax.text(0.5, y_max + 0.12*y_range, pval_text, ha='center', fontsize=10)
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(feature, fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    if title:
        plt.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved violin plots to {output_path}")
    
    plt.close()
    
    return results


def plot_tcell_distance_spatial(
    data: pd.DataFrame,
    x_col: str = 'spat_centroid_x',
    y_col: str = 'spat_centroid_y',
    cell_type_col: str = 'cell_type',
    output_path: Optional[str] = None,
    title: str = "T-cell Distance Analysis",
    figsize: Tuple[int, int] = (36, 12),
    point_size: float = 10
) -> None:
    """Plot 3-panel T-cell distance visualization (like notebook)
    
    Panel 1: Cell types (DZ, LZ, T-cells)
    Panel 2: Mean distance to T-cells (inferno colormap)
    Panel 3: Min distance to T-cells (inferno colormap)
    
    Args:
        data: DataFrame with cell data, coordinates, and T-cell distances
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        cell_type_col: Column with cell type labels
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Point size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Cell types
    for cell_type in data[cell_type_col].unique():
        mask = data[cell_type_col] == cell_type
        color = CELL_TYPE_COLORS.get(cell_type, '#95A5A6')
        axes[0].scatter(
            data.loc[mask, x_col],
            data.loc[mask, y_col],
            c=color,
            s=point_size,
            alpha=0.7,
            label=cell_type
        )
    axes[0].set_title('Cell Types')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].invert_yaxis()
    axes[0].set_aspect('equal')
    
    # Panel 2: Mean distance to T-cells
    if 'tcell_mean_distance' in data.columns:
        scatter1 = axes[1].scatter(
            data[x_col],
            data[y_col],
            c=data['tcell_mean_distance'],
            cmap='inferno',
            s=point_size,
            alpha=0.7
        )
        plt.colorbar(scatter1, ax=axes[1], label='Mean Distance')
        axes[1].set_title('Mean Distance to T-cells')
    else:
        axes[1].text(0.5, 0.5, 'Distance data\nnot available', 
                     ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')
    
    # Panel 3: Min distance to T-cells
    if 'tcell_min_distance' in data.columns:
        scatter2 = axes[2].scatter(
            data[x_col],
            data[y_col],
            c=data['tcell_min_distance'],
            cmap='inferno',
            s=point_size,
            alpha=0.7
        )
        plt.colorbar(scatter2, ax=axes[2], label='Min Distance')
        axes[2].set_title('Min Distance to T-cells')
    else:
        axes[2].text(0.5, 0.5, 'Distance data\nnot available', 
                     ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    axes[2].invert_yaxis()
    axes[2].set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved T-cell distance plot to {output_path}")
    
    plt.close()


def plot_correlation_lm(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    output_path: Optional[str] = None,
    title: str = "Correlation",
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """Plot regression line plot for correlation (like notebook lmplot)
    
    Args:
        data: DataFrame with x and y columns
        x_col: X column name (e.g., 'tcell_mean_distance')
        y_col: Y column name (e.g., 'min_intensity')
        hue_col: Optional hue column for different groups
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    if hue_col and hue_col in data.columns:
        hue_order = ['all B-cells', 'LZ B-cells', 'DZ B-cells']
        palette = {'all B-cells': 'k', 'LZ B-cells': 'tab:blue', 'DZ B-cells': 'tab:red'}
        
        g = sns.lmplot(
            data=data,
            x=x_col,
            y=y_col,
            scatter=False,
            hue=hue_col,
            hue_order=[h for h in hue_order if h in data[hue_col].unique()],
            palette={k: v for k, v in palette.items() if k in data[hue_col].unique()},
            height=figsize[1],
            aspect=figsize[0]/figsize[1]
        )
    else:
        g = sns.lmplot(
            data=data,
            x=x_col,
            y=y_col,
            scatter=False,
            height=figsize[1],
            aspect=figsize[0]/figsize[1]
        )
    
    plt.title(title)
    sns.despine(offset=2, trim=True)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved correlation plot to {output_path}")
    
    plt.close()


def plot_boundary_analysis_per_image(
    data: pd.DataFrame,
    x_col: str = 'centroid-1',
    y_col: str = 'centroid-0',
    output_path: Optional[str] = None,
    title: str = "Boundary Analysis",
    figsize: Tuple[int, int] = (20, 6),
    point_size: float = 15
) -> None:
    """Plot 3-panel boundary analysis for a single image (like notebook)
    
    Panel 1: Cell types (DZ vs LZ B-cells)
    Panel 2: Frequency-based distance to border (heatmap)
    Panel 3: Border proximity classification (close vs distant)
    
    Args:
        data: DataFrame with boundary analysis results for one image
        x_col: Column for x coordinate
        y_col: Column for y coordinate
        output_path: Path to save figure
        title: Plot title
        figsize: Figure size
        point_size: Point size
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import seaborn as sns
        from matplotlib.colors import Normalize
    except ImportError:
        logger.error("Required packages not installed")
        raise
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Cell types
    for cell_type in data['cell_type'].unique():
        mask = data['cell_type'] == cell_type
        color = CELL_TYPE_COLORS.get(cell_type, '#95A5A6')
        axes[0].scatter(
            data.loc[mask, x_col],
            data.loc[mask, y_col],
            c=color,
            s=point_size,
            alpha=0.7,
            label=cell_type
        )
    axes[0].set_title('Cell Types')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[0].invert_yaxis()
    axes[0].set_aspect('equal')
    
    # Panel 2: Frequency-based distance to border (heatmap)
    if 'frequency_based_distance_to_border' in data.columns:
        scatter = axes[1].scatter(
            data[x_col],
            data[y_col],
            c=data['frequency_based_distance_to_border'],
            cmap='inferno',
            s=point_size,
            alpha=0.7
        )
        # Add colorbar
        norm = Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=axes[1], label='Distance to Border')
        axes[1].set_title('Frequency-based Distance to Border')
    else:
        axes[1].text(0.5, 0.5, 'Distance data\nnot available', 
                     ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[1].invert_yaxis()
    axes[1].set_aspect('equal')
    
    # Panel 3: Border proximity classification
    if 'border_proximity' in data.columns:
        proximity_colors = {'close': '#F1C40F', 'distant': '#7F8C8D'}
        for prox in ['close', 'distant']:
            if prox in data['border_proximity'].values:
                mask = data['border_proximity'] == prox
                axes[2].scatter(
                    data.loc[mask, x_col],
                    data.loc[mask, y_col],
                    c=proximity_colors[prox],
                    s=point_size,
                    alpha=0.7,
                    label=prox
                )
        axes[2].legend(loc='upper right', fontsize=8)
        axes[2].set_title('Border Proximity')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')
    axes[2].invert_yaxis()
    axes[2].set_aspect('equal')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved boundary analysis plot to {output_path}")
    
    plt.close()


def plot_feature_importance_colored(
    importance: np.ndarray,
    feature_names: List[str],
    output_path: Optional[str] = None,
    feature_color_dict: Optional[Dict[str, str]] = None,
    title: str = "Feature Importance",
    n_features: int = 20,
    figsize: Tuple[int, int] = (10, 8)
) -> pd.DataFrame:
    """Plot feature importance with custom coloring
    
    Args:
        importance: Array of importance values
        feature_names: List of feature names
        output_path: Path to save figure
        feature_color_dict: Dict mapping feature names to colors
        title: Plot title
        n_features: Number of top features to show
        figsize: Figure size
        
    Returns:
        DataFrame with sorted feature importance
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        logger.error("matplotlib not installed")
        raise
    
    # Create DataFrame and sort
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top features
    top_features = imp_df.head(n_features)
    
    # Get colors
    if feature_color_dict:
        colors = [feature_color_dict.get(f, '#3498DB') for f in top_features['feature']]
    else:
        colors = '#3498DB'
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        range(len(top_features)),
        top_features['importance'],
        color=colors
    )
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path}")
    
    plt.close()
    
    return imp_df

