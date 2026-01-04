# -*- coding: utf-8 -*-
"""
Spatial graph analysis for germinal center microenvironment.

This module provides graph-based analysis of cell spatial relationships,
including:
- Cell interaction graph construction
- Graph-based features (centrality, clustering)
- Spatial autocorrelation (Moran's I)
- Voronoi tessellation features

Biological rationale:
- GC organization is network-based (T-B interactions, B-B clustering)
- Graph metrics capture local microenvironment context
- Spatial autocorrelation reveals environmental influence on chromatin
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import KDTree, Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def build_cell_interaction_graph(
    spatial_coords: pd.DataFrame,
    cell_types: Optional[pd.Series] = None,
    k_neighbors: int = 10,
    distance_threshold: float = 50.0,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1')
) -> Dict:
    """
    Build spatial graph where nodes = cells, edges = spatial proximity.
    
    Biological rationale:
    - GC organization is network-based (T-B interactions, B-B clustering)
    - Graph metrics capture local microenvironment
    - Enables community detection (identify GC subregions)
    
    Args:
        spatial_coords: DataFrame with spatial coordinates
        cell_types: Optional series with cell type labels
        k_neighbors: Number of nearest neighbors for KNN graph
        distance_threshold: Maximum distance for edge creation (microns)
        coord_cols: Column names for x,y coordinates
        
    Returns:
        Dict with graph data and computed features
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed. Install with: pip install networkx")
        return {}
    
    # Extract coordinates
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < 2:
        logger.warning("Need at least 2 cells to build graph")
        return {}
    
    # Build KD-tree for efficient neighbor queries
    tree = KDTree(coords)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with cell type attributes
    for i in range(n_cells):
        node_attrs = {
            'x': coords[i, 0],
            'y': coords[i, 1]
        }
        if cell_types is not None:
            node_attrs['cell_type'] = cell_types.iloc[i]
        G.add_node(i, **node_attrs)
    
    # Add edges based on k-nearest neighbors within distance threshold
    k_query = min(k_neighbors + 1, n_cells)  # +1 because query includes self
    distances, indices = tree.query(coords, k=k_query)
    
    for i in range(n_cells):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):  # Skip self
            if dist <= distance_threshold:
                G.add_edge(i, j, weight=dist)
    
    # Compute graph features
    graph_features = _compute_graph_features(G, cell_types)
    
    return {
        'graph': G,
        'features': graph_features,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges()
    }


def _compute_graph_features(G, cell_types: Optional[pd.Series] = None) -> pd.DataFrame:
    """Compute graph-based features for each node."""
    try:
        import networkx as nx
    except ImportError:
        return pd.DataFrame()
    
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return pd.DataFrame()
    
    features = {}
    
    # Degree centrality (number of neighbors normalized)
    degree_centrality = nx.degree_centrality(G)
    features['degree_centrality'] = [degree_centrality.get(i, 0) for i in range(n_nodes)]
    
    # Raw degree (number of neighbors)
    features['degree'] = [G.degree(i) for i in range(n_nodes)]
    
    # Clustering coefficient (local density/cliquishness)
    clustering = nx.clustering(G)
    features['clustering_coefficient'] = [clustering.get(i, 0) for i in range(n_nodes)]
    
    # Betweenness centrality (bridge cells between zones)
    # Only compute for smaller graphs due to computational cost
    if n_nodes <= 1000:
        betweenness = nx.betweenness_centrality(G)
        features['betweenness_centrality'] = [betweenness.get(i, 0) for i in range(n_nodes)]
    else:
        # Use approximate betweenness for large graphs
        betweenness = nx.betweenness_centrality(G, k=min(100, n_nodes))
        features['betweenness_centrality'] = [betweenness.get(i, 0) for i in range(n_nodes)]
    
    # PageRank (influence in network)
    try:
        pagerank = nx.pagerank(G)
        features['pagerank'] = [pagerank.get(i, 0) for i in range(n_nodes)]
    except nx.PowerIterationFailedConvergence:
        features['pagerank'] = [1.0 / n_nodes] * n_nodes
    
    # Average neighbor degree
    avg_neighbor_degree = nx.average_neighbor_degree(G)
    features['avg_neighbor_degree'] = [avg_neighbor_degree.get(i, 0) for i in range(n_nodes)]
    
    # If cell types provided, compute homophily metrics
    if cell_types is not None and len(cell_types) == n_nodes:
        same_type_neighbors = []
        for i in range(n_nodes):
            neighbors = list(G.neighbors(i))
            if len(neighbors) == 0:
                same_type_neighbors.append(0.0)
            else:
                same_count = sum(1 for n in neighbors if cell_types.iloc[n] == cell_types.iloc[i])
                same_type_neighbors.append(same_count / len(neighbors))
        features['same_type_neighbor_fraction'] = same_type_neighbors
    
    return pd.DataFrame(features)


def compute_spatial_autocorrelation(
    features: pd.DataFrame,
    spatial_coords: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1'),
    k_neighbors: int = 8
) -> Dict[str, Dict]:
    """
    Compute Moran's I spatial autocorrelation for chromatin features.
    
    Tests if chromatin features cluster spatially.
    
    Biological insight:
    - If chromatin features show spatial clustering, suggests
      microenvironmental influence (signaling gradients, cell-cell contact)
    - Distinguishes cell-intrinsic vs. environment-driven differences
    
    Args:
        features: DataFrame with feature values
        spatial_coords: DataFrame with spatial coordinates
        feature_names: List of feature columns to test (default: all numeric)
        coord_cols: Column names for x,y coordinates
        k_neighbors: Number of neighbors for spatial weights
        
    Returns:
        Dict mapping feature names to Moran's I statistics
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n = len(coords)
    
    if n < 5:
        logger.warning("Need at least 5 observations for Moran's I")
        return {}
    
    # Build spatial weights matrix (k-nearest neighbors)
    tree = KDTree(coords)
    k_query = min(k_neighbors + 1, n)
    _, indices = tree.query(coords, k=k_query)
    
    # Create binary weights matrix
    W = np.zeros((n, n))
    for i in range(n):
        for j in indices[i, 1:]:  # Skip self
            W[i, j] = 1
    
    # Row-standardize weights
    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = W / row_sums[:, np.newaxis]
    
    # Determine features to analyze
    if feature_names is None:
        feature_names = features.select_dtypes(include=[np.number]).columns.tolist()
    
    results = {}
    for feat_name in feature_names:
        if feat_name not in features.columns:
            continue
            
        x = features[feat_name].values
        
        # Handle missing values
        valid_mask = ~np.isnan(x)
        if np.sum(valid_mask) < 5:
            continue
        
        # Compute Moran's I
        moran_result = _compute_morans_i(x, W, valid_mask)
        results[feat_name] = moran_result
    
    return results


def _compute_morans_i(x: np.ndarray, W: np.ndarray, valid_mask: np.ndarray) -> Dict:
    """Compute Moran's I statistic with significance testing."""
    # Use only valid observations
    x_valid = x[valid_mask]
    W_valid = W[np.ix_(valid_mask, valid_mask)]
    
    n = len(x_valid)
    if n < 3:
        return {'I': np.nan, 'p_value': np.nan, 'z_score': np.nan}
    
    # Center the variable
    x_centered = x_valid - np.mean(x_valid)
    
    # Compute Moran's I
    numerator = n * np.sum(W_valid * np.outer(x_centered, x_centered))
    denominator = np.sum(W_valid) * np.sum(x_centered ** 2)
    
    if denominator == 0:
        return {'I': np.nan, 'p_value': np.nan, 'z_score': np.nan}
    
    I = numerator / denominator
    
    # Expected value under null hypothesis
    E_I = -1.0 / (n - 1)
    
    # Variance under normality assumption (simplified)
    S0 = np.sum(W_valid)
    S1 = 0.5 * np.sum((W_valid + W_valid.T) ** 2)
    S2 = np.sum((W_valid.sum(axis=0) + W_valid.sum(axis=1)) ** 2)
    
    n2 = n * n
    var_I = (n2 * S1 - n * S2 + 3 * S0 * S0) / ((n2 - 1) * S0 * S0) - E_I * E_I
    
    if var_I <= 0:
        return {'I': I, 'E_I': E_I, 'p_value': np.nan, 'z_score': np.nan}
    
    # Z-score and p-value
    z_score = (I - E_I) / np.sqrt(var_I)
    
    # Two-tailed p-value
    from scipy import stats
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return {
        'I': I,
        'E_I': E_I,
        'var_I': var_I,
        'z_score': z_score,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': 'clustered' if I > E_I else 'dispersed'
    }


def compute_voronoi_features(
    spatial_coords: pd.DataFrame,
    cell_types: Optional[pd.Series] = None,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1'),
    image_bounds: Optional[Tuple[float, float, float, float]] = None
) -> pd.DataFrame:
    """
    Compute Voronoi tessellation features for each nucleus.
    
    Features:
    - Voronoi cell area (local cell density)
    - Number of neighbors (coordination number)
    - Voronoi cell perimeter
    - Fraction of neighbors that are same type
    
    Biological relevance:
    - DZ cells are more densely packed (proliferative)
    - LZ cells may have larger Voronoi cells (less dense)
    
    Args:
        spatial_coords: DataFrame with spatial coordinates
        cell_types: Optional series with cell type labels
        coord_cols: Column names for x,y coordinates
        image_bounds: Optional (xmin, xmax, ymin, ymax) for boundary handling
        
    Returns:
        DataFrame with Voronoi features per cell
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < 4:
        logger.warning("Need at least 4 cells for Voronoi tessellation")
        return pd.DataFrame()
    
    # Add boundary points to handle edge cells
    if image_bounds is not None:
        xmin, xmax, ymin, ymax = image_bounds
    else:
        # Estimate bounds from data
        margin = 50
        xmin = coords[:, 0].min() - margin
        xmax = coords[:, 0].max() + margin
        ymin = coords[:, 1].min() - margin
        ymax = coords[:, 1].max() + margin
    
    # Add mirror points at boundaries for bounded Voronoi
    boundary_points = np.array([
        [xmin - 100, ymin - 100],
        [xmax + 100, ymin - 100],
        [xmin - 100, ymax + 100],
        [xmax + 100, ymax + 100]
    ])
    
    coords_extended = np.vstack([coords, boundary_points])
    
    try:
        vor = Voronoi(coords_extended)
    except Exception as e:
        logger.warning(f"Voronoi tessellation failed: {e}")
        return pd.DataFrame()
    
    features = {
        'voronoi_area': [],
        'voronoi_perimeter': [],
        'voronoi_n_vertices': [],
        'voronoi_n_neighbors': [],
        'voronoi_compactness': []
    }
    
    if cell_types is not None:
        features['voronoi_same_type_neighbor_fraction'] = []
    
    # Compute features for original cells only
    for i in range(n_cells):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        # Check if region is valid (no -1 vertices, not empty)
        if -1 in region or len(region) == 0:
            for key in features:
                features[key].append(np.nan)
            continue
        
        # Get region vertices
        vertices = vor.vertices[region]
        
        # Compute area using shoelace formula
        n = len(vertices)
        area = 0.5 * abs(sum(vertices[i, 0] * vertices[(i + 1) % n, 1] -
                             vertices[(i + 1) % n, 0] * vertices[i, 1]
                             for i in range(n)))
        
        # Compute perimeter
        perimeter = sum(np.linalg.norm(vertices[(i + 1) % n] - vertices[i])
                       for i in range(n))
        
        # Compactness (isoperimetric quotient)
        compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Find Voronoi neighbors (cells sharing an edge)
        neighbors = []
        ridge_points = np.array(vor.ridge_points)
        for ridge_i, (p1, p2) in enumerate(ridge_points):
            if p1 == i and p2 < n_cells:
                neighbors.append(p2)
            elif p2 == i and p1 < n_cells:
                neighbors.append(p1)
        
        features['voronoi_area'].append(area)
        features['voronoi_perimeter'].append(perimeter)
        features['voronoi_n_vertices'].append(len(vertices))
        features['voronoi_n_neighbors'].append(len(neighbors))
        features['voronoi_compactness'].append(compactness)
        
        # Same-type neighbor fraction
        if cell_types is not None:
            if len(neighbors) > 0:
                same_type = sum(1 for n in neighbors if cell_types.iloc[n] == cell_types.iloc[i])
                features['voronoi_same_type_neighbor_fraction'].append(same_type / len(neighbors))
            else:
                features['voronoi_same_type_neighbor_fraction'].append(np.nan)
    
    return pd.DataFrame(features)


def compute_local_cell_density(
    spatial_coords: pd.DataFrame,
    radii: List[float] = [25, 50, 100],
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1')
) -> pd.DataFrame:
    """
    Compute local cell density at multiple radii.
    
    Args:
        spatial_coords: DataFrame with spatial coordinates
        radii: List of radii for density computation (in pixels/microns)
        coord_cols: Column names for x,y coordinates
        
    Returns:
        DataFrame with density features at each radius
    """
    coords = spatial_coords[[coord_cols[0], coord_cols[1]]].values
    n_cells = len(coords)
    
    if n_cells < 2:
        return pd.DataFrame()
    
    tree = KDTree(coords)
    
    features = {}
    for r in radii:
        # Count neighbors within radius
        counts = tree.query_ball_point(coords, r, return_length=True)
        # Subtract 1 to exclude self
        counts = np.array(counts) - 1
        # Normalize by area
        density = counts / (np.pi * r * r)
        features[f'density_r{int(r)}'] = density
        features[f'neighbor_count_r{int(r)}'] = counts
    
    return pd.DataFrame(features)


def extract_all_spatial_graph_features(
    spatial_coords: pd.DataFrame,
    cell_types: Optional[pd.Series] = None,
    features: Optional[pd.DataFrame] = None,
    coord_cols: Tuple[str, str] = ('centroid-0', 'centroid-1'),
    k_neighbors: int = 10,
    distance_threshold: float = 50.0,
    density_radii: List[float] = [25, 50, 100],
    compute_autocorrelation: bool = True,
    autocorr_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract all spatial and graph-based features.
    
    This is the main entry point for spatial analysis, combining:
    - Graph features (centrality, clustering)
    - Voronoi features (area, neighbors)
    - Local density at multiple scales
    - Spatial autocorrelation (optional)
    
    Args:
        spatial_coords: DataFrame with spatial coordinates
        cell_types: Optional series with cell type labels
        features: Optional DataFrame with features for autocorrelation
        coord_cols: Column names for x,y coordinates
        k_neighbors: Number of neighbors for graph construction
        distance_threshold: Maximum edge distance for graph
        density_radii: Radii for density computation
        compute_autocorrelation: Whether to compute Moran's I
        autocorr_features: Feature names for autocorrelation (default: select key features)
        
    Returns:
        DataFrame with all spatial features
    """
    all_features = pd.DataFrame()
    
    # Graph features
    logger.info("Computing graph-based features...")
    graph_result = build_cell_interaction_graph(
        spatial_coords, cell_types, k_neighbors, distance_threshold, coord_cols
    )
    if 'features' in graph_result:
        all_features = pd.concat([all_features, graph_result['features']], axis=1)
    
    # Voronoi features
    logger.info("Computing Voronoi features...")
    voronoi_feat = compute_voronoi_features(spatial_coords, cell_types, coord_cols)
    if not voronoi_feat.empty:
        all_features = pd.concat([all_features, voronoi_feat], axis=1)
    
    # Local density
    logger.info("Computing local density features...")
    density_feat = compute_local_cell_density(spatial_coords, density_radii, coord_cols)
    if not density_feat.empty:
        all_features = pd.concat([all_features, density_feat], axis=1)
    
    # Spatial autocorrelation
    if compute_autocorrelation and features is not None:
        logger.info("Computing spatial autocorrelation...")
        
        if autocorr_features is None:
            # Select key chromatin features for autocorrelation
            autocorr_features = [c for c in features.columns 
                               if any(kw in c.lower() for kw in 
                                     ['area', 'entropy', 'homogeneity', 'contrast', 'intensity'])][:10]
        
        autocorr_results = compute_spatial_autocorrelation(
            features, spatial_coords, autocorr_features, coord_cols
        )
        
        # Add autocorrelation summary as features
        for feat_name, result in autocorr_results.items():
            all_features[f'{feat_name}_morans_i'] = result.get('I', np.nan)
    
    return all_features

