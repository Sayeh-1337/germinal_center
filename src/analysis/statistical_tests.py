"""Statistical tests for germinal center analysis"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from scipy import stats
import logging
import warnings

logger = logging.getLogger(__name__)


def welch_ttest(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
    """Perform Welch's t-test (unequal variance t-test)
    
    Args:
        group1: First group values
        group2: Second group values
        
    Returns:
        Tuple of (t-statistic, p-value)
    """
    return stats.ttest_ind(group1, group2, equal_var=False)


def permutation_test_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42,
    method: str = 'pearson'
) -> Tuple[float, float]:
    """Permutation test for correlation significance
    
    Args:
        x: First variable
        y: Second variable
        n_permutations: Number of permutations
        random_state: Random seed
        method: 'pearson' or 'spearman'
        
    Returns:
        Tuple of (correlation, p-value)
    """
    np.random.seed(random_state)
    
    # Compute observed correlation
    if method == 'pearson':
        observed_corr, _ = stats.pearsonr(x, y)
    else:
        observed_corr, _ = stats.spearmanr(x, y)
    
    # Permutation test
    perm_corrs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_y = np.random.permutation(y)
        if method == 'pearson':
            perm_corrs[i], _ = stats.pearsonr(x, perm_y)
        else:
            perm_corrs[i], _ = stats.spearmanr(x, perm_y)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(observed_corr))
    
    # Ensure minimum p-value
    if p_value == 0:
        p_value = 1 / (n_permutations + 1)
    
    return observed_corr, p_value


def fdr_correction(p_values: np.ndarray, method: str = 'fdr_bh') -> np.ndarray:
    """Apply FDR correction to p-values
    
    Args:
        p_values: Array of p-values
        method: Correction method ('fdr_bh' for Benjamini-Hochberg)
        
    Returns:
        Array of adjusted p-values
    """
    from statsmodels.stats.multitest import multipletests
    
    _, adjusted, _, _ = multipletests(p_values, method=method)
    return adjusted


def find_markers(
    data: pd.DataFrame,
    labels: Union[pd.Series, np.ndarray],
    test: str = 'welch',
    alpha: float = 0.05,
    min_fold_change: float = 0.1
) -> pd.DataFrame:
    """Find marker features for each class using differential analysis
    
    Args:
        data: Feature DataFrame
        labels: Class labels
        test: Statistical test ('welch', 'mannwhitneyu')
        alpha: Significance threshold after FDR correction
        min_fold_change: Minimum absolute fold change
        
    Returns:
        DataFrame with marker analysis results
    """
    from tqdm import tqdm
    
    if isinstance(labels, pd.Series):
        labels = labels.values
    
    unique_labels = np.unique(labels)
    results = []
    
    logger.info(f"Running marker screen for {len(unique_labels)} groups...")
    
    for label in tqdm(unique_labels, desc="Run marker screen"):
        label_mask = labels == label
        other_mask = ~label_mask
        
        for col in data.columns:
            label_vals = data.loc[label_mask, col].dropna()
            other_vals = data.loc[other_mask, col].dropna()
            
            if len(label_vals) < 3 or len(other_vals) < 3:
                continue
            
            # Compute test statistic
            if test == 'welch':
                stat, pval = welch_ttest(label_vals, other_vals)
            elif test == 'mannwhitneyu':
                stat, pval = stats.mannwhitneyu(label_vals, other_vals, alternative='two-sided')
            else:
                raise ValueError(f"Unknown test: {test}")
            
            # Compute effect size (fold change)
            mean_label = label_vals.mean()
            mean_other = other_vals.mean()
            
            if mean_other != 0:
                fold_change = mean_label / mean_other
                log2_fc = np.log2(fold_change) if fold_change > 0 else np.nan
            else:
                fold_change = np.nan
                log2_fc = np.nan
            
            delta = mean_label - mean_other
            
            results.append({
                'label': label,
                'feature': col,
                'mean_in_group': mean_label,
                'mean_out_group': mean_other,
                'delta': delta,
                'abs_delta_fc': abs(delta),
                'fold_change': fold_change,
                'log2_fc': log2_fc,
                'statistic': stat,
                'pval': pval
            })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if len(results_df) > 0:
        results_df['adjusted_pval'] = fdr_correction(results_df['pval'].values)
    
    return results_df


def run_cv_classification(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    n_folds: int = 10,
    random_state: int = 42,
    balance: bool = True
) -> Dict:
    """Run cross-validated classification with Random Forest
    
    Args:
        X: Feature DataFrame
        y: Labels
        n_folds: Number of CV folds
        random_state: Random seed
        balance: Whether to balance classes
        
    Returns:
        Dictionary with CV results
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        balanced_accuracy_score, confusion_matrix,
        classification_report, roc_auc_score
    )
    
    try:
        from imblearn.under_sampling import RandomUnderSampler
        has_imblearn = True
    except ImportError:
        has_imblearn = False
        logger.warning("imbalanced-learn not installed. Skipping balancing.")
    
    np.random.seed(random_state)
    
    # Balance if requested
    if balance and has_imblearn:
        rus = RandomUnderSampler(random_state=random_state)
        X_bal, y_bal = rus.fit_resample(X, y)
        logger.info(f"Balanced dataset: {len(X_bal)} samples")
    else:
        X_bal, y_bal = X, y
    
    # Setup classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state,
        class_weight='balanced'
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Get predictions
    y_pred = cross_val_predict(clf, X_bal, y_bal, cv=cv)
    y_proba = cross_val_predict(clf, X_bal, y_bal, cv=cv, method='predict_proba')
    
    # Compute metrics
    results = {
        'balanced_accuracy': balanced_accuracy_score(y_bal, y_pred),
        'confusion_matrix': confusion_matrix(y_bal, y_pred),
        'classification_report': classification_report(y_bal, y_pred, output_dict=True),
        'predictions': y_pred,
        'probabilities': y_proba,
        'true_labels': y_bal,
        'classes': clf.classes_ if hasattr(clf, 'classes_') else np.unique(y_bal)
    }
    
    # Compute per-fold accuracies
    fold_scores = []
    for train_idx, test_idx in cv.split(X_bal, y_bal):
        clf_fold = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=random_state
        )
        clf_fold.fit(X_bal.iloc[train_idx], y_bal.iloc[train_idx] if hasattr(y_bal, 'iloc') else y_bal[train_idx])
        y_pred_fold = clf_fold.predict(X_bal.iloc[test_idx])
        y_true_fold = y_bal.iloc[test_idx] if hasattr(y_bal, 'iloc') else y_bal[test_idx]
        fold_scores.append(balanced_accuracy_score(y_true_fold, y_pred_fold))
    
    results['cv_scores'] = np.array(fold_scores)
    results['cv_mean'] = np.mean(fold_scores)
    results['cv_std'] = np.std(fold_scores)
    
    # Fit final model for feature importance
    clf.fit(X_bal, y_bal)
    results['feature_importance'] = pd.DataFrame({
        'feature': X_bal.columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"CV Balanced Accuracy: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    
    return results


def run_correlation_screen(
    data: pd.DataFrame,
    target_col: str,
    n_permutations: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """Run correlation screen between features and target variable
    
    Args:
        data: DataFrame with features and target
        target_col: Name of target column
        n_permutations: Number of permutations for significance testing
        random_state: Random seed
        
    Returns:
        DataFrame with correlation results
    """
    from tqdm import tqdm
    
    results = []
    y = data[target_col].values
    feature_cols = [c for c in data.columns if c != target_col]
    
    logger.info(f"Running correlation screen for {len(feature_cols)} features...")
    
    for col in tqdm(feature_cols, desc="Correlation screen"):
        x = data[col].values
        
        # Remove NaN
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 3:
            continue
        
        # Pearson correlation with permutation test
        try:
            pearson_r, pearson_p = permutation_test_correlation(
                x_valid, y_valid, n_permutations, random_state, 'pearson'
            )
        except:
            pearson_r, pearson_p = np.nan, np.nan
        
        # Spearman correlation
        try:
            spearman_r, spearman_p = stats.spearmanr(x_valid, y_valid)
        except:
            spearman_r, spearman_p = np.nan, np.nan
        
        results.append({
            'feature': col,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p
        })
    
    results_df = pd.DataFrame(results)
    
    # Apply FDR correction
    if len(results_df) > 0:
        results_df['pearson_p_adj'] = fdr_correction(
            results_df['pearson_p'].fillna(1).values
        )
        results_df['spearman_p_adj'] = fdr_correction(
            results_df['spearman_p'].fillna(1).values
        )
    
    return results_df.sort_values('pearson_p_adj')


def test_group_difference(
    data: pd.DataFrame,
    feature: str,
    group_col: str,
    group1: str,
    group2: str,
    test: str = 'welch'
) -> Dict:
    """Test for significant difference between two groups
    
    Args:
        data: DataFrame with feature and group columns
        feature: Feature column to test
        group_col: Grouping column
        group1: First group value
        group2: Second group value
        test: Statistical test to use
        
    Returns:
        Dictionary with test results
    """
    vals1 = data[data[group_col] == group1][feature].dropna()
    vals2 = data[data[group_col] == group2][feature].dropna()
    
    if test == 'welch':
        stat, pval = welch_ttest(vals1, vals2)
    elif test == 'mannwhitneyu':
        stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
    elif test == 'wilcoxon':
        # For paired data - need same length
        min_len = min(len(vals1), len(vals2))
        stat, pval = stats.wilcoxon(vals1[:min_len], vals2[:min_len])
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((vals1.std()**2 + vals2.std()**2) / 2)
    cohens_d = (vals1.mean() - vals2.mean()) / pooled_std if pooled_std > 0 else np.nan
    
    return {
        'feature': feature,
        'group1': group1,
        'group2': group2,
        'mean_group1': vals1.mean(),
        'mean_group2': vals2.mean(),
        'std_group1': vals1.std(),
        'std_group2': vals2.std(),
        'n_group1': len(vals1),
        'n_group2': len(vals2),
        'statistic': stat,
        'p_value': pval,
        'cohens_d': cohens_d,
        'test': test
    }

