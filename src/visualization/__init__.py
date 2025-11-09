"""
Módulo de visualizaciones
"""
from .plots import (
    plot_survival_distribution,
    plot_survival_by_feature,
    plot_age_distribution,
    plot_fare_distribution,
    plot_correlation_heatmap,
    plot_train_test_split
)
from .tree_viz import (
    plot_decision_tree_matplotlib,
    plot_tree_depth_performance,
    plot_feature_importance_tree
)
from .metrics_viz import (
    plot_confusion_matrix,
    plot_metrics_comparison,
    plot_models_comparison,
    plot_cross_validation_scores,
    plot_feature_importance_comparison
)

__all__ = [
    'plot_survival_distribution',
    'plot_survival_by_feature',
    'plot_age_distribution',
    'plot_fare_distribution',
    'plot_correlation_heatmap',
    'plot_train_test_split',
    'plot_decision_tree_matplotlib',
    'plot_tree_depth_performance',
    'plot_feature_importance_tree',
    'plot_confusion_matrix',
    'plot_metrics_comparison',
    'plot_models_comparison',
    'plot_cross_validation_scores',
    'plot_feature_importance_comparison'
]