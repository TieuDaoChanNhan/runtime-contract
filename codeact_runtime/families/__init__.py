"""Task family generators."""

from .knapsack import sample_knapsack_instance
from .navigation import sample_navigation_instance
from .rule_diagnosis import sample_rule_diagnosis_instance

__all__ = [
    "sample_knapsack_instance",
    "sample_navigation_instance",
    "sample_rule_diagnosis_instance",
]
