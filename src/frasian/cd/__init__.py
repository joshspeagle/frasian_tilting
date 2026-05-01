"""Confidence distributions.

Public surface kept minimal; the universal constructor lands in Phase D
(`cd.from_pvalue.build_cd_from_pvalue`) and the concrete container in
Phase B (`cd.grid.GridConfidenceDistribution`).
"""

from .base import ConfidenceDistribution

__all__ = ["ConfidenceDistribution"]
