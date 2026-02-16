"""Analysis tools for expressibility, alignment, and geometric difference."""

from src.analysis.alignment import (
    compute_all_kernel_alignments,
    compute_centered_kernel_target_alignment,
    compute_kernel_target_alignment,
)
from src.analysis.expressibility import (
    compute_all_eigenspectra,
    compute_effective_dimension,
    compute_eigenspectrum,
    compute_participation_ratio,
)
from src.analysis.geometric import (
    compute_bidirectional_geometric_difference,
    compute_geometric_difference,
    compute_pairwise_geometric_differences,
)

__all__ = [
    "compute_all_eigenspectra",
    "compute_all_kernel_alignments",
    "compute_bidirectional_geometric_difference",
    "compute_centered_kernel_target_alignment",
    "compute_effective_dimension",
    "compute_eigenspectrum",
    "compute_geometric_difference",
    "compute_kernel_target_alignment",
    "compute_pairwise_geometric_differences",
    "compute_participation_ratio",
]
