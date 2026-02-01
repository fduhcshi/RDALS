"""
Base class for label shift estimation methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np
from torch.utils.data import Dataset

from ..config import Config


class BaseLabelShiftEstimator(ABC):
    """Abstract base class for label shift estimation methods."""
    
    @abstractmethod
    def estimate(
        self,
        source_dset: Dataset,
        target_dset: Dataset,
        p_true: np.ndarray,
        q_true: np.ndarray,
        train_downstream: bool = False,
        config: Config = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Estimate target label distribution and importance weights.
        
        Args:
            source_dset: Source domain dataset
            target_dset: Target domain dataset
            p_true: True source label distribution
            q_true: True target label distribution
            train_downstream: Whether to train downstream classifier
            config: Configuration object
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
                - q_hat: Estimated target distribution
                - w_hat: Estimated importance weights
                - w_true: True importance weights
                - time_sec: Execution time in seconds
                - acc (optional): Downstream accuracy
                - macro_f1 (optional): Downstream macro F1 score
        """
        pass
