# SPDX-License-Identifier: Apache-2.0
"""DEPRECATED: SamplePopulatorBlock for backward compatibility.

This module provides a deprecated stub for SamplePopulatorBlock.
This block is deprecated and will be replaced with a router block.
"""

# Standard
import warnings
from typing import Any, Dict, List

# Third Party
from datasets import Dataset

# Local
from ...logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register("SamplePopulatorBlock", "deprecated")
class SamplePopulatorBlock(BaseBlock):
    """DEPRECATED: Block for populating dataset with data from configuration files.
    
    .. deprecated:: 
        This block is deprecated and will be replaced with a router block.
    """

    def __init__(
        self,
        block_name: str,
        config_paths: List[str],
        column_name: str,
        post_fix: str = "",
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        warnings.warn(
            "SamplePopulatorBlock is deprecated and will be replaced with a router block.",
            DeprecationWarning,
            stacklevel=2,
        )
        
        # Initialize with dummy values for BaseBlock validation
        super().__init__(
            block_name=block_name,
            input_cols=[column_name],
            output_cols=[column_name]
        )

    def generate(self, samples: Dataset) -> Dataset:
        """Generate method - raises error as block is deprecated."""
        raise NotImplementedError(
            "SamplePopulatorBlock is deprecated and will be replaced with a router block."
        )