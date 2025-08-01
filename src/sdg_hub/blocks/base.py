# SPDX-License-Identifier: Apache-2.0
"""Enhanced base block implementation with standardized patterns.

This module provides a comprehensive base class for all blocks in the system,
with unified constructor patterns, column handling, and common functionality.
"""

# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from pydantic import BaseModel, ConfigDict, Field, field_validator
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Local
from ..logger_config import setup_logger
from ..utils.error_handling import (
    EmptyDatasetError,
    MissingColumnError,
    OutputColumnCollisionError,
)

logger = setup_logger(__name__)
console = Console()


class BaseBlock(BaseModel, ABC):
    """Base class for all blocks, with standardized patterns and full Pydantic compatibility.

    This class defines a unified, configurable base for building composable data processing blocks
    that operate over HuggingFace Datasets. It supports field-based initialization, validation,
    and rich logging for inputs and outputs.

    Attributes
    ----------
    block_name : str
        Unique identifier for this block instance.
    input_cols : Union[List[str], Dict[str, Any]]
        Input columns from the dataset (string, list of strings, or mapping).
    output_cols : Union[List[str], Dict[str, Any]]
        Output columns to write to the dataset (string, list of strings, or mapping).
    """

    block_name: str = Field(
        ..., description="Unique identifier for this block instance"
    )
    input_cols: Union[str, List[str], Dict[str, Any], None] = Field(
        None, description="Input columns: str, list, or dict"
    )
    output_cols: Union[str, List[str], Dict[str, Any], None] = Field(
        None, description="Output columns: str, list, or dict"
    )

    # Allow extra config fields and complex types like Dataset
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    # Normalize input columns before model construction
    @field_validator("input_cols", mode="before")
    @classmethod
    def normalize_input_cols(cls, v):
        return BaseBlock._normalize_columns(v)

    # Normalize output columns before model construction
    @field_validator("output_cols", mode="before")
    @classmethod
    def normalize_output_cols(cls, v):
        return BaseBlock._normalize_columns(v)

    @staticmethod
    def _normalize_columns(
        cols: Optional[Union[str, List[str], Dict[str, Any]]],
    ) -> Union[List[str], Dict[str, Any]]:
        """Normalize column inputs into a standard internal format.

        Parameters
        ----------
        cols : str, list, dict, or None
            Raw column specification provided by the user.

        Returns
        -------
        Union[List[str], Dict[str, Any]]
            Cleaned and deep-copied column specification.

        Raises
        ------
        ValueError
            If the column format is unsupported.
        """
        if cols is None:
            return []
        if isinstance(cols, str):
            return [cols]
        if isinstance(cols, list):
            return cols.copy()
        if isinstance(cols, dict):
            return dict(cols)
        raise ValueError(f"Invalid column specification: {cols} (type: {type(cols)})")

    def _validate_columns(self, dataset: Dataset) -> None:
        """Check that all required input columns are present in the dataset.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset to validate against.

        Raises
        ------
        MissingColumnError
            If any expected input column is missing.
        """
        if not self.input_cols:
            return
        columns_to_check = (
            list(self.input_cols.keys())
            if isinstance(self.input_cols, dict)
            else self.input_cols
        )
        missing_columns = [
            col for col in columns_to_check if col not in dataset.column_names
        ]
        if missing_columns:
            raise MissingColumnError(
                block_name=self.block_name,
                missing_columns=missing_columns,
                available_columns=dataset.column_names,
            )

    def _validate_output_columns(self, dataset: Dataset) -> None:
        """Check that the output columns will not overwrite existing ones.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset to validate.

        Raises
        ------
        OutputColumnCollisionError
            If output columns already exist in the dataset.
        """
        if not self.output_cols:
            return
        columns_to_check = (
            list(self.output_cols.keys())
            if isinstance(self.output_cols, dict)
            else self.output_cols
        )
        collisions = [col for col in columns_to_check if col in dataset.column_names]
        if collisions:
            raise OutputColumnCollisionError(
                block_name=self.block_name,
                collision_columns=collisions,
                existing_columns=dataset.column_names,
            )

    def _validate_dataset_not_empty(self, dataset: Dataset) -> None:
        """Raise an error if the dataset is empty.

        Parameters
        ----------
        dataset : Dataset

        Raises
        ------
        EmptyDatasetError
        """
        if len(dataset) == 0:
            raise EmptyDatasetError(block_name=self.block_name)

    def _validate_dataset(self, dataset: Dataset) -> None:
        """Perform all default dataset validations."""
        self._validate_dataset_not_empty(dataset)
        self._validate_columns(dataset)
        self._validate_output_columns(dataset)

    def _validate_custom(self, dataset: Dataset) -> None:
        """Hook for subclasses to add extra validation logic."""
        pass

    def _log_input_data(self, dataset: Dataset) -> None:
        """Print a summary of the input dataset with Rich formatting."""
        row_count = len(dataset)
        columns = dataset.column_names
        content = Text()
        content.append("\U0001f4ca Processing Input Data\n", style="bold blue")
        content.append(f"Block Type: {self.__class__.__name__}\n", style="cyan")
        content.append(f"Input Rows: {row_count:,}\n", style="bold cyan")
        content.append(f"Input Columns: {len(columns)}\n", style="cyan")
        content.append(f"Column Names: {', '.join(columns)}\n", style="white")
        expected = (
            (
                ", ".join(self.output_cols.keys())
                if isinstance(self.output_cols, dict)
                else ", ".join(self.output_cols)
            )
            if self.output_cols
            else "None specified"
        )
        content.append(f"Expected Output Columns: {expected}", style="green")
        console.print(
            Panel(content, title=f"[bold]{self.block_name}[/bold]", border_style="blue")
        )

    def _log_output_data(self, input_dataset: Dataset, output_dataset: Dataset) -> None:
        """Print a Rich panel summarizing output dataset differences."""
        in_rows, out_rows = len(input_dataset), len(output_dataset)
        in_cols, out_cols = (
            set(input_dataset.column_names),
            set(output_dataset.column_names),
        )
        added_cols, removed_cols = out_cols - in_cols, in_cols - out_cols
        content = Text()
        content.append("\u2705 Processing Complete\n", style="bold green")
        content.append(f"Rows: {in_rows:,} → {out_rows:,}\n", style="cyan")
        content.append(f"Columns: {len(in_cols)} → {len(out_cols)}\n", style="cyan")
        if added_cols:
            content.append(
                f"\U0001f7e2 Added: {', '.join(sorted(added_cols))}\n", style="green"
            )
        if removed_cols:
            content.append(
                f"\U0001f534 Removed: {', '.join(sorted(removed_cols))}\n", style="red"
            )
        content.append(
            f"\U0001f4cb Final Columns: {', '.join(sorted(out_cols))}", style="white"
        )
        console.print(
            Panel(
                content,
                title=f"[bold green]{self.block_name} - Complete[/bold green]",
                border_style="green",
            )
        )

    @abstractmethod
    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Subclass method to implement data generation logic.

        Parameters
        ----------
        samples : Dataset
            Input dataset to process.

        Returns
        -------
        Dataset
            Transformed dataset with new columns or values.
        """
        pass

    def __call__(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Run the block on a dataset with full validation and logging.

        Parameters
        ----------
        samples : Dataset
            Input dataset.

        Returns
        -------
        Dataset
            Output dataset after block processing.
        """
        self._log_input_data(samples)
        self._validate_dataset(samples)
        self._validate_custom(samples)
        output_dataset = self.generate(samples, **kwargs)
        self._log_output_data(samples, output_dataset)
        return output_dataset

    def __repr__(self) -> str:
        """Compact string representation."""
        return f"{self.__class__.__name__}(name='{self.block_name}', input_cols={self.input_cols}, output_cols={self.output_cols})"

    def get_config(self) -> Dict[str, Any]:
        """Return only constructor arguments for serialization.

        Returns
        -------
        Dict[str, Any]
        """
        return self.model_dump()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseBlock":
        """Instantiate block from serialized config.

        Parameters
        ----------
        config : Dict[str, Any]

        Returns
        -------
        BaseBlock
        """
        return cls(**config)

    def get_info(self) -> Dict[str, Any]:
        """Return a high-level summary of block metadata and config.

        Returns
        -------
        Dict[str, Any]
        """
        config = self.get_config()
        config["block_type"] = self.__class__.__name__
        return config
