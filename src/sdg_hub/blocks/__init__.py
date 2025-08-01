"""Block implementations for SDG Hub.

This package provides various block implementations for data generation, processing, and transformation.
"""

# Local
from ..registry import BlockRegistry
from .deprecated_blocks import (
    CombineColumnsBlock,
    DuplicateColumns,
    FilterByValueBlock,
    FlattenColumnsBlock,
    LLMBlock,
    RenameColumns,
    SamplePopulatorBlock,
    SelectorBlock,
    SetToMajorityValue,
)
from .filtering import ColumnValueFilterBlock
from .llm import LLMChatBlock, PromptBuilderBlock, TextParserBlock
from .transform import (
    DuplicateColumnsBlock,
    IndexBasedMapperBlock,
    MeltColumnsBlock,
    RenameColumnsBlock,
    TextConcatBlock,
    UniformColumnValueSetter,
)
# All blocks moved to deprecated_blocks or transform modules

__all__ = [
    "ColumnValueFilterBlock",
    "DuplicateColumnsBlock",
    "IndexBasedMapperBlock",
    "MeltColumnsBlock",
    "RenameColumnsBlock",
    "TextConcatBlock",
    "UniformColumnValueSetter",
    "CombineColumnsBlock",  # Deprecated
    "DuplicateColumns",  # Deprecated
    "FilterByValueBlock",  # Deprecated
    "FlattenColumnsBlock",  # Deprecated
    "RenameColumns",  # Deprecated
    "SamplePopulatorBlock",  # Deprecated
    "SelectorBlock",  # Deprecated
    "SetToMajorityValue",  # Deprecated
    "LLMBlock",  # Deprecated
    "LLMChatBlock",
    "TextParserBlock",
    "BlockRegistry",
    "PromptBuilderBlock",
]
