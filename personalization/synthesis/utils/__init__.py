"""
Cupid utilities package.

This package provides common utility functions for:
- File operations (JSON loading/saving, directory management)
- Logging operations (error logging, file logging setup)
- LLM generation (API clients, generation functionality)
- Data parsing (JSON/YAML parsing utilities)
- Validation (style similarity validation)
"""

from .files import load_json, save_json, ensure_directory
from .generation import generate, generate_chat, Generator, GeneratorChat
from .parsing import parse_json, parse_yaml, json_to_yaml_str
from .logging import setup_main_logging, setup_worker_logging
from .validator import (
    StyleSimilarityValidator,
    Entry,
    load_entries_from_list,
    load_entries_from_csv,
    StyleFeatureExtractor
)

__all__ = [
    # File utilities
    'load_json', 'save_json', 'ensure_directory',

    # Generation utilities
    'generate', 'generate_chat', 'Generator', 'GeneratorChat',

    # Parsing utilities
    'parse_json', 'parse_yaml', 'json_to_yaml_str',

    # Logging utilities
    'setup_main_logging', 'setup_worker_logging',

    # Validation utilities
    'StyleSimilarityValidator', 'Entry', 'load_entries_from_list',
    'load_entries_from_csv', 'StyleFeatureExtractor',
]
