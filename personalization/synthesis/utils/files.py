"""
Shared utility functions for file operations across the cupid package.

This module provides common functionality for:
- JSON file loading and saving with consistent formatting
- Directory creation and management
- Error handling for file operations
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def load_json(path: str, default: Any = None) -> Any:
    """
    Load JSON data from file if it exists, otherwise return default value.
    
    Args:
        path: Path to the JSON file
        default: Default value to return if file doesn't exist or can't be loaded
        
    Returns:
        Loaded JSON data or default value
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load JSON from {path}: {e}")
            return default
    return default

def save_json(data_or_path: Any, output_dir_or_data: Any = None, filename: str = None, indent: int = 4) -> bool:
    """
    Save data to JSON file with pretty formatting.

    Supports two calling conventions:
    1. save_json(path, data, indent=4) - original signature
    2. save_json(data, output_dir, filename) - new signature

    Args:
        data_or_path: Either the data to save OR the path to save to
        output_dir_or_data: Either the output directory OR the data to save
        filename: Optional filename (used with new signature)
        indent: Number of spaces for indentation

    Returns:
        True if successful, False otherwise
    """
    try:
        # Detect which calling convention is being used
        if filename is not None:
            # New signature: save_json(data, output_dir, filename)
            data = data_or_path
            output_dir = output_dir_or_data
            if not isinstance(output_dir, str):
                raise ValueError(f"output_dir must be a string, got {type(output_dir)}: {output_dir}")
            path = os.path.join(output_dir, filename)
        elif isinstance(data_or_path, str) and output_dir_or_data is not None:
            # Old signature: save_json(path, data, indent)
            path = data_or_path
            data = output_dir_or_data
        else:
            # Invalid arguments
            raise ValueError(f"Invalid arguments to save_json: data_or_path type={type(data_or_path)}, output_dir_or_data type={type(output_dir_or_data)}, filename={filename}")

        # Create directory if it doesn't exist
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=indent)
        return True
    except (IOError, TypeError, ValueError) as e:
        logger.error(f"Failed to save JSON: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def ensure_directory(path: str) -> bool:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create

    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def get_persona_dir(output_dir: str, persona_id: int) -> str:
    """
    Get the directory path for a specific persona.

    Args:
        output_dir: Base output directory
        persona_id: ID of the persona

    Returns:
        Path to the persona directory
    """
    return os.path.join(output_dir, f"persona_{persona_id}")


def save_per_persona(data_by_persona: Dict[int, List], output_dir: str, filename: str, indent: int = 4) -> bool:
    """
    Save data organized by persona into separate persona directories.

    Args:
        data_by_persona: Dictionary mapping persona_id to list of data
        output_dir: Base output directory
        filename: Filename to use for each persona's data
        indent: JSON indentation

    Returns:
        True if all saves successful, False otherwise
    """
    all_successful = True
    for persona_id, persona_data in data_by_persona.items():
        persona_dir = get_persona_dir(output_dir, persona_id)
        ensure_directory(persona_dir)
        success = save_json(persona_data, persona_dir, filename, indent=indent)
        if not success:
            all_successful = False
            logger.error(f"Failed to save {filename} for persona {persona_id}")
    return all_successful 