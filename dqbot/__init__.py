"""
Data Quality Bot - Configurable data quality analysis tool
"""

__version__ = "1.0.0"
__author__ = "Varshitha Venkatesh"

from .core import generate_report, get_default_config, load_config

__all__ = ["generate_report", "get_default_config", "load_config"]
