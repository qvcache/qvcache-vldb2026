"""
QVCache Python bindings

The main module is 'qvcache' which is a compiled C++ extension.
"""

__version__ = "0.1.0"

try:
    # Try importing the compiled extension directly
    import qvcache
    # Import all public symbols
    from qvcache import *
except ImportError as e:
    raise ImportError(f"Could not import qvcache extension module: {e}") from e

