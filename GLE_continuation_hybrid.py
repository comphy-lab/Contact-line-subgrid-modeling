#!/usr/bin/env python
"""
DEPRECATED: This file has been renamed to GLE_critical_ca_advanced.py

This is a compatibility wrapper that redirects to the new file.
Please update your scripts to use GLE_critical_ca_advanced.py directly.
"""

import warnings
import sys

# Issue deprecation warning
warnings.warn(
    "GLE_continuation_hybrid.py has been renamed to GLE_critical_ca_advanced.py. "
    "This compatibility wrapper will be removed in a future version. "
    "Please update your scripts to use the new filename.",
    DeprecationWarning,
    stacklevel=2
)

# Import and run the new module
from GLE_critical_ca_advanced import *

if __name__ == "__main__":
    # Execute the new module as main
    import runpy
    runpy.run_module('GLE_critical_ca_advanced', run_name='__main__')