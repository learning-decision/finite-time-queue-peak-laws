"""
qpeak — queue-peak experiment framework (IQS first).

Importing this package registers config factories for model / arrivals / policy types.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Side-effect: populate registries via subpackage imports
from qpeak import service_times as _service_times  # noqa: F401
from qpeak import arrivals as _arrivals  # noqa: F401
from qpeak import models as _models  # noqa: F401
from qpeak import policies as _policies  # noqa: F401
