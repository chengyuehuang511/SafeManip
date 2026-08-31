"""
SafeManip's own extension layer over the (unmodified) vendor `robocasa`
package -- see kitchen_ext.py's module docstring for the full rationale.
Importing this package applies the `Kitchen.get_privileged_information`
monkeypatch as a side effect; import it once before constructing any Kitchen
env.
"""
from . import kitchen_ext  # noqa: F401  (side effect: patches Kitchen)
