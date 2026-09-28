"""Ares — local multi-agent AI system."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("ares")
except PackageNotFoundError:  # running from a checkout that isn't installed
    __version__ = "0.0.0"

__all__ = ["AresSystem"]


def __getattr__(name: str):
    # Lazy so `import ares.config` doesn't boot the whole agent runtime.
    if name == "AresSystem":
        from ares.system import AresSystem
        return AresSystem
    raise AttributeError(f"module 'ares' has no attribute {name!r}")
