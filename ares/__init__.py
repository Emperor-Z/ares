"""Ares — local multi-agent AI system built on OpenJarvis."""

__all__ = ["AresSystem"]


def __getattr__(name: str):
    # Lazy so `import ares.config` doesn't boot the whole agent runtime.
    if name == "AresSystem":
        from ares.system import AresSystem
        return AresSystem
    raise AttributeError(f"module 'ares' has no attribute {name!r}")
