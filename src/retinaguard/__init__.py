"""RetinaGuard: DR screening triage research prototype. Not a medical device."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("retinaguard")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

DISCLAIMER = (
    "RetinaGuard is a research and portfolio prototype. It is not a medical device "
    "and must not be used for clinical decisions."
)
