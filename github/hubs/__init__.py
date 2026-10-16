"""Package-registry hub clients for the OSS Maintainers module."""

from github.hubs.crates import CratesHubClient
from github.hubs.derive import RegistryTarget, derive_registry_targets
from github.hubs.hf import HuggingFaceHubClient
from github.hubs.npm import NpmHubClient
from github.hubs.pypi import PyPiHubClient

__all__ = [
    "CratesHubClient",
    "HuggingFaceHubClient",
    "NpmHubClient",
    "PyPiHubClient",
    "RegistryTarget",
    "derive_registry_targets",
]
