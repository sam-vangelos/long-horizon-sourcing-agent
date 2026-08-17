"""Package-registry hub clients for the OSS Maintainers module."""

from github.hubs.crates import CratesHubClient
from github.hubs.derive import RegistryTarget, derive_registry_targets
from github.hubs.npm import NpmHubClient

__all__ = [
    "CratesHubClient",
    "NpmHubClient",
    "RegistryTarget",
    "derive_registry_targets",
]
