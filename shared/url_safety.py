"""URL safety validation for candidate-controlled URLs.

Prevents SSRF by blocking requests to private/reserved IPs, cloud metadata
endpoints, and dangerous URL schemes. All DNS resolution is async.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}

# Exact-match hostname blocklist (cloud metadata endpoints)
_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "metadata.goog",
}


def _is_blocked_ip(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a reason string if the IP is in a blocked range, else None."""
    if addr.is_loopback:
        return "loopback"
    if addr.is_private:
        return "private"
    if addr.is_link_local:
        return "link-local"
    if addr.is_unspecified:
        return "unspecified"
    if addr.is_reserved:
        return "reserved"
    if addr.is_multicast:
        return "multicast"
    return None


def check_ip(ip_str: str) -> tuple[bool, str]:
    """Check whether a single IP address string is safe to connect to."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False, f"invalid IP: {ip_str}"
    reason = _is_blocked_ip(addr)
    if reason:
        return False, f"{reason} address: {ip_str}"
    return True, "ok"


async def check_url(url: str) -> tuple[bool, str]:
    """Validate a URL for safe fetching. Async — resolves DNS.

    Returns (True, "ok") or (False, "reason").
    """
    if not url:
        return False, "empty URL"

    parsed = urlparse(url)

    # Scheme check
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return False, f"blocked scheme: {parsed.scheme or '(none)'}"

    hostname = parsed.hostname
    if not hostname:
        return False, "no hostname"

    # Exact hostname blocklist
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        return False, f"blocked hostname: {hostname}"

    # Raw IP literal check (before DNS)
    try:
        addr = ipaddress.ip_address(hostname)
        reason = _is_blocked_ip(addr)
        if reason:
            return False, f"{reason} address: {hostname}"
        # Valid public IP literal — allow
        return True, "ok"
    except ValueError:
        pass  # Not an IP literal, continue to DNS resolution

    # Async DNS resolution
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        loop = asyncio.get_running_loop()
        results = await loop.getaddrinfo(
            hostname, port, type=socket.SOCK_STREAM,
        )
    except socket.gaierror:
        return False, f"DNS resolution failed: {hostname}"

    if not results:
        return False, f"DNS returned no results: {hostname}"

    for family, _type, _proto, _canonname, sockaddr in results:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"invalid resolved IP: {ip_str}"
        reason = _is_blocked_ip(addr)
        if reason:
            return False, f"DNS resolved to {reason} address: {ip_str}"

    return True, "ok"
