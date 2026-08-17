"""Kit extractor — fetches kit data directly from Supabase REST API.

No browser or cheap model needed. The kit JSON is structured and can be
walked directly to produce KitString objects.
"""

from __future__ import annotations
import re
import ssl
import urllib.request
import urllib.error
import json
import os
from shared.schemas import KitString
import shared.config as config

try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

KIT_BASE_URL = "https://search-kit-library.vercel.app/kit"
# Set CLORIS_SEARCH_KITS_REST_URL to your own Supabase REST endpoint.
SUPABASE_REST_URL = os.environ.get(
    "CLORIS_SEARCH_KITS_REST_URL",
    "https://YOUR_PROJECT.supabase.co/rest/v1/search_kits",
)


def _extract_kit_id(kit_url_or_id: str) -> str:
    """Extract the UUID kit ID from a full URL or bare ID."""
    # If it's already a UUID-like string, return as-is
    if re.match(r"^[0-9a-f-]{36}$", kit_url_or_id):
        return kit_url_or_id
    # Extract from URL path
    match = re.search(r"/kit/([0-9a-f-]{36})", kit_url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract kit ID from: {kit_url_or_id}")


def extract_kit_strings(kit_url_or_id: str) -> list[KitString]:
    """Fetch kit data from Supabase and return a list of KitString objects.

    Args:
        kit_url_or_id: Full kit URL or bare UUID kit ID.

    Returns:
        List of KitString objects with block/subblock/type metadata.
    """
    kit_id = _extract_kit_id(kit_url_or_id)

    # Fetch from Supabase REST API
    url = f"{SUPABASE_REST_URL}?select=kit_data&id=eq.{kit_id}"
    req = urllib.request.Request(url)
    req.add_header("apikey", config.SUPABASE_ANON_KEY)
    req.add_header("Authorization", f"Bearer {config.SUPABASE_ANON_KEY}")

    print(f"  Fetching kit {kit_id} from Supabase...")
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CONTEXT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch kit from Supabase: {e}") from e

    if not data:
        raise RuntimeError(f"Kit not found: {kit_id}")

    kit_data = data[0]["kit_data"]

    # Walk the JSON tree
    kit_strings: list[KitString] = []
    string_id = 0

    for block in kit_data.get("blocks", []):
        block_title = block.get("title", f"Block {block.get('number', '?')}")

        for sub_block in block.get("sub_blocks", []):
            subblock_type = sub_block.get("type", "")  # Concepts / Methods / Tools

            for cluster in sub_block.get("clusters", []):
                string_type = cluster.get("label", "")  # Recall / Precision

                for group in cluster.get("groups", []):
                    terms = group.get("terms", "")
                    if not terms:
                        continue

                    string_id += 1
                    kit_strings.append(KitString(
                        id=string_id,
                        block=block_title,
                        subblock=subblock_type,
                        string_type=string_type,
                        boolean=terms,
                    ))

    print(f"  Extracted {len(kit_strings)} Boolean strings from kit")
    return kit_strings
