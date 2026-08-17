#!/usr/bin/env python3
"""One-shot live smoke test for the GLM-5.2 shadow-judge seam (Fireworks).

The Fireworks account backing SHADOW_FACIAL_MODEL_ENABLED was suspended
(billing) at the time this seam was built, so the seam itself
(shared/config.py, shared/llm_clients.shadow_facial_llm,
shared/judger.py's shadow hooks) was built and unit-tested entirely against
mocked HTTP. This script is the live check Sam runs once billing clears (or
any time the seam needs a live sanity check) — it is intentionally NOT
covered by pytest's live path; only its pure/arg-parsing helpers are.

Three steps:

1. List models (GET /models) and grep for "glm". CAVEAT: on at least one
   Fireworks serverless key, the listing endpoint returns a misleading
   "Account suspended" error even while /chat/completions works fine —
   this looks like a control-plane permission quirk for serverless keys,
   not an actual billing/suspension problem. This step tolerates that
   specific error, prints an explanation, and continues to step 2 rather
   than treating it as fatal.
2. Send one tiny facial-shaped JSON-mode prompt to the configured slug
   (SHADOW_FACIAL_MODEL_NAME) and print the parsed decision, raw usage,
   and computed cost (via shared.llm_usage.estimate_usage_cost_usd). This
   step is the AUTHORITATIVE health check — a real chat completion is
   proof the key and model slug both work, independent of whatever step 1
   reported.
3. Cache-visibility check: send the SAME facial-shaped prompt TWICE in a
   row and print each call's raw `usage.prompt_tokens_details.cached_tokens`
   plus the resulting billable-input/cache-read cost split (mirrors
   shared.llm_usage.fireworks_shadow_usage_dict's accounting — see that
   function's docstring for the confirmed field name and the Fireworks doc
   URLs it cites). This is diagnostic, not authoritative: Fireworks'
   automatic prefix caching is best-effort, so a 0 on the second call is a
   note, not a failure — it does not affect the exit code.

Never prints the API key. Exit codes: 0 if step 2 succeeds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys


# Substrings seen in Fireworks' listing-endpoint error body on an otherwise
# healthy serverless key. Kept as a tuple (not a single string) so a second
# observed wording can be added later without touching the call site.
_SPURIOUS_LISTING_SUSPENSION_MARKERS: tuple[str, ...] = (
    "account suspended",
    "account is suspended",
)


def _looks_like_spurious_listing_suspension(error_text: str) -> bool:
    """True if an /models error looks like the known serverless-key quirk.

    Pure string check, no network — kept separate from the network call so
    it's unit-testable without mocking HTTP.
    """
    lowered = (error_text or "").lower()
    return any(marker in lowered for marker in _SPURIOUS_LISTING_SUSPENSION_MARKERS)


_FACIAL_SMOKE_SYSTEM = (
    "You are a facial-triage classifier for recruiting candidate snippets. "
    'Respond with ONLY a JSON object: {"decision": "FACIAL_YES" or '
    '"FACIAL_NO", "reason": "<one short sentence>"}. No prose outside the '
    "JSON object."
)

_FACIAL_SMOKE_USER = (
    "Name: Jordan Rivera\n"
    "Headline: Senior Machine Learning Engineer at a mid-size fintech\n"
    "Current Title: Senior ML Engineer\n"
    "Current Company: Acme Fintech\n"
    "Location: New York, NY\n"
    "Education: BS Computer Science, State University\n\n"
    "This is a synthetic smoke-test candidate, not a real person. "
    "Decide: FACIAL_YES or FACIAL_NO for a generic 'ML engineer with "
    "production experience' bar."
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Live smoke test for the GLM-5.2 (Fireworks) shadow-judge seam. "
            "Requires FIREWORKS_API_KEY in the environment/.env; makes real "
            "network calls to Fireworks."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Override the model slug (default: shared.config."
            "SHADOW_FACIAL_MODEL_NAME)."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Override the API base URL (default: shared.config."
            "SHADOW_FACIAL_BASE_URL)."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="max_tokens for the step-2 completion smoke call (default: 256).",
    )
    parser.add_argument(
        "--skip-list-models",
        action="store_true",
        help="Skip step 1 (model listing) and go straight to the completion check.",
    )
    parser.add_argument(
        "--skip-cache-check",
        action="store_true",
        help="Skip step 3 (prefix-cache visibility check).",
    )
    return parser


def _run_list_models_step(client, configured_model: str) -> None:
    print("=" * 60)
    print("Step 1: list models (GET /models)")
    print("=" * 60)
    try:
        models = client.models.list()
    except Exception as exc:  # noqa: BLE001 — this step is diagnostic, not fatal
        error_text = str(exc)
        if _looks_like_spurious_listing_suspension(error_text):
            print(
                "  [KNOWN QUIRK] The /models listing endpoint reported "
                "'account suspended' on this key. On at least one Fireworks "
                "serverless key this is a spurious control-plane response — "
                "NOT proof the account can't serve completions. Treating "
                "this as non-fatal; step 2 (the actual completion call) is "
                "the authoritative health check."
            )
        else:
            print(f"  [WARN] Listing models failed (non-fatal, continuing to step 2): {exc}")
        return

    ids = []
    for item in getattr(models, "data", None) or []:
        model_id = getattr(item, "id", None)
        if isinstance(model_id, str):
            ids.append(model_id)
    glm_ids = sorted(i for i in ids if "glm" in i.lower())
    if glm_ids:
        print(f"  Found {len(glm_ids)} GLM model slug(s):")
        for model_id in glm_ids:
            marker = "  <- configured slug" if model_id == configured_model else ""
            print(f"    - {model_id}{marker}")
        if configured_model not in glm_ids:
            print(
                f"  [NOTE] Configured slug {configured_model!r} was not in "
                "the listing above. If step 2 below also fails, update "
                "SHADOW_FACIAL_MODEL_NAME to one of the slugs listed here."
            )
    else:
        print(f"  No 'glm' model id found in {len(ids)} listed model(s).")


def _run_completion_step(client, *, model: str, max_tokens: int) -> int:
    import shared.config as config
    from shared.llm_usage import estimate_usage_cost_usd

    print("=" * 60)
    print(f"Step 2: tiny facial-shaped completion against {model!r} (authoritative)")
    print("=" * 60)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": _FACIAL_SMOKE_SYSTEM},
                {"role": "user", "content": _FACIAL_SMOKE_USER},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=60.0,
        )
    except Exception as exc:  # noqa: BLE001 — top-level CLI error surface
        message = str(exc)
        print(f"  [ERROR] Completion call failed: {message}")
        if "suspend" in message.lower() or "billing" in message.lower():
            print(
                "  This looks like a real billing/suspension error (unlike "
                "the known step-1 listing quirk) — check the Fireworks "
                "dashboard billing status before retrying."
            )
        return 1

    choice = (response.choices or [None])[0]
    raw_content = getattr(getattr(choice, "message", None), "content", "") or ""
    print(f"  Raw content: {raw_content!r}")
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        print("  [WARN] Response was not valid JSON despite JSON mode being requested.")
        parsed = None
    else:
        print(f"  Parsed decision: {parsed.get('decision')!r}")
        print(f"  Parsed reason: {parsed.get('reason')!r}")

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    print(f"  Usage: input_tokens={input_tokens} output_tokens={output_tokens}")

    cost_usd, rate_source = estimate_usage_cost_usd(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    if cost_usd is None:
        print(
            f"  Cost: unknown (rate_source={rate_source!r} — no rate row for "
            f"{model!r} in shared.llm_usage.MODEL_RATE_TABLE_USD_PER_MTOKEN)"
        )
    else:
        print(f"  Estimated cost: ${cost_usd:.6f} (rate_source={rate_source!r})")

    print()
    print("  [OK] Completion call succeeded — the key and model slug both work.")
    return 0


def _run_cache_visibility_step(client, *, model: str, max_tokens: int) -> None:
    """Step 3: exercise + surface Fireworks' automatic prefix caching.

    Sends the SAME system+user prompt twice in a row. Fireworks enables
    prefix caching by default (no request-side opt-in needed) but a hit
    requires a byte-identical prompt prefix across calls, so the second
    call here is the one worth watching for a nonzero
    ``usage.prompt_tokens_details.cached_tokens`` — the field name/shape
    confirmed against Fireworks' public API reference (see
    ``shared.llm_usage.fireworks_shadow_usage_dict``'s docstring for the
    doc URLs and the inclusive/subset convention this mirrors).

    Diagnostic only, not authoritative: caching is best-effort and may not
    hit on a cold account or a very short prompt. Never affects the
    process exit code.
    """
    from shared.llm_usage import estimate_usage_cost_usd

    print("=" * 60)
    print("Step 3: prefix-cache visibility (two identical calls)")
    print("=" * 60)

    for call_num in (1, 2):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": _FACIAL_SMOKE_SYSTEM},
                    {"role": "user", "content": _FACIAL_SMOKE_USER},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=60.0,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic step, not fatal
            print(f"  [WARN] Call {call_num} failed (non-fatal): {exc}")
            continue

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
        cached_tokens = (
            int(getattr(prompt_details, "cached_tokens", 0) or 0)
            if prompt_details is not None
            else 0
        )
        billable_input = max(prompt_tokens - cached_tokens, 0)

        print(
            f"  Call {call_num}: prompt_tokens={prompt_tokens} "
            f"cached_tokens={cached_tokens} billable_input={billable_input} "
            f"completion_tokens={completion_tokens}"
        )
        cost_usd, rate_source = estimate_usage_cost_usd(
            model=model,
            input_tokens=billable_input,
            output_tokens=completion_tokens,
            cache_read_input_tokens=cached_tokens,
        )
        if cost_usd is None:
            print(f"    Cost: unknown (rate_source={rate_source!r})")
        else:
            print(f"    Estimated cost: ${cost_usd:.6f} (rate_source={rate_source!r})")

        if call_num == 2 and cached_tokens == 0:
            print(
                "  [NOTE] Second call reported 0 cached_tokens — either this "
                "account/model hasn't warmed the cache yet, the prompt is "
                "too short to cache, or Fireworks' automatic caching just "
                "didn't hit this time. Not fatal; re-run to confirm."
            )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    import shared.config as config

    if not config.FIREWORKS_API_KEY:
        print(
            "[ERROR] FIREWORKS_API_KEY is not set (check .env / environment). "
            "Never printing key values — just confirming presence.",
            file=sys.stderr,
        )
        return 1

    model = args.model or config.SHADOW_FACIAL_MODEL_NAME
    base_url = args.base_url or config.SHADOW_FACIAL_BASE_URL

    from openai import OpenAI

    client = OpenAI(api_key=config.FIREWORKS_API_KEY, base_url=base_url, timeout=60.0)

    print(f"Base URL: {base_url}")
    print(f"Configured model slug: {model}")
    print()

    if not args.skip_list_models:
        _run_list_models_step(client, model)
        print()

    exit_code = _run_completion_step(client, model=model, max_tokens=args.max_tokens)

    if not args.skip_cache_check:
        print()
        _run_cache_visibility_step(client, model=model, max_tokens=args.max_tokens)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
