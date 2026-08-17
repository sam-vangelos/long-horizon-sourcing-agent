"""Readiness, onboarding, and Chrome lifecycle HTTP routes."""

from __future__ import annotations

from cloris import __version__
from fastapi import HTTPException

from cloris.models import (
    AcknowledgmentRequest,
    AnthropicHealthResponse,
    ChromeStatusResponse,
    CredentialUpsertRequest,
    OnboardingStatusResponse,
)

from .routing import router


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Readiness probe used by the run_app lifecycle and tests."""

    return {
        "status": "ok",
        "slice": "v0-shell-slice-1",
        "version": __version__,
    }


@router.get("/api/bootstrap")
def api_bootstrap() -> dict[str, str]:
    """Deliver the process-lifetime bearer token to the Svelte frontend.

    Exempt from BearerAuthMiddleware (the frontend needs this before it
    has a token). Safe because uvicorn binds to 127.0.0.1 only — no
    external network access is possible.
    """
    from cloris.api.auth import SESSION_TOKEN

    return {"token": SESSION_TOKEN}


@router.get("/api/chrome-status", response_model=ChromeStatusResponse)
def api_chrome_status() -> ChromeStatusResponse:
    """Phase 0 ``chrome-launcher`` slice. Read-only CDP / Chrome-profile
    health snapshot for the welcome surface's polling loop and the
    Settings "Cloris's hands" panel.

    Pure read; never spawns or kills. The welcome surface polls this
    every ~1s while the recipient is on first launch so it can
    transition from "Opening Chrome..." to "Sign into LinkedIn" the
    moment :func:`cloris.chrome_launcher.is_healthy` returns true.
    """

    from cloris.chrome_launcher import status

    snapshot = status()
    return ChromeStatusResponse(
        state=snapshot.state,  # type: ignore[arg-type]
        cdp_url=snapshot.cdp_url,
        profile_dir=snapshot.profile_dir,
        message=snapshot.message,
    )


@router.get("/api/anthropic-health", response_model=AnthropicHealthResponse)
def api_anthropic_health() -> AnthropicHealthResponse:
    """Cached provider readiness for setup and launch pre-flight."""

    from cloris.anthropic_health import probe_anthropic_health

    health = probe_anthropic_health()
    return AnthropicHealthResponse(
        state=health.state,
        message=health.message,
        checked_at=health.checked_at,
        cache_age_s=health.cache_age_s,
    )


@router.get("/api/onboarding/status", response_model=OnboardingStatusResponse)
def api_onboarding_status() -> OnboardingStatusResponse:
    """Phase 0 ``apikey-ui`` slice. Welcome-gate read endpoint.

    The frontend calls this on app mount; ``welcome_complete=False``
    means render Welcome.svelte, ``welcome_complete=True`` means
    fall through to the route table. Pure read — never writes.
    """

    from cloris.onboarding import onboarding_status

    s = onboarding_status()
    return OnboardingStatusResponse(
        welcome_complete=s.welcome_complete,
        anthropic_present=s.anthropic_present,
        acknowledged=s.acknowledged,
        acknowledged_at=s.acknowledged_at,
        env_path=s.env_path,
        acknowledgment_path=s.acknowledgment_path,
    )


@router.post("/api/onboarding/credential", response_model=OnboardingStatusResponse)
def api_onboarding_credential(
    req: CredentialUpsertRequest,
) -> OnboardingStatusResponse:
    """Phase 0 ``apikey-ui`` slice. Insert-or-update a credential in
    the user-data ``.env``.

    Atomicity + chmod 600 are owned by
    :func:`cloris.onboarding.upsert_credential`; this route is the
    HTTP entry, an unknown key surfaces as HTTP 422 with the allowed
    set so the welcome surface can render a precise diagnostic.

    Returns the post-write :class:`OnboardingStatusResponse` so the
    welcome surface doesn't need a follow-up GET to learn whether the
    write succeeded.
    """

    from cloris.onboarding import (
        UnknownCredentialKeyError,
        onboarding_status,
        upsert_credential,
    )

    try:
        upsert_credential(req.key, req.value)
    except UnknownCredentialKeyError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "unknown_credential_key",
                "key": exc.key,
                "allowed": list(exc.allowed),
            },
        ) from exc
    except ValueError as exc:
        error = (
            "invalid_credential_value"
            if "cannot contain" in str(exc)
            else "empty_credential_value"
        )
        raise HTTPException(
            status_code=422,
            detail={"error": error, "key": req.key, "message": str(exc)},
        ) from exc

    s = onboarding_status()
    return OnboardingStatusResponse(
        welcome_complete=s.welcome_complete,
        anthropic_present=s.anthropic_present,
        acknowledged=s.acknowledged,
        acknowledged_at=s.acknowledged_at,
        env_path=s.env_path,
        acknowledgment_path=s.acknowledgment_path,
    )


@router.post("/api/onboarding/acknowledge", response_model=OnboardingStatusResponse)
def api_onboarding_acknowledge(
    req: AcknowledgmentRequest,
) -> OnboardingStatusResponse:
    """Phase 0 ``apikey-ui`` / ``disclosure`` slice. Record the
    recipient's acknowledgment of Cloris's LinkedIn-Recruiter
    operational surface.

    Idempotent: re-acknowledging overwrites the timestamp + version
    with the current values. ``acknowledged=False`` in the request
    is rejected with HTTP 422 — the only way to reach this endpoint
    is the explicit checkbox-checked + Continue flow on the welcome
    surface.
    """

    from cloris.onboarding import onboarding_status, record_acknowledgment

    if not req.acknowledged:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "acknowledgment_required",
                "message": (
                    "Cloris cannot start until you check the acknowledgment "
                    "box and click Continue."
                ),
            },
        )

    record_acknowledgment(cloris_version=__version__)

    s = onboarding_status()
    return OnboardingStatusResponse(
        welcome_complete=s.welcome_complete,
        anthropic_present=s.anthropic_present,
        acknowledged=s.acknowledged,
        acknowledged_at=s.acknowledged_at,
        env_path=s.env_path,
        acknowledgment_path=s.acknowledgment_path,
    )


@router.post("/api/chrome-relaunch", response_model=ChromeStatusResponse)
def api_chrome_relaunch() -> ChromeStatusResponse:
    """Phase 0 ``chrome-launcher`` slice. Recycle the Cloris Chrome
    instance — only the dedicated profile, never the recipient's
    personal Chrome.

    Wired into the welcome surface's "Re-open Chrome" affordance and
    the Settings "Re-open Chrome" control. Returns the post-action
    :class:`ChromeStatusResponse` so the caller doesn't need to poll
    immediately to see whether the relaunch succeeded.
    """

    from cloris.chrome_launcher import ensure_running

    snapshot = ensure_running(force=True)
    return ChromeStatusResponse(
        state=snapshot.state,  # type: ignore[arg-type]
        cdp_url=snapshot.cdp_url,
        profile_dir=snapshot.profile_dir,
        message=snapshot.message,
    )


@router.post("/api/chrome-open-linkedin", response_model=ChromeStatusResponse)
def api_chrome_open_linkedin() -> ChromeStatusResponse:
    """Open LinkedIn Recruiter in Cloris's dedicated Chrome profile.

    Non-destructive: unlike ``/api/chrome-relaunch``, this endpoint does not
    recycle Chrome. It ensures the Cloris Chrome profile is running and opens a
    Recruiter tab in that profile so the launch-readiness probe can observe it.
    """

    from cloris.chrome_launcher import open_linkedin_recruiter

    snapshot = open_linkedin_recruiter()
    return ChromeStatusResponse(
        state=snapshot.state,  # type: ignore[arg-type]
        cdp_url=snapshot.cdp_url,
        profile_dir=snapshot.profile_dir,
        message=snapshot.message,
    )
