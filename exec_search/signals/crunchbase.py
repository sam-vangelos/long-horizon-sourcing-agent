"""Crunchbase signal adapter for executive-search dossier evaluation.

Slice 5 of the executive-search module. Pulls company stage / funding
/ leadership-history context for the candidate's most-recent
employer. This is the kind of signal that's load-bearing when the
client cares about "operator who's seen Series-D-to-acquisition" or
"PE-backed turnaround background."

Maintenance honesty (per spec maintenance table): Crunchbase API has
had multiple pricing/API breakage events. The adapter is wrapped in
a per-source circuit-breaker — a 5xx response degrades to
:class:`SignalFailure(reason="upstream_5xx")` and the dossier eval
continues without this section. The recruiter sees an honest "signal
unavailable" placeholder.

Slice 5 ships the adapter shell + tests against fixture HTTP. The
real Crunchbase contract / data API call is gated behind
``CRUNCHBASE_API_KEY`` env; without it the adapter returns
:class:`SignalFailure(reason="disabled_no_api_key")`. Production
deployment will set the key and exercise the real path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from shared.brief_schema import Brief
from shared.schemas import CandidateProfileSummary

from exec_search.signals import SignalFailure, SignalRequestContext, SignalResult


@dataclass(frozen=True)
class CompanyStageSignal:
    """Normalized Crunchbase organization signal."""

    company_name: str
    stage: str = ""
    last_funding_round: str = ""
    last_funding_amount_usd: float | None = None
    last_funding_at: str = ""
    headcount_estimate: str = ""
    operating_status: str = ""


class CrunchbaseClient:
    """Thin Crunchbase API client. Replaceable per the adapter contract."""

    BASE_URL = "https://api.crunchbase.com/api/v4/entities/organizations"

    def __init__(self, api_key: str, *, http_session: Any | None = None) -> None:
        self.api_key = api_key
        self._session = http_session

    def fetch_company(self, company_name: str) -> CompanyStageSignal | None:
        """Look up a company by name. Returns ``None`` on no-match.

        Raises :class:`CrunchbaseApiError` on HTTP / API errors so
        the adapter's per-source circuit-breaker can convert to a
        :class:`SignalFailure`.
        """

        if self._session is None:
            import urllib.parse
            import urllib.request

            params = urllib.parse.urlencode(
                {"query": company_name, "user_key": self.api_key}
            )
            url = f"{self.BASE_URL}?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Cloris/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status >= 500:
                    raise CrunchbaseApiError(
                        f"crunchbase 5xx ({resp.status})",
                        status_code=resp.status,
                    )
                if resp.status >= 400:
                    raise CrunchbaseApiError(
                        f"crunchbase http {resp.status}",
                        status_code=resp.status,
                    )
                import json as _json

                payload = _json.loads(resp.read().decode("utf-8"))
        else:
            response = self._session.get(
                self.BASE_URL,
                params={"query": company_name, "user_key": self.api_key},
                timeout=30,
            )
            if response.status_code >= 500:
                raise CrunchbaseApiError(
                    f"crunchbase 5xx ({response.status_code})",
                    status_code=response.status_code,
                )
            if response.status_code >= 400:
                raise CrunchbaseApiError(
                    f"crunchbase http {response.status_code}",
                    status_code=response.status_code,
                )
            payload = response.json()
        return _parse_crunchbase_payload(payload, company_name)


class CrunchbaseApiError(RuntimeError):
    """Crunchbase returned a non-success response."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _parse_crunchbase_payload(
    payload: Mapping[str, Any], company_name: str
) -> CompanyStageSignal | None:
    """Map Crunchbase's response shape into a CompanyStageSignal."""

    if not isinstance(payload, Mapping):
        return None
    entities = payload.get("entities") or payload.get("data") or []
    if not isinstance(entities, list) or not entities:
        return None
    first = entities[0]
    if not isinstance(first, Mapping):
        return None
    properties = first.get("properties") or first
    if not isinstance(properties, Mapping):
        return None
    return CompanyStageSignal(
        company_name=str(properties.get("name") or company_name),
        stage=str(properties.get("operating_status") or ""),
        last_funding_round=str(properties.get("last_funding_type") or ""),
        last_funding_amount_usd=_safe_float(properties.get("last_funding_total")),
        last_funding_at=str(properties.get("last_funding_at") or ""),
        headcount_estimate=str(properties.get("num_employees_enum") or ""),
        operating_status=str(properties.get("operating_status") or ""),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        amount = value.get("value_usd") or value.get("value") or value.get("amount")
        if isinstance(amount, (int, float)):
            return float(amount)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class CrunchbaseSignalSource:
    """Off-LinkedIn signal adapter wrapping the Crunchbase API."""

    name: str = "crunchbase"
    client: CrunchbaseClient | None = None

    def fetch(
        self,
        *,
        candidate: CandidateProfileSummary,
        brief: Brief,
        context: SignalRequestContext,
    ) -> SignalResult | SignalFailure:
        api_key = (os.environ.get("CRUNCHBASE_API_KEY") or "").strip()
        if self.client is None and not api_key:
            return SignalFailure(
                source=self.name,
                reason="disabled_no_api_key",
                detail="CRUNCHBASE_API_KEY not set",
            )
        client = self.client or CrunchbaseClient(api_key=api_key)

        companies = _candidate_companies(candidate)
        if not companies:
            return SignalFailure(
                source=self.name,
                reason="no_companies_to_query",
                detail="candidate experiences carry no company names",
            )

        signals: list[CompanyStageSignal] = []
        for company_name in companies[:3]:
            try:
                signal = client.fetch_company(company_name)
            except CrunchbaseApiError as exc:
                if exc.status_code is not None and exc.status_code >= 500:
                    return SignalFailure(
                        source=self.name,
                        reason="upstream_5xx",
                        detail=str(exc),
                    )
                return SignalFailure(
                    source=self.name,
                    reason="upstream_error",
                    detail=str(exc),
                )
            except Exception as exc:
                return SignalFailure(
                    source=self.name,
                    reason="adapter_exception",
                    detail=f"{exc.__class__.__name__}: {exc}",
                )
            if signal is not None:
                signals.append(signal)

        if not signals:
            return SignalResult(
                source=self.name,
                section_text=(
                    "Company-stage signals (Crunchbase):\n"
                    "  [no Crunchbase match for any of the candidate's "
                    f"recent companies: {', '.join(companies[:3])}]"
                ),
            )
        return SignalResult(
            source=self.name,
            section_text=_format_crunchbase_section(signals),
        )


def _candidate_companies(candidate: CandidateProfileSummary) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for exp in (candidate.experiences or []):
        company = (getattr(exp, "company", "") or "").strip()
        if company and company not in seen:
            seen.add(company)
            out.append(company)
    return out


def _format_crunchbase_section(signals: list[CompanyStageSignal]) -> str:
    lines = ["Company-stage signals (Crunchbase):"]
    for s in signals:
        lines.append(f"  - {s.company_name}")
        if s.stage:
            lines.append(f"      Status: {s.stage}")
        if s.last_funding_round:
            funding = s.last_funding_round
            if s.last_funding_amount_usd:
                funding += f" (${s.last_funding_amount_usd:,.0f})"
            if s.last_funding_at:
                funding += f", {s.last_funding_at[:10]}"
            lines.append(f"      Last funding: {funding}")
        if s.headcount_estimate:
            lines.append(f"      Headcount: {s.headcount_estimate}")
    return "\n".join(lines)


__all__ = (
    "CompanyStageSignal",
    "CrunchbaseApiError",
    "CrunchbaseClient",
    "CrunchbaseSignalSource",
)
