"""PitchBook signal adapter for executive-search dossier evaluation.

Slice 5 of the executive-search module. Pulls PE/VC company context
(buyout history, M&A events, fund-stage details) for the candidate's
most-recent employer. Higher-quality data than Crunchbase for late-
stage / PE-backed companies; gated behind enterprise-tier API access.

Maintenance honesty (per spec): PitchBook API access is enterprise-
tier and has been restricted historically. Two of five sources
(Crunchbase + PitchBook) are at meaningful provider-side churn risk.
The adapter degrades gracefully — a 5xx / 401 response returns
:class:`SignalFailure(reason="upstream_5xx")` and the dossier eval
proceeds without this section.

Slice 5 ships the adapter shell + tests against fixture HTTP. Real
production deployment will set the API key and exercise the live
path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from shared.brief_schema import Brief
from shared.schemas import CandidateProfileSummary

from exec_search.signals import SignalFailure, SignalRequestContext, SignalResult


@dataclass(frozen=True)
class PitchBookCompanySignal:
    """Normalized PitchBook organization signal."""

    company_name: str
    deal_type: str = ""
    deal_stage: str = ""
    last_deal_at: str = ""
    last_deal_amount_usd: float | None = None
    pe_backed: bool = False
    investors: tuple[str, ...] = ()


class PitchBookClient:
    """Thin PitchBook API client. Replaceable per the adapter contract."""

    BASE_URL = "https://api.pitchbook.com/v1/companies"

    def __init__(self, api_key: str, *, http_session: Any | None = None) -> None:
        self.api_key = api_key
        self._session = http_session

    def fetch_company(self, company_name: str) -> PitchBookCompanySignal | None:
        if self._session is None:
            import urllib.parse
            import urllib.request

            params = urllib.parse.urlencode({"name": company_name})
            url = f"{self.BASE_URL}?{params}"
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Cloris/1.0",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status >= 500:
                    raise PitchBookApiError(
                        f"pitchbook 5xx ({resp.status})",
                        status_code=resp.status,
                    )
                if resp.status >= 400:
                    raise PitchBookApiError(
                        f"pitchbook http {resp.status}",
                        status_code=resp.status,
                    )
                import json as _json

                payload = _json.loads(resp.read().decode("utf-8"))
        else:
            response = self._session.get(
                self.BASE_URL,
                params={"name": company_name},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30,
            )
            if response.status_code >= 500:
                raise PitchBookApiError(
                    f"pitchbook 5xx ({response.status_code})",
                    status_code=response.status_code,
                )
            if response.status_code >= 400:
                raise PitchBookApiError(
                    f"pitchbook http {response.status_code}",
                    status_code=response.status_code,
                )
            payload = response.json()
        return _parse_pitchbook_payload(payload, company_name)


class PitchBookApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _parse_pitchbook_payload(
    payload: Mapping[str, Any], company_name: str
) -> PitchBookCompanySignal | None:
    if not isinstance(payload, Mapping):
        return None
    companies = payload.get("companies") or payload.get("data") or []
    if not isinstance(companies, list) or not companies:
        return None
    first = companies[0]
    if not isinstance(first, Mapping):
        return None
    investors = first.get("investors") or []
    investor_names: list[str] = []
    if isinstance(investors, list):
        for inv in investors:
            if isinstance(inv, str):
                investor_names.append(inv)
            elif isinstance(inv, Mapping):
                name = inv.get("name") or inv.get("display_name") or ""
                if name:
                    investor_names.append(str(name))
    return PitchBookCompanySignal(
        company_name=str(first.get("name") or company_name),
        deal_type=str(first.get("last_deal_type") or ""),
        deal_stage=str(first.get("last_deal_stage") or ""),
        last_deal_at=str(first.get("last_deal_date") or ""),
        last_deal_amount_usd=_safe_float(first.get("last_deal_amount_usd")),
        pe_backed=bool(first.get("pe_backed", False)),
        investors=tuple(investor_names),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class PitchBookSignalSource:
    """Off-LinkedIn signal adapter wrapping the PitchBook API."""

    name: str = "pitchbook"
    client: PitchBookClient | None = None

    def fetch(
        self,
        *,
        candidate: CandidateProfileSummary,
        brief: Brief,
        context: SignalRequestContext,
    ) -> SignalResult | SignalFailure:
        api_key = (os.environ.get("PITCHBOOK_API_KEY") or "").strip()
        if self.client is None and not api_key:
            return SignalFailure(
                source=self.name,
                reason="disabled_no_api_key",
                detail="PITCHBOOK_API_KEY not set",
            )
        client = self.client or PitchBookClient(api_key=api_key)

        companies = _candidate_companies(candidate)
        if not companies:
            return SignalFailure(
                source=self.name,
                reason="no_companies_to_query",
                detail="candidate experiences carry no company names",
            )

        signals: list[PitchBookCompanySignal] = []
        for company_name in companies[:3]:
            try:
                signal = client.fetch_company(company_name)
            except PitchBookApiError as exc:
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
                    "PE/VC context (PitchBook):\n"
                    f"  [no PitchBook match for any of "
                    f"{', '.join(companies[:3])}]"
                ),
            )
        return SignalResult(
            source=self.name,
            section_text=_format_pitchbook_section(signals),
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


def _format_pitchbook_section(signals: list[PitchBookCompanySignal]) -> str:
    lines = ["PE/VC context (PitchBook):"]
    for s in signals:
        lines.append(f"  - {s.company_name}")
        if s.deal_type or s.deal_stage:
            deal = " / ".join(p for p in (s.deal_type, s.deal_stage) if p)
            lines.append(f"      Last deal: {deal}")
        if s.last_deal_at:
            amount = (
                f", ${s.last_deal_amount_usd:,.0f}"
                if s.last_deal_amount_usd
                else ""
            )
            lines.append(f"      Date: {s.last_deal_at[:10]}{amount}")
        if s.pe_backed:
            lines.append("      PE-backed: yes")
        if s.investors:
            lines.append(f"      Investors: {', '.join(s.investors[:5])}")
    return "\n".join(lines)


__all__ = (
    "PitchBookApiError",
    "PitchBookClient",
    "PitchBookCompanySignal",
    "PitchBookSignalSource",
)
