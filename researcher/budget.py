"""Campaign cost reservations in existing run payloads; provider receipts settle them."""

from __future__ import annotations

import math

from exa.cost_meter import ExaCostMeter
from shared.failures import ApiBudgetExhaustedError
from shared.llm_usage import estimate_usage_cost_usd


def cost_summary(receipts: list[dict]) -> dict:
    measured = sum(r["actual_usd"] for r in receipts if r.get("actual_usd") is not None)
    estimated = sum(r["estimate_usd"] for r in receipts if r.get("actual_usd") is None)
    return {"measured_usd": measured, "estimated_usd": estimated, "charged_usd": measured + estimated,
            "unavailable_calls": sum(r.get("actual_usd") is None for r in receipts), "calls": len(receipts)}


class CampaignBudget:
    def __init__(self, bridge, run_id, check_stop, on_dispatch=lambda stage, campaign: None):
        self.bridge, self.run_id, self.check_stop = bridge, run_id, check_stop
        self.on_dispatch = on_dispatch
        self.fatal_error = None
        self.query_id = None

    def reserve(self, stage: str, estimate: float) -> int:
        self.check_stop()
        if isinstance(estimate, bool) or not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or estimate < 0:
            raise ValueError("A finite conservative dispatch cost estimate is required")
        campaign = self.bridge.campaign(self.run_id)
        receipts = campaign.setdefault("cost_receipts", [])
        if cost_summary(receipts)["charged_usd"] + estimate > campaign["limits"]["max_cost_usd"] + 1e-12:
            raise ApiBudgetExhaustedError("Researcher campaign cannot fund the next dispatch")
        self.on_dispatch(stage, campaign)
        receipt_id = len(receipts)
        receipts.append({"stage": stage, "estimate_usd": estimate, "actual_usd": None, "status": "reserved", "run_id": self.run_id, "query_id": self.query_id})
        self.bridge.save_campaign(self.run_id, campaign)
        return receipt_id

    def settle(self, receipt_id: int, actual=None, *, status="returned"):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < 0:
            actual = None
        campaign = self.bridge.campaign(self.run_id)
        campaign["cost_receipts"][receipt_id].update(actual_usd=actual, status=status)
        self.bridge.save_campaign(self.run_id, campaign)

    def dispatch(self, stage, call, *, estimate=0.001, retries=1):
        for attempt in range(retries):
            receipt = self.reserve(stage, estimate)
            try:
                result = call()
            except BaseException as exc:
                self.settle(receipt, status="unknown_outcome")
                status = getattr(getattr(exc, "response", None), "status_code", None)
                from shared.failures import classify_runtime_failure
                if isinstance(exc, Exception) and attempt + 1 < retries and (status in {429, 500, 502, 503, 504} or classify_runtime_failure(exc, source=stage).retryable):
                    continue
                raise
            actual = (result.get("meta") or {}).get("cost_usd") if isinstance(result, dict) else None
            self.settle(receipt, actual)
            return result

    def model_caller(self, stage, caller=None, *, estimate=None):
        from shared import config
        from shared.judger import _invoke_llm_caller
        from shared.llm_clients import opus_llm_cached
        model = config.FACIAL_MODEL_NAME if stage == "facial" else config.OPUS_MODEL_NAME
        if caller is not None and estimate is None:
            raise ValueError(f"Injected {stage} transport requires a cost estimate")

        def bounded(system, user, *, usage_context=None):
            context = usage_context if usage_context is not None else {}
            context.update(stage=f"researcher_{stage}", source="researcher", run_id=self.run_id, brief_id=self.bridge.brief_id)
            cap = max(4096, getattr(config, "FIREWORKS_PRIMARY_MIN_MAX_TOKENS", 4096))
            bound = estimate
            if bound is None:
                # UTF-8 bytes bound input tokens; double for framing and cache writes.
                bound, rate = estimate_usage_cost_usd(model=model, input_tokens=2 * (len(system.encode()) + len(user.encode())), output_tokens=cap)
                if bound is None or rate == "unknown":
                    raise ValueError("Configured model has no bounded cost estimate")
            receipt = self.reserve(stage, bound)
            try:
                if caller is None:
                    result = opus_llm_cached(system, user, model_name=model, expect_json=False, max_tokens=4096,
                                             max_attempts=1, timeout_seconds=60, usage_context=context)
                else:
                    result = _invoke_llm_caller(caller, system_prompt=system, user_prompt=user, usage_context=context)
            except BaseException as exc:
                detail = context.get("_llm_receipt", {}).get("actual_detail", {})
                actual = detail.get("estimated_cost_usd") if detail.get("cost_completeness") == "complete" else None
                self.settle(receipt, actual, status="failed_or_unknown")
                status = getattr(exc, "status_code", getattr(getattr(exc, "response", None), "status_code", None))
                if status in {401, 403}:
                    self.fatal_error = exc
                raise
            detail = context.get("_llm_receipt", {}).get("actual_detail", {})
            actual = detail.get("estimated_cost_usd") if detail.get("cost_completeness") == "complete" else None
            self.settle(receipt, actual)
            return result
        return bounded


class CampaignExaMeter(ExaCostMeter):
    """Use Exa's price calculation and retry hooks with canonical reservations."""
    def __init__(self, budget):
        super().__init__(cap_usd=0)
        self.budget, self.reservations = budget, []

    def reserve(self, estimate_usd):
        self.reservations.append(self.budget.reserve("exa", estimate_usd))
        super().reserve(estimate_usd)

    def release(self, estimate_usd):
        self.budget.settle(self.reservations.pop(), 0, status="provider_declined")
        super().release(estimate_usd)

    def settle(self, *, endpoint, estimate_usd, cost_dollars, request_id, num_results):
        actual = cost_dollars.get("total") if isinstance(cost_dollars, dict) else None
        self.budget.settle(self.reservations.pop(), actual)
        super().settle(endpoint=endpoint, estimate_usd=estimate_usd, cost_dollars=cost_dollars,
                       request_id=request_id, num_results=num_results)
