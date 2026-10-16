"""
Bias controls for the sourcing agent orchestrator.

These are SYSTEM-LEVEL behaviors that run identically across all searches.
They protect evaluation quality over long runs by detecting compounding bias
patterns and surfacing them before they corrupt the pipeline.

Integration: the orchestrator calls record_decision() after every evaluation
and check_alerts() periodically (e.g., after each page or each string).

The controls themselves are role-agnostic. The thresholds come from the Brief's
BiasControls parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import json
from pathlib import Path


@dataclass
class DecisionRecord:
    """One evaluation decision, recorded for bias monitoring."""
    candidate_id: str
    string_id: str
    stage: str                  # "facial" | "full"
    decision: str               # "FACIAL_YES" | "FACIAL_NO" | "SAVE" | "INFERENTIAL_SAVE" | "TRANSFERABLE_SAVE" | "SIGNAL_SAVE" | "REJECT" | "PARSE_FAILURE"
    confidence: Optional[float]  # None on valid V2 facial verdicts (the
                                 # facial contract carries no confidence; stored
                                 # and JSON-serialized only, no arithmetic here)
    capability_area: Optional[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """A bias control alert surfaced by the monitoring system."""
    alert_type: str             # See AlertType constants below
    severity: str               # "pause" | "flag" | "info"
    message: str
    string_id: str
    data: dict = field(default_factory=dict)


class AlertType:
    CONSECUTIVE_SAVES = "consecutive_saves"
    CONSECUTIVE_REJECTS = "consecutive_rejects"
    SAVE_RATE_SPIKE = "save_rate_spike"
    PARSE_FAILURE_RATE = "parse_failure_rate"
    FACIAL_RATE_ANOMALY = "facial_rate_anomaly"
    VOLUME_INVERSION = "volume_inversion"


SAVE_DECISIONS = {
    "SAVE",
    "INFERENTIAL_SAVE",
    "TRANSFERABLE_SAVE",
    "SIGNAL_SAVE",
}


def is_save_decision(decision: str) -> bool:
    """Return True when a terminal decision should count as a save."""
    return decision in SAVE_DECISIONS


class BiasMonitor:
    """
    Monitors evaluation decisions for compounding bias patterns.

    Tracks decisions per-string and across the session. Surfaces alerts
    when patterns indicate the evaluation quality may be drifting.

    Usage:
        monitor = BiasMonitor(max_consecutive_saves=5, ...)
        # After each evaluation:
        monitor.record_decision(record)
        # After each page or string:
        alerts = monitor.check_alerts(current_string_id)
        for alert in alerts:
            if alert.severity == "pause":
                # Pause the current string, surface to operator
            elif alert.severity == "flag":
                # Log warning, continue
    """

    def __init__(
        self,
        max_consecutive_saves: int = 5,
        max_consecutive_rejects: int = 20,
        parse_failure_alarm_rate: float = 0.03,
        expected_facial_yes_low: float = 0.25,
        expected_facial_yes_high: float = 0.55,
        save_rate_spike_threshold: float = 0.60,
        save_rate_spike_window: int = 15,
    ):
        self.max_consecutive_saves = max_consecutive_saves
        self.max_consecutive_rejects = max_consecutive_rejects
        self.parse_failure_alarm_rate = parse_failure_alarm_rate
        self.expected_facial_yes_low = expected_facial_yes_low
        self.expected_facial_yes_high = expected_facial_yes_high
        self.save_rate_spike_threshold = save_rate_spike_threshold
        self.save_rate_spike_window = save_rate_spike_window

        # State
        self._decisions: list[DecisionRecord] = []
        self._per_string: dict[str, list[DecisionRecord]] = {}
        self._alerts_fired: set[str] = set()  # Dedup key: f"{alert_type}:{string_id}"
        # Retain a raw FACIAL_BORDERLINE counter for diagnostics. Current
        # persistence keeps FACIAL_BORDERLINE; the counter also covers checkpoints
        # whose rows were aliased. It does not feed alarms and only distinguishes
        # the class in session summaries.
        self._facial_borderline_counts: dict[str, int] = {}
        # Store full fired-Alert payloads in firing order. ``_alerts_fired`` keeps
        # only dedup keys, while reports need alert messages and data after the
        # transient Alert objects are gone. Checkpoint the payload list with the
        # other bias state so reports can recover it after a restart.
        self._fired_alert_records: list[dict] = []

    @classmethod
    def from_brief(cls, brief) -> "BiasMonitor":
        """Construct a BiasMonitor with parameters from a Brief."""
        bc = brief.bias_controls
        fc = brief.facial_calibration
        return cls(
            max_consecutive_saves=bc.max_consecutive_saves,
            max_consecutive_rejects=bc.max_consecutive_rejects,
            parse_failure_alarm_rate=bc.parse_failure_alarm_rate,
            expected_facial_yes_low=fc.expected_yes_rate_low,
            expected_facial_yes_high=fc.expected_yes_rate_high,
        )

    def record_decision(self, record: DecisionRecord) -> None:
        """Record a single evaluation decision."""
        self._decisions.append(record)
        if record.string_id not in self._per_string:
            self._per_string[record.string_id] = []
        self._per_string[record.string_id].append(record)

    def record_facial_borderline_seen(self, string_id: str) -> None:
        """Increment the raw borderline counter for a string.

        The counter records a borderline observation for diagnostics. Current
        persistence keeps ``FACIAL_BORDERLINE``; it also supplies detail for
        compatibility checkpoints whose rows used the alias. This method does
        not feed ``_check_facial_rate_anomaly`` or ``get_tightening_status``;
        alarm and tightening logic independently compute the
        opens-for-full-evaluation rate, counting YES + BORDERLINE as applicable.
        """
        if not string_id:
            return
        self._facial_borderline_counts[string_id] = (
            self._facial_borderline_counts.get(string_id, 0) + 1
        )

    def check_alerts(self, current_string_id: str) -> list[Alert]:
        """
        Run all bias checks and return any triggered alerts.
        Call after each page of results or after each string completes.
        """
        alerts = []
        alerts.extend(self._check_consecutive_saves(current_string_id))
        alerts.extend(self._check_consecutive_rejects(current_string_id))
        alerts.extend(self._check_save_rate_spike(current_string_id))
        alerts.extend(self._check_parse_failure_rate(current_string_id))
        alerts.extend(self._check_facial_rate_anomaly(current_string_id))
        for alert in alerts:
            self._fired_alert_records.append({
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "string_id": alert.string_id,
                "data": dict(alert.data),
            })
        return alerts

    @property
    def fired_alert_records(self) -> list[dict]:
        """Full payloads of every alert that has fired, in firing order."""
        return [dict(record) for record in self._fired_alert_records]

    def string_context(self, string_id: str) -> dict | None:
        """Return compact per-string context for the adaptation model.

        Count-based checks report through this context rather than interrupting a
        string. ``opens_for_full_eval_rate`` counts FACIAL_YES plus
        FACIAL_BORDERLINE and excludes FACIAL_SKIP; ``facial_n`` exposes small
        samples. Per-string alert types are included, while parse-failure remains
        session-scoped.

        Return ``None`` when the string has no recorded decisions.
        """
        decisions = self._per_string.get(string_id, [])
        if not decisions:
            return None
        full = [d for d in decisions if d.stage == "full"]
        facial = [
            d for d in decisions
            if d.stage == "facial" and d.decision != "FACIAL_SKIP"
        ]
        saves = sum(1 for d in full if is_save_decision(d.decision))
        rejects = sum(1 for d in full if d.decision == "REJECT")
        facial_opens = sum(
            1
            for d in facial
            if d.decision in {"FACIAL_YES", "FACIAL_BORDERLINE"}
        )
        fired_types = sorted({
            record["alert_type"]
            for record in self._fired_alert_records
            if record.get("string_id") == string_id
            and record["alert_type"] != AlertType.PARSE_FAILURE_RATE
        })
        return {
            "full_evals": len(full),
            "saves": saves,
            "rejects": rejects,
            "save_rate": saves / len(full) if full else 0.0,
            "opens_for_full_eval_rate": (
                facial_opens / len(facial) if facial else 0.0
            ),
            "facial_n": len(facial),
            "fired_alert_types": fired_types,
        }

    # --- Individual checks ---

    def _check_consecutive_saves(self, string_id: str) -> list[Alert]:
        """
        Detects N consecutive SAVE decisions on one string without a REJECT.
        A streak is ambiguous: a refined pocket and an over-broad judge can have
        the same count. Record the signal in adaptation context so the model can
        weigh the saves' rationales; never interrupt the string here.
        """
        key = f"{AlertType.CONSECUTIVE_SAVES}:{string_id}"
        if key in self._alerts_fired:
            return []

        string_decisions = [
            d for d in self._per_string.get(string_id, [])
            if d.stage == "full"
        ]
        if not string_decisions:
            return []

        # Count consecutive saves from the end
        consecutive = 0
        for d in reversed(string_decisions):
            if is_save_decision(d.decision):
                consecutive += 1
            else:
                break

        if consecutive >= self.max_consecutive_saves:
            self._alerts_fired.add(key)
            return [Alert(
                alert_type=AlertType.CONSECUTIVE_SAVES,
                severity="flag",
                message=(
                    f"String {string_id}: {consecutive} consecutive saves with no reject. "
                    f"High save density — recorded for the adaptation context."
                ),
                string_id=string_id,
                data={"consecutive_saves": consecutive},
            )]
        return []

    def _check_consecutive_rejects(self, string_id: str) -> list[Alert]:
        """
        Detects: N consecutive REJECT decisions on the same string.
        Why: Possible prompt drift toward avoidance, or a dead string.
        Severity: FLAG. Log warning, suggest operator review.
        """
        key = f"{AlertType.CONSECUTIVE_REJECTS}:{string_id}"
        if key in self._alerts_fired:
            return []

        string_decisions = [
            d for d in self._per_string.get(string_id, [])
            if d.stage == "full"
        ]
        if not string_decisions:
            return []

        consecutive = 0
        for d in reversed(string_decisions):
            if d.decision == "REJECT":
                consecutive += 1
            else:
                break

        if consecutive >= self.max_consecutive_rejects:
            self._alerts_fired.add(key)
            return [Alert(
                alert_type=AlertType.CONSECUTIVE_REJECTS,
                severity="flag",
                message=(
                    f"String {string_id}: {consecutive} consecutive rejects. "
                    f"May indicate avoidant drift or a poorly targeted string."
                ),
                string_id=string_id,
                data={"consecutive_rejects": consecutive},
            )]
        return []

    def _check_save_rate_spike(self, string_id: str) -> list[Alert]:
        """
        Detects a save rate above threshold within a rolling window.
        Density alone cannot distinguish an over-broad net from a refined pocket;
        record the flag for adaptation with its string context rather than
        changing execution flow.
        """
        key = f"{AlertType.SAVE_RATE_SPIKE}:{string_id}"
        if key in self._alerts_fired:
            return []

        string_decisions = [
            d for d in self._per_string.get(string_id, [])
            if d.stage == "full"
        ]

        # Only check after enough decisions for the window
        window = self.save_rate_spike_window
        if len(string_decisions) < window:
            return []

        recent = string_decisions[-window:]
        save_count = sum(1 for d in recent if is_save_decision(d.decision))
        save_rate = save_count / len(recent)

        if save_rate >= self.save_rate_spike_threshold:
            self._alerts_fired.add(key)
            return [Alert(
                alert_type=AlertType.SAVE_RATE_SPIKE,
                severity="flag",
                message=(
                    f"String {string_id}: {save_rate:.0%} save rate over last {window} evaluations "
                    f"(threshold {self.save_rate_spike_threshold:.0%}). "
                    f"High save density — recorded for the adaptation context."
                ),
                string_id=string_id,
                data={"save_rate": save_rate, "window": window, "save_count": save_count},
            )]
        return []

    def _check_parse_failure_rate(self, string_id: str) -> list[Alert]:
        """
        Detects: Parse failure rate exceeding threshold across the session.
        Why: Indicates prompt structure or extraction quality issues.
        Severity: FLAG. Don't pause, but surface for investigation.
        """
        key = f"{AlertType.PARSE_FAILURE_RATE}:session"
        if key in self._alerts_fired:
            return []

        total = len(self._decisions)
        if total < 20:  # Don't alert on small samples
            return []

        parse_failures = sum(1 for d in self._decisions
                             if "PARSE_FAILURE" in d.decision or "JUDGMENT_FAILURE" in d.decision)
        failure_rate = parse_failures / total

        if failure_rate >= self.parse_failure_alarm_rate:
            self._alerts_fired.add(key)
            return [Alert(
                alert_type=AlertType.PARSE_FAILURE_RATE,
                severity="flag",
                message=(
                    f"Session parse failure rate: {failure_rate:.1%} ({parse_failures}/{total}). "
                    f"Threshold: {self.parse_failure_alarm_rate:.0%}. Check prompt structure or extraction quality."
                ),
                string_id=string_id,
                data={"failure_rate": failure_rate, "failures": parse_failures, "total": total},
            )]
        return []

    # The historical field name says YES rate; the operational quantity is
    # opens-for-full-eval, now computed from distinct YES + BORDERLINE rows.
    def _check_facial_rate_anomaly(self, string_id: str) -> list[Alert]:
        """
        Detects: Facial YES rate outside expected range for a string.
        Why: Too high → triage isn't filtering. Too low → avoidant drift at triage stage.
        Severity: INFO. Diagnostic, not actionable on its own.
        """
        key = f"{AlertType.FACIAL_RATE_ANOMALY}:{string_id}"
        if key in self._alerts_fired:
            return []

        facial_decisions = [
            d for d in self._per_string.get(string_id, [])
            if d.stage == "facial" and d.decision != "FACIAL_SKIP"
        ]

        if len(facial_decisions) < 10:  # Need enough samples
            return []

        yes_count = sum(
            1
            for d in facial_decisions
            if d.decision in {"FACIAL_YES", "FACIAL_BORDERLINE"}
        )
        yes_rate = yes_count / len(facial_decisions)

        if yes_rate < self.expected_facial_yes_low:
            self._alerts_fired.add(key)
            return [Alert(
                alert_type=AlertType.FACIAL_RATE_ANOMALY,
                severity="info",
                message=(
                    f"String {string_id}: Facial YES rate {yes_rate:.0%} below expected range "
                    f"({self.expected_facial_yes_low:.0%}–{self.expected_facial_yes_high:.0%}). "
                    f"Possible triage-stage avoidance or off-target string."
                ),
                string_id=string_id,
                data={"yes_rate": yes_rate, "yes_count": yes_count, "total": len(facial_decisions)},
            )]
        elif yes_rate > self.expected_facial_yes_high:
            self._alerts_fired.add(key)
            return [Alert(
                alert_type=AlertType.FACIAL_RATE_ANOMALY,
                severity="info",
                message=(
                    f"String {string_id}: Facial YES rate {yes_rate:.0%} above expected range "
                    f"({self.expected_facial_yes_low:.0%}–{self.expected_facial_yes_high:.0%}). "
                    f"Triage may not be filtering effectively."
                ),
                string_id=string_id,
                data={"yes_rate": yes_rate, "yes_count": yes_count, "total": len(facial_decisions)},
            )]
        return []

    # --- Triage tightening ---

    # Tightening uses the same distinct YES + BORDERLINE open rate.
    def get_tightening_status(self, string_id: str) -> dict | None:
        """Check if triage should be tightened for this string.
        Returns dict with rate info if tightening needed, else None.
        """
        facial_decisions = [
            d for d in self._per_string.get(string_id, [])
            if d.stage == "facial" and d.decision != "FACIAL_SKIP"
        ]
        if len(facial_decisions) < 10:
            return None
        yes_count = sum(
            1
            for d in facial_decisions
            if d.decision in {"FACIAL_YES", "FACIAL_BORDERLINE"}
        )
        yes_rate = yes_count / len(facial_decisions)
        threshold = self.expected_facial_yes_high * 2
        if yes_rate > threshold:
            return {
                "actual_rate": yes_rate,
                "expected_high": self.expected_facial_yes_high,
                "multiplier": yes_rate / self.expected_facial_yes_high,
            }
        return None

    # --- Session diagnostics ---

    def session_summary(self) -> dict:
        """Generate a diagnostic summary of the current session's bias metrics."""
        total = len(self._decisions)
        if total == 0:
            return {"total_decisions": 0}

        facial = [d for d in self._decisions if d.stage == "facial"]
        full = [d for d in self._decisions if d.stage == "full"]

        facial_yes = sum(1 for d in facial if d.decision == "FACIAL_YES")
        facial_opens = sum(
            1
            for d in facial
            if d.decision in {"FACIAL_YES", "FACIAL_BORDERLINE"}
        )
        saves = sum(1 for d in full if is_save_decision(d.decision))
        rejects = sum(1 for d in full if d.decision == "REJECT")
        parse_failures = sum(1 for d in self._decisions
                             if "PARSE_FAILURE" in d.decision or "JUDGMENT_FAILURE" in d.decision)

        # Per-string breakdown
        per_string_stats = {}
        for sid, decisions in self._per_string.items():
            full_d = [d for d in decisions if d.stage == "full"]
            s = sum(1 for d in full_d if is_save_decision(d.decision))
            r = sum(1 for d in full_d if d.decision == "REJECT")
            per_string_stats[sid] = {
                "saves": s,
                "rejects": r,
                "save_rate": s / len(full_d) if full_d else 0,
                "total_full_evals": len(full_d),
            }

        # ``facial_open_rate`` is the distinct YES + BORDERLINE open rate, while
        # ``facial_yes_rate`` remains its compatibility alias. The borderline
        # counter supports aliased rows; its denominator excludes skips, while
        # the alias uses the full facial list by design.
        facial_open_rate = facial_opens / len(facial) if facial else 0
        distinct_borderline_count = sum(
            1 for d in facial if d.decision == "FACIAL_BORDERLINE"
        )
        borderline_count = max(
            distinct_borderline_count,
            sum(self._facial_borderline_counts.values()),
        )
        facial_post_skip_total = sum(
            1 for d in facial if d.decision != "FACIAL_SKIP"
        )
        facial_borderline_rate = (
            borderline_count / facial_post_skip_total
            if facial_post_skip_total
            else 0.0
        )

        return {
            "total_decisions": total,
            "facial_total": len(facial),
            "facial_open_rate": facial_open_rate,
            "facial_yes_rate": facial_open_rate,  # deprecation alias for one release; migrate to facial_open_rate
            "facial_borderline_rate": facial_borderline_rate,
            "facial_borderline_count": borderline_count,
            "full_total": len(full),
            "save_rate": saves / len(full) if full else 0,
            "saves": saves,
            "rejects": rejects,
            "parse_failures": parse_failures,
            "parse_failure_rate": parse_failures / total if total else 0,
            "per_string": per_string_stats,
            "alerts_fired": list(self._alerts_fired),
        }

    def save_checkpoint(self, path: str) -> None:
        """Persist decision history and alert state for crash recovery."""
        checkpoint = {
            "decisions": [
                {
                    "candidate_id": d.candidate_id,
                    "string_id": d.string_id,
                    "stage": d.stage,
                    "decision": d.decision,
                    "confidence": d.confidence,
                    "capability_area": d.capability_area,
                    "timestamp": d.timestamp,
                }
                for d in self._decisions
            ],
            "alerts_fired": list(self._alerts_fired),
            # Persist the borderline diagnostic counter with the checkpoint so
            # crash recovery preserves the summary value, including checkpoints
            # whose stored rows used the compatibility alias.
            "facial_borderline_counts": dict(self._facial_borderline_counts),
            # Store full fired-Alert payloads for the run report, with the same
            # backward-compatible checkpoint behavior as the borderline counter.
            "fired_alerts": [dict(record) for record in self._fired_alert_records],
        }
        Path(path).write_text(json.dumps(checkpoint, indent=2))

    def load_checkpoint(self, path: str) -> None:
        """Restore from checkpoint after crash/restart."""
        data = json.loads(Path(path).read_text())
        self._decisions = []
        self._per_string = {}
        self._alerts_fired = set(data.get("alerts_fired", []))
        for d in data.get("decisions", []):
            record = DecisionRecord(
                candidate_id=d["candidate_id"],
                string_id=d["string_id"],
                stage=d["stage"],
                decision=d["decision"],
                confidence=d["confidence"],
                capability_area=d.get("capability_area"),
                timestamp=d.get("timestamp", 0),
            )
            self.record_decision(record)
        # Restore the borderline counter. ``.get(..., {})`` keeps checkpoints
        # written without this key loadable; they resume with an empty counter.
        self._facial_borderline_counts = dict(data.get("facial_borderline_counts", {}))
        # Restore fired-Alert payloads. Checkpoints without this key load with
        # an empty list.
        self._fired_alert_records = [
            dict(record) for record in data.get("fired_alerts", [])
        ]
