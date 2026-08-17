"""Bounded LinkedIn browser recovery service."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .stop_reasons import RunStopReason

if TYPE_CHECKING:
    from linkedin.recruiter_recovery import RecruiterRecoverySnapshot
    from shared.safety.coordinator import RunSafetyCoordinator


class LinkedInRecoveryService:
    """Owns bounded browser recovery and runtime events for LinkedIn."""

    def __init__(
        self,
        *,
        coordinator: "RunSafetyCoordinator | None",
        browser,
        max_attempts: int = 2,
        wait_seconds: int = 10,
    ):
        self.coordinator = coordinator
        self.browser = browser
        self.max_attempts = max_attempts
        self.wait_seconds = wait_seconds

    async def recover(
        self,
        *,
        run_id: int | None,
        recovery_url: str | None = None,
        snapshot: "RecruiterRecoverySnapshot | None" = None,
    ) -> bool:
        if snapshot is not None:
            return await self._recover_with_snapshot(run_id=run_id, snapshot=snapshot)
        return await self._recover_legacy(run_id=run_id, recovery_url=recovery_url)

    async def _recover_with_snapshot(
        self,
        *,
        run_id: int | None,
        snapshot: "RecruiterRecoverySnapshot",
    ) -> bool:
        from linkedin.recruiter_recovery import recover_recruiter_context

        def event_recorder(event_type: str, payload: dict[str, Any]) -> None:
            if self.coordinator and run_id:
                self.coordinator.record_browser_recovery_event(
                    run_id=run_id,
                    status=event_type.replace("recruiter_recovery_", ""),
                    payload=payload,
                )

        result = await recover_recruiter_context(
            self.browser,
            snapshot,
            max_attempts=self.max_attempts,
            event_recorder=event_recorder,
        )
        return result.success

    async def _recover_legacy(
        self,
        *,
        run_id: int | None,
        recovery_url: str | None = None,
    ) -> bool:
        for attempt in range(1, self.max_attempts + 1):
            if self.coordinator and run_id:
                self.coordinator.record_browser_recovery_event(
                    run_id=run_id,
                    status="attempt",
                    payload={"attempt": attempt, "recovery_url": recovery_url or ""},
                )
            recovered = await self.browser.recover_from_target_crash(recovery_url=recovery_url)
            if not recovered:
                print(
                    f"  [reconnect] Attempt {attempt}/{self.max_attempts} — waiting {self.wait_seconds}s for Chrome..."
                )
                await asyncio.sleep(self.wait_seconds)
                try:
                    await self.browser.disconnect()
                except Exception:
                    pass
                try:
                    await self.browser.connect()
                    if recovery_url:
                        await self.browser.navigate_to_search(recovery_url)
                    recovered = True
                except Exception as exc:
                    recovered = False
                    if self.coordinator and run_id:
                        self.coordinator.record_browser_recovery_event(
                            run_id=run_id,
                            status="failed",
                            payload={"attempt": attempt, "error": str(exc)},
                        )

            if recovered:
                if self.coordinator and run_id:
                    self.coordinator.record_browser_recovery_event(
                        run_id=run_id,
                        status="succeeded",
                        payload={"attempt": attempt},
                    )
                return True

        if self.coordinator and run_id:
            self.coordinator.record_browser_recovery_event(
                run_id=run_id,
                status="abandoned",
                payload={"stop_reason": RunStopReason.BROWSER_DISCONNECT_UNRECOVERED},
            )
        return False
