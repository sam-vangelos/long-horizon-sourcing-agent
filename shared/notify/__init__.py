"""Tell the operator something went wrong: macOS notification always, Slack DM when configured.

The single entry point is :func:`notify_sam`. It never raises. See
``shared/notify/notifier.py`` for transports, dedupe, and the secret-handling
rules, and ``docs/watchdog.md`` for setup.
"""

from .notifier import already_sent, default_state_dir, notify_sam, retry_pending

__all__ = ["already_sent", "default_state_dir", "notify_sam", "retry_pending"]
