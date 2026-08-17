"""Test helpers for the conversational intake suite (Phase C9 + C10).

This subpackage holds shared helpers — currently
:mod:`tests.intake_conversation.voice_asserts` for voice-property
checks against Cloris's outputs. Helpers are imported by the
behavioral transcript tests (``tests/test_intake_conversation_transcripts.py``)
and can be used directly in any test that needs voice-grounded
assertions.

Not a test module itself; pytest's discovery still operates on
``tests/test_*.py`` files.
"""
