from linkedin.activity_parser import (
    extract_profile_recent_activity_lines,
    extract_profile_status_summary,
    extract_recruiter_activity_from_card_text,
)


def test_extract_recruiter_activity_from_card_text_parses_list_view_counts():
    text = """
    Activity 9 messages · In 3 projects · 3 views
    Saved by Sam Vangelos on April 11, 2026
    """

    activity = extract_recruiter_activity_from_card_text(text)

    assert activity.message_count == 9
    assert activity.project_count == 3
    assert activity.view_count == 3
    assert activity.saved_by == "Sam Vangelos"


def test_extract_profile_recent_activity_lines_reads_recent_activity_block():
    text = """
    Most recent activity
    Viewed by Sam Vangelos
    Sam Vangelos added candidate to FDE - Leah
    Sam Vangelos archived candidate from Head of Applied AI Lab
    Summary
    """

    lines = extract_profile_recent_activity_lines(text)

    assert lines == [
        "Viewed by Sam Vangelos",
        "Sam Vangelos added candidate to FDE - Leah",
        "Sam Vangelos archived candidate from Head of Applied AI Lab",
    ]


def test_extract_profile_recent_activity_lines_tolerates_header_variants():
    text = """
    Most recent activity ·
    Viewed by Sam Vangelos
    Projects
    """

    lines = extract_profile_recent_activity_lines(text)

    assert lines == ["Viewed by Sam Vangelos"]


def test_extract_profile_recent_activity_lines_does_not_stop_on_message_content():
    text = """
    Most recent activity
    Message from recruiter about onsite
    Summary
    """

    lines = extract_profile_recent_activity_lines(text)

    assert lines == ["Message from recruiter about onsite"]


def test_extract_profile_status_summary_captures_outbound_status():
    text = """
    Last Outbound Contact
    1 month ago
    Isabella Lopes sent email stage of a sequence
    Sequences
    Isabella Lopes's sequence FDE
    Projects
    """

    summary = extract_profile_status_summary(text)

    assert summary["last_outbound_contact"].startswith("1 month ago")
    assert summary["reachout_status"] == "recent_outbound_contact"
    assert summary["sequences"] == ["Isabella Lopes's sequence FDE"]
