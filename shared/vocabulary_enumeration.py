"""Ask the model to ENUMERATE a domain's named artifacts, as search vocabulary.

Why this exists (measured 2026-07-27, brief 3000000001 "Principal Research
Engineer - Code"). The strategy step produced 72 search strings whose entire
benchmark vocabulary was nine names — SWE-bench, LiveCodeBench, HumanEval, MBPP,
BigCodeBench, terminal-bench, Codeforces, SWE-Gym, SWE-agent — every one
head-of-distribution. The operator immediately named five it had missed.

The cause was NOT missing knowledge. Asked directly, with the same capability
areas the strategy step already receives, claude-fable-5 returned 292 named
artifacts including 98 benchmarks, among them SWE-bench Verified, Terminal-Bench
2.0, SWE-smith, SWE-Lancer, R2E-Gym, SWE-rebench, SWE-Perf, SWT-Bench and
Multi-SWE-bench. It knows the long tail. Nothing ever asked for it: the strategy
call asks for SEARCH STRINGS, the model produces good search strings, and it
stops — satisficing, not ignorance.

So this is a separate call with a separate job. Enumeration is not string
formation, and asking one generation to do both is what produced nine names.

The output rides the EXISTING vocabulary channel — ``form_strategy(brief,
kit_strings, ...)`` — rather than a new one. That argument is already
structurally separate from the brief, already rendered into the prompt as
"raw building blocks", and already lint-covered by ``summarize_kit_lint``. On
this campaign it was empty ("No kit URL provided"), so this fills a socket the
design already assumes is full.

Register discipline (feedback_vocabulary_channels_audit): every channel into
formation needs a register audit at design time, or it launders paper-vocabulary
into search strings. Both models rate each artifact's ``on_profile`` register
unprompted. That rating is carried through as the subblock LABEL rather than
used as a drop filter — the rare-on-a-profile names are exactly the marginal
candidates this work exists to catch, so dropping them would invert the goal.
What IS dropped is ``certainty == "unsure"``: the model's own
did-I-invent-this flag, which is a precision control, not a register one.

Fail-soft end to end. Every failure path returns ``[]``, which is byte-identical
to the current no-kit behaviour.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from shared import config
from shared.llm_usage import openai_usage_dict, record_llm_usage
from shared.schemas import KitString

# Families the enumeration covers. Order is the render order in the prompt, so
# it runs most-searchable first: a benchmark name on a profile is a far stronger
# signal than an employer name, which the company facet already handles.
FAMILY_LABELS: dict[str, str] = {
    "benchmarks": "Named benchmarks and evals",
    "rl_environments": "RL environments and task suites",
    "training_frameworks": "Post-training and RL frameworks",
    "agent_harnesses": "Agent harnesses and scaffolds",
    "datasets": "Named datasets and corpora",
    "orgs": "Labs and companies",
}

REGISTER_LABELS: dict[str, str] = {
    "common": "Common on profiles",
    "occasional": "Occasional on profiles",
    "rare": "Rare on profiles",
}

# Vocabulary is never executed directly (the strategy prompt says so
# explicitly), but an OR group the strategist might paste should still sit near
# the executable lint's sensibility — _check_overlong_or_groups warns above 8.
MAX_TERMS_PER_GROUP = 10

# A double quote inside a term closes the term early and corrupts the group.
# Verified against the production compiler: '("SWE-bench" OR "quote"inside")'
# lints as unbalanced_quote AND unbalanced_parenthesis. Zero real casualties —
# no name in the 514-artifact live enumeration contained one.
#
# Parentheses are deliberately NOT refused. The compiler is quote-aware and
# round-trips '("Berkeley Function-Calling Leaderboard (BFCL)" OR "SWE-bench")'
# byte-identically with no lint findings, so an earlier guard here was solving
# a problem that did not exist — and it silently dropped "Model Context
# Protocol (MCP)", whose acronym this very JD names ("MCP-based Environments").
_UNSAFE_IN_BOOLEAN = re.compile(r'"')

# A glossed name carries TWO real surface forms and matches on neither: nobody
# writes "Model Context Protocol (MCP)" on a profile, they write one or the
# other. Splitting is parsing, not invention — both strings are literally
# present in what the model returned. Requires whitespace before the paren, so
# stylized names like "BigO(Bench)" are left whole rather than shredded into
# "BigO" and "Bench".
_GLOSSED_NAME = re.compile(r"^(.+?)\s+\(([^()]+)\)$")


def _surface_forms(name: str) -> list[str]:
    """The searchable strings inside one enumerated name."""
    match = _GLOSSED_NAME.match(name.strip())
    if not match:
        return [name.strip()]
    long_form, gloss = match.group(1).strip(), match.group(2).strip()
    return [form for form in (long_form, gloss) if len(form) > 1]

ENUMERATION_SYSTEM = """You are a technical sourcing researcher. You enumerate \
the named artifacts of a technical domain so a recruiter can search for the \
people who built them.

Two rules govern every answer.

Exhaustiveness over safety. A list of the six most famous names is a failure. \
The value is entirely in the long tail — the artifact that only forty people \
worldwide have worked on is worth more to a search than the one everybody \
lists. Reach for the obscure, the recent, the superseded, and the \
lab-internal-turned-public.

Honesty about recall. You will be scored on precision as well as recall, so \
mark anything you are not certain exists. Never invent a plausible-sounding \
name to lengthen the list.

Return JSON only. No prose, no markdown fences."""


def _enumeration_user_prompt(domain_context: str, scope_notes: str) -> str:
    return f"""<domain>
{domain_context}
</domain>

<scope>
{scope_notes}

An artifact belonging ONLY to an out-of-scope area is noise, not long tail —
omit it. An artifact that spans both stays in.
</scope>

<task>
Enumerate every NAMED artifact in this domain — proper nouns a practitioner \
could plausibly have written on their LinkedIn profile or in a paper.

Cover these families:
  benchmarks          (including every named VARIANT and version of each one)
  rl_environments     (environment suites, gyms, task collections)
  training_frameworks (post-training / RL / fine-tuning libraries)
  agent_harnesses     (scaffolds, agent frameworks, terminal/repo harnesses)
  datasets            (training / eval corpora)
  orgs                (labs and companies whose people build these)

For EACH artifact give:
  name       exact canonical spelling
  family     one of the six above
  year       first public release year, or null if unsure
  certainty  "certain" | "likely" | "unsure" that this artifact exists
  on_profile "common" | "occasional" | "rare" — how often a practitioner would
             actually write this string on a LinkedIn profile, as opposed to it
             living only in papers and leaderboards

Be exhaustive on benchmarks and their variants specifically. If a benchmark has \
Verified / Lite / Multimodal / v2 / Pro / Hard variants, list each as its own entry.

Return: {{"artifacts": [{{"name":..., "family":..., "year":..., "certainty":..., "on_profile":...}}]}}
</task>"""


@dataclass(frozen=True)
class EnumeratedArtifact:
    name: str
    family: str
    certainty: str
    on_profile: str
    year: int | None = None
    source: str = "parametric"

    @property
    def dedupe_key(self) -> str:
        return " ".join(self.name.lower().split())


def _field(obj: Any, name: str) -> Any:
    """Attribute or key, whichever this brief representation uses."""
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        value = obj.get(name)
    return value


def domain_context_from_brief(brief: Any) -> tuple[str, str]:
    """What is IN this domain, and what is explicitly OUT of it.

    In-scope is the capability areas — deliberately the same context
    ``form_strategy`` already renders, so a thin enumeration means the prompt is
    at fault rather than the brief.

    Out-of-scope needs its own sourcing, found by verifying the live seam on
    2026-07-27: a preflight-GENERATED brief has no ``instructions`` key at all
    (the seed's four operator instructions are absorbed during generation), so
    reading only that field handed the enumeration an empty exclusion set. The
    operator's actual exclusions — "generic RL, robotics, sim-to-real, and
    autonomous-vehicles RL are out of scope" — survive in ``non_fit_patterns``.
    Without them the enumeration is free to return the robotics-RL artifacts
    this role specifically does not want.
    """
    lines: list[str] = []
    for area in _field(brief, "capability_areas") or ():
        name = _field(area, "name")
        if not name:
            continue
        line = f"- {name}: {_field(area, 'description') or ''}"
        key_terms = _field(area, "key_terms")
        if key_terms:
            line += f"\n  key terms: {', '.join(str(t) for t in key_terms)}"
        lines.append(line)

    scope: list[str] = []
    raw_instructions = _field(brief, "instructions") or ()
    if isinstance(raw_instructions, str):
        raw_instructions = [raw_instructions]
    scope.extend(str(i) for i in raw_instructions if str(i).strip())

    for pattern in _field(brief, "non_fit_patterns") or ():
        description = _field(pattern, "description") or _field(pattern, "name")
        if description:
            scope.append(f"OUT OF SCOPE: {description}")

    return "\n".join(lines), "\n".join(scope)


def _coerce_artifacts(payload: Any, *, source: str) -> list[EnumeratedArtifact]:
    """Parse a provider response into artifacts, discarding anything malformed.

    Tolerant by design: this is an enrichment channel, and one bad row must not
    cost the whole enumeration.
    """
    if isinstance(payload, str):
        text = re.sub(r"^\s*```(?:json)?|```\s*$", "", payload.strip(), flags=re.M)
        try:
            payload = json.loads(text.strip())
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return []
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
    if not isinstance(payload, dict):
        return []

    out: list[EnumeratedArtifact] = []
    for row in payload.get("artifacts") or ():
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        family = str(row.get("family") or "").strip().lower()
        if not name or family not in FAMILY_LABELS:
            continue
        certainty = str(row.get("certainty") or "").strip().lower()
        register = str(row.get("on_profile") or "").strip().lower()
        if register not in REGISTER_LABELS:
            register = "occasional"
        year = row.get("year")
        out.append(
            EnumeratedArtifact(
                name=name,
                family=family,
                certainty=certainty or "likely",
                on_profile=register,
                year=year if isinstance(year, int) and not isinstance(year, bool) else None,
                source=source,
            )
        )
    return out


def _is_searchable(artifact: EnumeratedArtifact) -> bool:
    """Drop what cannot become a Boolean term or should not be trusted as one.

    ``unsure`` is the model's own did-I-invent-this flag — a precision control.
    Register is NOT filtered here: rare-on-a-profile is the marginal candidate
    this whole channel exists to reach.
    """
    if artifact.certainty == "unsure":
        return False
    if _UNSAFE_IN_BOOLEAN.search(artifact.name):
        return False
    # A one-character "name" is punctuation noise, not an artifact.
    return len(artifact.name.strip()) > 1


def _expand_glosses(
    artifacts: Iterable[EnumeratedArtifact],
) -> list[EnumeratedArtifact]:
    """Replace a glossed artifact with one entry per real surface form."""
    out: list[EnumeratedArtifact] = []
    for artifact in artifacts:
        forms = _surface_forms(artifact.name)
        if len(forms) == 1 and forms[0] == artifact.name:
            out.append(artifact)
            continue
        for form in forms:
            out.append(
                EnumeratedArtifact(
                    name=form,
                    family=artifact.family,
                    certainty=artifact.certainty,
                    on_profile=artifact.on_profile,
                    year=artifact.year,
                    source=artifact.source,
                )
            )
    return out


def merge_artifacts(
    *batches: Iterable[EnumeratedArtifact],
) -> list[EnumeratedArtifact]:
    """Union across providers, first writer wins on a name collision.

    The 2026-07-27 measurement: Fable 292, Perplexity 222, overlap only 106.
    The providers are substantially complementary, so the union is worth the
    second call even though neither alone found the newest three artifacts.
    """
    seen: dict[str, EnumeratedArtifact] = {}
    for batch in batches:
        for artifact in _expand_glosses(batch):
            if not _is_searchable(artifact):
                continue
            seen.setdefault(artifact.dedupe_key, artifact)
    return sorted(seen.values(), key=lambda a: (a.family, a.on_profile, a.name.lower()))


def artifacts_to_kit_strings(
    artifacts: list[EnumeratedArtifact],
    *,
    max_terms_per_group: int = MAX_TERMS_PER_GROUP,
) -> list[KitString]:
    """Group artifacts into the vocabulary shape the strategy prompt renders.

    One KitString per (family, register) chunk. ``subblock`` carries the
    register so the strategist can see which names are safe anchors and which
    are high-precision probes, and ``string_type`` is always Precision — a
    named artifact is the opposite of a recall term.
    """
    grouped: dict[tuple[str, str], list[str]] = {}
    for artifact in artifacts:
        grouped.setdefault((artifact.family, artifact.on_profile), []).append(artifact.name)

    kit: list[KitString] = []
    next_id = 1
    for family in FAMILY_LABELS:
        for register in ("common", "occasional", "rare"):
            names = grouped.get((family, register)) or []
            for start in range(0, len(names), max_terms_per_group):
                chunk = names[start : start + max_terms_per_group]
                boolean = "(" + " OR ".join(f'"{name}"' for name in chunk) + ")"
                kit.append(
                    KitString(
                        id=next_id,
                        block=FAMILY_LABELS[family],
                        subblock=REGISTER_LABELS[register],
                        string_type="Precision",
                        boolean=boolean,
                    )
                )
                next_id += 1
    return kit


def default_research_call(system: str, user: str) -> Any:
    """External-research enumeration through the configured provider.

    Mirrors ``market_intelligence.research_agent``'s Perplexity call shape —
    OpenAI SDK against api.perplexity.ai, responses.create with web_search plus
    fetch_url and a json_schema response format. Raises on any failure; the
    caller treats a research failure as "no second batch", never as a run error.
    """
    from openai import OpenAI

    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "domain_artifact_enumeration",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "artifacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "family": {
                                    "type": "string",
                                    "enum": list(FAMILY_LABELS),
                                },
                                "year": {"type": ["integer", "null"]},
                                "certainty": {
                                    "type": "string",
                                    "enum": ["certain", "likely", "unsure"],
                                },
                                "on_profile": {
                                    "type": "string",
                                    "enum": list(REGISTER_LABELS),
                                },
                            },
                            "required": [
                                "name", "family", "year", "certainty", "on_profile",
                            ],
                        },
                    }
                },
                "required": ["artifacts"],
            },
        },
    }
    client = OpenAI(
        api_key=config.PERPLEXITY_API_KEY,
        base_url="https://api.perplexity.ai/v1",
        timeout=config.MARKET_INTEL_EXTERNAL_RESEARCH_TIMEOUT_SECONDS,
    )
    instructions = (
        system
        + "\n\nYou have web search. Use it aggressively — leaderboards, arXiv "
        "listings, GitHub org pages, and release announcements from the last "
        "18 months are where the long tail lives."
    )
    response = client.responses.create(
        input=user,
        instructions=instructions,
        max_output_tokens=32768,
        tools=[
            {"type": "web_search", "filters": {"search_recency_filter": "year"}},
            {"type": "fetch_url"},
        ],
        extra_body={"preset": "deep-research", "response_format": schema},
    )
    try:
        record_llm_usage(
            provider="perplexity",
            model=str(getattr(response, "model", "") or "").strip()
            or "perplexity-response-api",
            usage=openai_usage_dict(response),
            request={
                "max_tokens": 32768,
                "instructions_chars": len(instructions),
                "input_chars": len(user),
            },
            usage_context={
                "stage": "vocabulary_enumeration",
                "provider_preset": "deep-research",
            },
        )
    except Exception:  # noqa: BLE001 — telemetry must not break enumeration
        pass
    text = getattr(response, "output_text", None)
    if text:
        return text
    dumped = response.model_dump() if hasattr(response, "model_dump") else {}
    chunks: list[str] = []
    for item in dumped.get("output") or ():
        if not isinstance(item, dict):
            continue
        for block in item.get("content") or ():
            if isinstance(block, dict) and block.get("text"):
                chunks.append(str(block["text"]))
    return "\n".join(chunks)


def enumerate_domain_vocabulary(
    brief: Any,
    *,
    llm_call: Optional[Callable[..., Any]] = None,
    research_call: Optional[Callable[[str, str], Any]] = None,
    artifact_dir: Path | None = None,
) -> list[KitString]:
    """Enumerate the domain's named artifacts as strategy vocabulary.

    Returns ``[]`` on any failure, which is exactly today's no-kit behaviour.
    ``llm_call`` and ``research_call`` are injected by tests; production passes
    neither and gets the strategy-tier model plus, when configured, the
    external research provider.
    """
    domain_context, instructions = domain_context_from_brief(brief)
    if not domain_context.strip():
        return []

    user_prompt = _enumeration_user_prompt(domain_context, instructions)
    batches: list[list[EnumeratedArtifact]] = []

    try:
        invoke = llm_call
        if invoke is None:
            from shared.llm_clients import opus_llm_cached

            def invoke(system: str, user: str) -> Any:  # type: ignore[misc]
                return opus_llm_cached(
                    system,
                    user,
                    expect_json=False,
                    max_tokens=32768,
                    model_name=config.STRATEGY_MODEL_NAME,
                    usage_context={
                        "stage": "linkedin_vocabulary_enumeration",
                        "brief_id": getattr(brief, "id", ""),
                    },
                )

        batches.append(
            _coerce_artifacts(
                invoke(ENUMERATION_SYSTEM, user_prompt), source="parametric"
            )
        )
    except Exception as exc:  # noqa: BLE001 — enrichment must never break a run
        print(f"  [warn] vocabulary enumeration failed, continuing without it: {exc}")

    if research_call is not None:
        try:
            batches.append(
                _coerce_artifacts(
                    research_call(ENUMERATION_SYSTEM, user_prompt), source="research"
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] vocabulary research pass failed, continuing: {exc}")

    artifacts = merge_artifacts(*batches)
    if not artifacts:
        return []

    kit = artifacts_to_kit_strings(artifacts)

    if artifact_dir is not None:
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "vocabulary_enumeration.json").write_text(
                json.dumps(
                    {
                        "artifact_count": len(artifacts),
                        "kit_string_count": len(kit),
                        "by_family": {
                            family: sum(1 for a in artifacts if a.family == family)
                            for family in FAMILY_LABELS
                        },
                        "by_register": {
                            register: sum(1 for a in artifacts if a.on_profile == register)
                            for register in REGISTER_LABELS
                        },
                        "artifacts": [
                            {
                                "name": a.name,
                                "family": a.family,
                                "year": a.year,
                                "certainty": a.certainty,
                                "on_profile": a.on_profile,
                                "source": a.source,
                            }
                            for a in artifacts
                        ],
                    },
                    indent=2,
                )
            )
        except Exception as exc:  # noqa: BLE001 — a receipt must not break a run
            print(f"  [warn] could not write vocabulary receipt: {exc}")

    return kit
