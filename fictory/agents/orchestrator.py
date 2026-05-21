from pathlib import Path

from .. import llm
from ..models.handoff import CharacterHandoff, HandoffDocument
from ..models.story_bible import (
    Character,
    StoryBible,
    Synopsis,
    VoiceExample,
)
from ..models.verdict import SceneVerdict

PROMPTS = Path(__file__).parent.parent / "prompts" / "orchestrator"

VOICE_RECENT_CAP = 10
VOICE_BEST_CAP = 5


def _load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def _dedup(xs: list[str]) -> list[str]:
    """Order-preserving deduplication."""
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _collect_all_facts(bible: StoryBible) -> list[str]:
    """Gather every unique fact string from the outline (world_rules,
    character.state.knowledge, planned_deltas[*].knowledge_to_gain[*].fact).
    Order-preserving dedup."""
    out: list[str] = []
    out.extend(bible.world_rules)
    for ch in bible.characters.values():
        out.extend(ch.state.knowledge)
    for scene in bible.scenes:
        for delta in scene.planned_deltas.values():
            for d in delta.knowledge_to_gain:
                out.append(d.fact)
    return _dedup(out)


def normalize_fact_strings(bible: StoryBible) -> list[str]:
    """Cluster paraphrased fact strings across the outline and replace each
    paraphrase with a chosen canonical wording. Mutates the bible in place.

    Returns a list of human-readable normalization log entries.
    """
    facts = _collect_all_facts(bible)
    if len(facts) < 2:
        return []

    system = _load_prompt("normalize_outline.txt")
    fact_listing = "\n".join(f"- {f}" for f in facts)
    user = (
        "Below is the full set of unique fact strings used across the outline "
        "(world_rules, character starting knowledge, and per-scene planned "
        "deliveries). Find any cluster of 2+ strings that describe the SAME "
        "underlying fact in different wordings.\n\n"
        f"FACTS:\n{fact_listing}\n\n"
        "Respond with ONLY the JSON object. Begin your response with the "
        "character `{` and end with `}`. No prose, no explanation, no markdown "
        "fences. If you find no paraphrase clusters, respond with: "
        '{"clusters": []}'
    )

    data: dict[str, object] | None = None
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            # Bump temperature on retries to escape the empty-response attractor.
            temp = 0.2 + 0.2 * attempt
            data = llm.chat_json(system=system, user=user, max_tokens=4096, temperature=temp)
            break
        except Exception as e:
            last_err = e
            print(f"[normalize] parse/LLM error (attempt {attempt + 1}/3, temp={temp}): {e}")
    if data is None:
        return [f"WARN normalize_fact_strings: LLM call failed after 3 attempts: {last_err}"]

    fact_set = set(facts)
    mapping: dict[str, str] = {}
    log: list[str] = []
    for cluster in data.get("clusters", []) or []:
        canonical = (cluster.get("canonical") or "").strip()
        members = cluster.get("members") or []
        if not canonical or canonical not in fact_set:
            log.append(f"WARN cluster skipped — canonical not found in input: '{canonical}'")
            continue
        for m in members:
            m = (m or "").strip()
            if not m or m == canonical:
                continue
            if m not in fact_set:
                log.append(f"WARN cluster member skipped — not in input: '{m}'")
                continue
            if m in mapping:
                continue  # already mapped by an earlier cluster; first-wins
            mapping[m] = canonical
            log.append(f"NORMALIZE: '{m}' → '{canonical}'")

    if not mapping:
        return log

    # Apply substitutions across the bible (with dedup of the resulting lists).
    bible.world_rules = _dedup([mapping.get(f, f) for f in bible.world_rules])
    for ch in bible.characters.values():
        ch.state.knowledge = _dedup([mapping.get(f, f) for f in ch.state.knowledge])
    for scene in bible.scenes:
        for delta in scene.planned_deltas.values():
            for d in delta.knowledge_to_gain:
                if d.fact in mapping:
                    d.fact = mapping[d.fact]

    return log


def validate_and_backfill(bible: StoryBible) -> list[str]:
    """Forward-simulate state through the outline to validate contract preconditions.

    For every fact whose source is `delivered_by_character`, the deliverer must
    project to know that fact at the start of the relevant scene. If they don't,
    we seed-backfill the fact into the deliverer's seeded state so they're
    entitled to deliver it.

    Returns a list of human-readable log entries for every backfill or warning.
    """
    log: list[str] = []
    world = set(bible.world_rules)

    # Projected knowledge per character (deep copy of seed state).
    projected: dict[str, set[str]] = {
        name: set(ch.state.knowledge) for name, ch in bible.characters.items()
    }

    for scene in sorted(bible.scenes, key=lambda s: s.scene_number):
        # First pass: validate every delivered_by_character precondition for this scene.
        for receiver_name, delta in scene.planned_deltas.items():
            for delivery in delta.knowledge_to_gain:
                if delivery.source != "delivered_by_character":
                    continue
                deliverer = delivery.delivered_by
                fact = delivery.fact
                if not deliverer:
                    log.append(
                        f"WARN scene {scene.scene_number}: fact '{fact}' for {receiver_name} "
                        f"marked delivered_by_character but no deliverer named — skipping check."
                    )
                    continue
                if deliverer not in bible.characters:
                    log.append(
                        f"WARN scene {scene.scene_number}: fact '{fact}' delivered_by '{deliverer}' "
                        f"who is not in characters{{}} — skipping check."
                    )
                    continue
                if deliverer not in scene.characters_present:
                    log.append(
                        f"WARN scene {scene.scene_number}: fact '{fact}' delivered_by '{deliverer}' "
                        f"who is not present in this scene — receiver {receiver_name} can't hear it."
                    )
                    continue
                if fact in world or fact in projected[deliverer]:
                    continue  # precondition met
                # Seed-backfill: the deliverer must know this from the start.
                bible.characters[deliverer].state.knowledge.append(fact)
                projected[deliverer].add(fact)
                log.append(
                    f"BACKFILL scene {scene.scene_number}: '{fact}' added to {deliverer}.seed "
                    f"(needed to deliver to {receiver_name})"
                )

        # Second pass: project contracted gains forward for receivers.
        for receiver_name, delta in scene.planned_deltas.items():
            for delivery in delta.knowledge_to_gain:
                projected[receiver_name].add(delivery.fact)

    return log


def generate_synopsis(premise: str, genre: str, num_scenes: int) -> Synopsis:
    system = _load_prompt("generate_synopsis.txt")
    user = (
        f"Genre: {genre}\n"
        f"Number of scenes the story will be told in: {num_scenes}\n"
        f"Premise: {premise}\n\n"
        f"Draft the synopsis JSON now."
    )
    data = llm.chat_json(system=system, user=user, max_tokens=2048, temperature=0.7)
    return Synopsis.model_validate(data)


def generate_outline(
    premise: str, genre: str, num_scenes: int, synopsis: Synopsis
) -> tuple[StoryBible, list[str], list[str]]:
    """Generate the scene-grid outline given a pre-drafted synopsis.

    Returns (bible, normalize_log, backfill_log). The logs are transient —
    pipeline prints them; they are not persisted on the bible.
    """
    system = _load_prompt("generate_outline.txt").replace("NUM_SCENES", str(num_scenes))
    synopsis_json = synopsis.model_dump_json(indent=2)
    user = (
        f"Genre: {genre}\n"
        f"Number of scenes: {num_scenes}\n"
        f"Premise: {premise}\n\n"
        f"SYNOPSIS (drafted in the previous step — your scene grid must land this):\n"
        f"{synopsis_json}\n\n"
        f"Generate the JSON now."
    )
    data = llm.chat_json(system=system, user=user, max_tokens=16384, temperature=0.7)
    # Ensure the synopsis we drafted is what's saved, regardless of what the LLM echoed.
    data["synopsis"] = synopsis.model_dump()
    bible = StoryBible.model_validate(data)
    # Normalize paraphrased fact strings BEFORE validation so the validator's
    # exact-match checks operate on canonical wordings.
    normalize_log = normalize_fact_strings(bible)
    # Validate the outline's contract chain and seed-backfill any unearned
    # deliveries so the writer never receives an impossible contract.
    backfill_log = validate_and_backfill(bible)
    return bible, normalize_log, backfill_log


def summarize_scene(prose: str) -> str:
    system = _load_prompt("summarize_scene.txt")
    return llm.chat(system=system, user=prose, max_tokens=300, temperature=0.3)


def update_story_so_far(bible: StoryBible, scene_number: int, prose: str) -> str:
    """Roll the 'story so far' narrative forward to include the scene just written."""
    system = _load_prompt("update_story_so_far.txt")
    previous = bible.story_so_far or "(no previous story-so-far — this is the first scene)"
    user = (
        f"SCENE NUMBER JUST CONCLUDED: {scene_number}\n\n"
        f"PREVIOUS STORY SO FAR:\n{previous}\n\n"
        f"PROSE OF SCENE {scene_number}:\n{prose}\n\n"
        f"Produce the updated story-so-far narrative now. Plain prose only."
    )
    return llm.chat(system=system, user=user, max_tokens=1500, temperature=0.3).strip()


def assemble_handoff(
    bible: StoryBible, scene_number: int, rewrite_notes: str | None = None
) -> HandoffDocument:
    scene = next(s for s in bible.scenes if s.scene_number == scene_number)
    char_handoffs: dict[str, CharacterHandoff] = {}
    for name in scene.characters_present:
        ch = bible.characters.get(name)
        if ch is None:
            continue
        char_handoffs[name] = CharacterHandoff(
            name=ch.name,
            profile=ch.profile,
            voice_examples=_select_voice_examples(ch),
            state=ch.state,
            planned_delta=scene.planned_deltas.get(name),
        )
    return HandoffDocument(
        scene_outline=scene,
        characters=char_handoffs,
        world_rules=list(bible.world_rules),
        story_so_far=bible.story_so_far,
        rewrite_notes=rewrite_notes,
    )


def _select_voice_examples(ch: Character) -> list[str]:
    """Return up to 10 most-recent + up to 5 best (deduped, best first)."""
    by_recent = sorted(ch.voice_examples, key=lambda e: e.scene_number, reverse=True)
    recent = by_recent[:VOICE_RECENT_CAP]
    best = [e for e in ch.voice_examples if e.is_best][:VOICE_BEST_CAP]
    seen: set[str] = set()
    out: list[str] = []
    for e in best + recent:
        if e.line in seen:
            continue
        seen.add(e.line)
        out.append(e.line)
    return out


def apply_scene(bible: StoryBible, scene_number: int, verdict: SceneVerdict) -> StoryBible:
    """Mechanically apply the verdict's findings to character state.

    State evolves strictly from the outline's planned_deltas, gated by the
    evaluator's per-fact and per-emotion landing decisions. No organic-fact
    channel from prose — the orchestrator is the only source of plot truth.
    """
    scene = next(s for s in bible.scenes if s.scene_number == scene_number)

    for name in scene.characters_present:
        ch = bible.characters.get(name)
        if ch is None:
            continue
        delta = scene.planned_deltas.get(name)
        cv = verdict.per_character.get(name)

        if delta and cv:
            existing = set(ch.state.knowledge)
            for delivery in delta.knowledge_to_gain:
                landed = cv.facts_landed.get(delivery.fact, False)
                if landed and delivery.fact not in existing:
                    ch.state.knowledge.append(delivery.fact)
                    existing.add(delivery.fact)
            if cv.emotional_state_landed and delta.target_emotional_state:
                ch.state.emotional_state = delta.target_emotional_state

        # Voice examples — appended from the evaluator's picks.
        if cv:
            for vd in cv.voice_examples:
                line = (vd.line or "").strip()
                if not line:
                    continue
                ch.voice_examples.append(
                    VoiceExample(
                        line=line,
                        scene_number=scene_number,
                        is_best=bool(vd.is_best),
                    )
                )
            _trim_voice_bank(ch)

        ch.state.last_updated_scene = scene_number

    return bible


def _trim_voice_bank(ch: Character) -> None:
    """Keep up to 10 most-recent + 5 best, deduped by line."""
    bests = sorted(
        (e for e in ch.voice_examples if e.is_best),
        key=lambda e: e.scene_number,
        reverse=True,
    )[:VOICE_BEST_CAP]
    recents = sorted(ch.voice_examples, key=lambda e: e.scene_number, reverse=True)[:VOICE_RECENT_CAP]
    seen: set[str] = set()
    kept: list[VoiceExample] = []
    for e in bests + recents:
        if e.line in seen:
            continue
        seen.add(e.line)
        kept.append(e)
    kept.sort(key=lambda e: e.scene_number)
    ch.voice_examples = kept
