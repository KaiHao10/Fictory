from pathlib import Path

from .. import llm
from ..models.handoff import CharacterHandoff, HandoffDocument
from ..models.story_bible import (
    Character,
    CharacterState,
    CharacterStateSnapshot,
    ContractCheckResult,
    PlannedDelta,
    PlannedKnowledgeDelivery,
    SceneEvaluation,
    SceneOutline,
    StoryBible,
    VoiceExample,
)


def _snapshot_state(ch: Character, scene_number: int) -> None:
    """Record a snapshot of `ch.state` keyed by scene_number into state_history.
    Use scene_number=0 for the initial seeded state, N for the end of scene N."""
    ch.state_history[scene_number] = CharacterStateSnapshot(
        knowledge=list(ch.state.knowledge),
        emotional_state=ch.state.emotional_state,
    )


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
                    # No backfill repairs absence-from-scene; flag and move on.
                    continue
                if fact in world or fact in projected[deliverer]:
                    continue  # precondition met, nothing to do
                # Seed-backfill: the deliverer must know this from the start.
                bible.characters[deliverer].state.knowledge.append(fact)
                seed_snap = bible.characters[deliverer].state_history.get(0)
                if seed_snap is not None:
                    seed_snap.knowledge.append(fact)
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

PROMPTS = Path(__file__).parent.parent / "prompts" / "orchestrator"

VOICE_RECENT_CAP = 10
VOICE_BEST_CAP = 5


def _load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text()


def generate_outline(premise: str, genre: str, num_scenes: int) -> StoryBible:
    system = _load_prompt("generate_outline.txt").replace("NUM_SCENES", str(num_scenes))
    user = (
        f"Genre: {genre}\n"
        f"Number of scenes: {num_scenes}\n"
        f"Premise: {premise}\n\n"
        f"Generate the JSON now."
    )
    data = llm.chat_json(system=system, user=user, max_tokens=16384, temperature=0.7)
    bible = StoryBible.model_validate(data)
    # Snapshot the initial seeded state for every character at scene 0.
    for ch in bible.characters.values():
        _snapshot_state(ch, 0)
    # Normalize paraphrased fact strings BEFORE validation so the validator's
    # exact-match checks operate on canonical wordings.
    bible.normalize_log = normalize_fact_strings(bible)
    # Validate the outline's contract chain and seed-backfill any unearned
    # deliveries so the writer never receives an impossible contract.
    bible.backfill_log = validate_and_backfill(bible)
    # Re-snapshot scene 0: normalization may have rewritten seeded knowledge
    # strings, and backfill may have added new ones.
    for ch in bible.characters.values():
        _snapshot_state(ch, 0)
    return bible


def summarize_scene(prose: str) -> str:
    system = _load_prompt("summarize_scene.txt")
    return llm.chat(system=system, user=prose, max_tokens=300, temperature=0.3)


def assemble_handoff(bible: StoryBible, scene_number: int) -> HandoffDocument:
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
    prior = bible.scene_summaries.get(scene_number - 1) if scene_number > 1 else None
    return HandoffDocument(
        scene_outline=scene,
        characters=char_handoffs,
        world_rules=list(bible.world_rules),
        prior_scene_summary=prior,
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


def update_bible(bible: StoryBible, scene_number: int, prose: str) -> StoryBible:
    scene = next(s for s in bible.scenes if s.scene_number == scene_number)
    present = {n: bible.characters[n] for n in scene.characters_present if n in bible.characters}

    # Build a per-character block that includes the contract for this scene.
    char_blocks: list[str] = []
    for name, ch in present.items():
        delta = scene.planned_deltas.get(name)
        if delta and delta.knowledge_to_gain:
            fact_lines = []
            for d in delta.knowledge_to_gain:
                attribution = f"source={d.source}"
                if d.delivered_by:
                    attribution += f", delivered_by={d.delivered_by}"
                if d.via:
                    attribution += f", via={d.via}"
                fact_lines.append(f"      - [{attribution}] {d.fact}")
            facts_block = "\n".join(fact_lines)
            contract_text = (
                f"  CONTRACT for this scene:\n"
                f"    knowledge_to_gain:\n{facts_block}\n"
                f"    target_emotional_state: {delta.target_emotional_state}\n"
            )
        elif delta:
            contract_text = (
                f"  CONTRACT for this scene:\n"
                f"    knowledge_to_gain: (none)\n"
                f"    target_emotional_state: {delta.target_emotional_state}\n"
            )
        else:
            contract_text = "  CONTRACT for this scene: (none — character is present but state should not change)\n"
        char_blocks.append(
            f"--- {name} ---\n"
            f"  profile: {ch.profile}\n"
            f"  current knowledge: {ch.state.knowledge}\n"
            f"  current emotional_state: {ch.state.emotional_state}\n"
            f"{contract_text}"
        )
    state_block = "\n".join(char_blocks)

    system = _load_prompt("update_bible.txt")
    user = (
        f"SCENE OUTLINE (scene {scene_number}):\n"
        f"summary: {scene.summary}\n"
        f"beats: {scene.beats}\n"
        f"arc_position: {scene.arc_position}\n\n"
        f"CURRENT WORLD RULES:\n- " + "\n- ".join(bible.world_rules) + "\n\n"
        f"CHARACTERS IN SCENE (with contracts):\n{state_block}\n\n"
        f"SCENE PROSE:\n{prose}\n\n"
        f"Return the JSON update object now. Remember: PASS 1 verifies the contract, PASS 2 captures organic facts."
    )

    data = llm.chat_json(system=system, user=user, max_tokens=8192, temperature=0.3)

    char_updates = data.get("character_updates", {}) or {}

    # Build the SceneEvaluation as we go.
    evaluation = SceneEvaluation(scene_number=scene_number)

    for name, upd in char_updates.items():
        ch = bible.characters.get(name)
        if ch is None:
            continue

        # --- PASS 1 results: contract check ---
        contract_results: list[ContractCheckResult] = []
        for r in upd.get("contract_results") or []:
            fact = (r.get("planned_fact") or "").strip()
            if not fact:
                continue
            contract_results.append(
                ContractCheckResult(
                    planned_fact=fact,
                    landed=bool(r.get("landed", False)),
                    evidence=(r.get("evidence") or None),
                )
            )
        evaluation.contract_results[name] = contract_results
        evaluation.emotional_state_landed[name] = bool(upd.get("emotional_state_landed", False))

        # --- Apply: contracted facts that landed go into knowledge ---
        existing = set(ch.state.knowledge)
        for cr in contract_results:
            if cr.landed and cr.planned_fact not in existing:
                ch.state.knowledge.append(cr.planned_fact)
                existing.add(cr.planned_fact)

        # --- PASS 2 results: organic facts ---
        organic = [k for k in (upd.get("organic_knowledge") or []) if k]
        evaluation.organic_knowledge[name] = list(organic)
        for k in organic:
            if k not in existing:
                ch.state.knowledge.append(k)
                existing.add(k)

        # --- Emotional state replacement ---
        new_emo = upd.get("emotional_state")
        if new_emo:
            ch.state.emotional_state = new_emo
        ch.state.last_updated_scene = scene_number

        # --- Voice examples ---
        for ex in upd.get("voice_examples") or []:
            line = ex.get("line", "").strip()
            if not line:
                continue
            ch.voice_examples.append(
                VoiceExample(
                    line=line,
                    scene_number=scene_number,
                    is_best=bool(ex.get("is_best", False)),
                )
            )
        _trim_voice_bank(ch)

    for fact in data.get("world_rules_added") or []:
        if fact and fact not in bible.world_rules:
            bible.world_rules.append(fact)

    # Persist the per-scene evaluation.
    bible.scene_evaluations[scene_number] = evaluation

    # Snapshot end-of-scene state for every character (not just those who
    # appeared this scene — so absent characters still have a frame for this
    # scene number, which simplifies diff/eval logic).
    for ch in bible.characters.values():
        _snapshot_state(ch, scene_number)

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
