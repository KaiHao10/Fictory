from .agents import evaluator, orchestrator, writer
from . import storage
from .models.story_bible import StoryBible

MAX_RETRIES = 2  # up to MAX_RETRIES + 1 writer attempts per scene before accepting


def _banner(text: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{text}\n{bar}")


def run(premise: str, genre: str, num_scenes: int, story_id: str) -> StoryBible:
    existing = (storage.STORIES_DIR / story_id / "bible.json")
    if existing.exists():
        bible = storage.load(story_id)
        start_at = bible.current_scene
        _banner(f"RESUMING '{story_id}' from scene {start_at} (of {len(bible.scenes)})")
    else:
        _banner(f"SYNOPSIS — drafting for '{story_id}'")
        synopsis = orchestrator.generate_synopsis(premise, genre, num_scenes)
        print(synopsis.model_dump_json(indent=2))

        _banner(f"OUTLINE — generating scene grid for '{story_id}' ({num_scenes} scenes)")
        bible, normalize_log, backfill_log = orchestrator.generate_outline(
            premise, genre, num_scenes, synopsis
        )
        storage.save(story_id, bible)
        print(bible.model_dump_json(indent=2))

        if normalize_log:
            _banner(f"OUTLINE NORMALIZATION — {len(normalize_log)} substitutions")
            for entry in normalize_log:
                print(f"  • {entry}")
        else:
            _banner("OUTLINE NORMALIZATION — no paraphrases found")
        if backfill_log:
            _banner(f"OUTLINE VALIDATION — {len(backfill_log)} repairs/warnings")
            for entry in backfill_log:
                print(f"  • {entry}")
        else:
            _banner("OUTLINE VALIDATION — clean (no backfills needed)")
        start_at = 1

    num_scenes = len(bible.scenes)
    for n in range(start_at, num_scenes + 1):
        prose, verdict = _write_with_retries(bible, n)

        bible.generated_prose[n] = prose

        _banner(f"SCENE {n} — SUMMARY")
        summary = orchestrator.summarize_scene(prose)
        bible.scene_summaries[n] = summary
        print(summary)

        _banner(f"SCENE {n} — STORY SO FAR (update)")
        bible.story_so_far = orchestrator.update_story_so_far(bible, n, prose)
        print(bible.story_so_far)

        _banner(f"SCENE {n} — APPLY VERDICT TO BIBLE")
        bible = orchestrator.apply_scene(bible, n, verdict)
        bible.current_scene = n + 1
        storage.save(story_id, bible)

        for name, ch in bible.characters.items():
            print(f"\n[{name}] (last_updated_scene={ch.state.last_updated_scene})")
            print(f"  knowledge: {ch.state.knowledge}")
            print(f"  emotional_state: {ch.state.emotional_state}")
            print(f"  voice_examples ({len(ch.voice_examples)}):")
            for ex in ch.voice_examples:
                star = "★" if ex.is_best else " "
                print(f"    {star} [s{ex.scene_number}] {ex.line}")

        print(f"\n[fictory] scene {n} done ({len(prose.split())} words).")

    _banner(f"COMPLETE — output in stories/{story_id}/")
    return bible


def _write_with_retries(bible: StoryBible, n: int):
    """Run writer → evaluator up to MAX_RETRIES+1 times. Returns (prose, verdict)
    from the final attempt — passing on success, last attempt accepted otherwise."""
    rewrite_notes = None
    prose = ""
    verdict = None
    for attempt in range(MAX_RETRIES + 1):
        label = "DRAFT" if attempt == 0 else f"REVISION {attempt}"
        _banner(f"SCENE {n} — HANDOFF ({label})")
        handoff = orchestrator.assemble_handoff(bible, n, rewrite_notes=rewrite_notes)
        print(handoff.model_dump_json(indent=2))

        _banner(f"SCENE {n} — PROSE ({label})")
        prose = writer.write_scene(handoff)
        print(prose)

        _banner(f"SCENE {n} — EVALUATION ({label})")
        verdict = evaluator.evaluate_scene(bible, n, prose)
        _print_verdict(verdict)

        if verdict.passed:
            return prose, verdict

        if attempt < MAX_RETRIES:
            print(f"\n[fictory] scene {n} failed evaluation; retrying with rewrite notes.")
            rewrite_notes = verdict.notes_for_rewrite
        else:
            print(
                f"\n[fictory] WARNING: scene {n} accepted after {MAX_RETRIES + 1} attempts "
                f"without passing evaluation. Continuing with last draft."
            )
    return prose, verdict  # type: ignore[return-value]


def _print_verdict(verdict) -> None:
    print(f"passed: {verdict.passed}")
    for name, cv in verdict.per_character.items():
        landed = sum(1 for v in cv.facts_landed.values() if v)
        total = len(cv.facts_landed)
        emo_mark = "✓" if cv.emotional_state_landed else "✗"
        print(f"  [{name}] knowledge {landed}/{total}, emotional_state {emo_mark}")
        for fact, ok in cv.facts_landed.items():
            mark = "✓" if ok else "✗"
            print(f"    {mark} {fact}")
            if ok and (ev := cv.facts_evidence.get(fact)):
                print(f"        evidence: {ev}")
    if verdict.overcreativity_flags:
        print("OVERCREATIVITY:")
        for f in verdict.overcreativity_flags:
            print(f"  ! {f}")
    if verdict.coherence_flags:
        print("COHERENCE:")
        for f in verdict.coherence_flags:
            print(f"  ! {f}")
    if not verdict.passed and verdict.notes_for_rewrite:
        print(f"NOTES FOR REWRITE:\n{verdict.notes_for_rewrite}")
