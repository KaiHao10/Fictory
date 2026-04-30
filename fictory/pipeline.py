from .agents import orchestrator, writer
from . import storage
from .models.story_bible import StoryBible


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
        _banner(f"OUTLINE — generating for '{story_id}' ({num_scenes} scenes)")
        bible = orchestrator.generate_outline(premise, genre, num_scenes)
        storage.save(story_id, bible)
        print(bible.model_dump_json(indent=2))
        if bible.normalize_log:
            _banner(f"OUTLINE NORMALIZATION — {len(bible.normalize_log)} substitutions")
            for entry in bible.normalize_log:
                print(f"  • {entry}")
        else:
            _banner("OUTLINE NORMALIZATION — no paraphrases found")
        if bible.backfill_log:
            _banner(f"OUTLINE VALIDATION — {len(bible.backfill_log)} repairs/warnings")
            for entry in bible.backfill_log:
                print(f"  • {entry}")
        else:
            _banner("OUTLINE VALIDATION — clean (no backfills needed)")
        start_at = 1

    num_scenes = len(bible.scenes)
    for n in range(start_at, num_scenes + 1):
        _banner(f"SCENE {n} — HANDOFF")
        handoff = orchestrator.assemble_handoff(bible, n)
        print(handoff.model_dump_json(indent=2))

        _banner(f"SCENE {n} — PROSE")
        prose = writer.write_scene(handoff)
        bible.generated_prose[n] = prose
        print(prose)

        _banner(f"SCENE {n} — SUMMARY")
        summary = orchestrator.summarize_scene(prose)
        bible.scene_summaries[n] = summary
        print(summary)

        _banner(f"SCENE {n} — BIBLE UPDATE (post-scene)")
        bible = orchestrator.update_bible(bible, n, prose)
        bible.current_scene = n + 1
        storage.save(story_id, bible)

        # Contract evaluation report
        evaluation = bible.scene_evaluations.get(n)
        if evaluation:
            print("CONTRACT RESULTS:")
            for name, results in evaluation.contract_results.items():
                emo_landed = evaluation.emotional_state_landed.get(name, False)
                emo_mark = "✓" if emo_landed else "✗"
                if results:
                    landed = sum(1 for r in results if r.landed)
                    print(f"  [{name}] knowledge {landed}/{len(results)}, emotional_state {emo_mark}")
                    for r in results:
                        mark = "✓" if r.landed else "✗"
                        print(f"    {mark} {r.planned_fact}")
                        if r.landed and r.evidence:
                            print(f"        evidence: {r.evidence}")
                else:
                    print(f"  [{name}] no contracted facts; emotional_state {emo_mark}")
            organic_total = sum(len(v) for v in evaluation.organic_knowledge.values())
            if organic_total:
                print(f"ORGANIC FACTS ({organic_total}):")
                for name, facts in evaluation.organic_knowledge.items():
                    for f in facts:
                        print(f"  + [{name}] {f}")

        print("\nworld_rules:")
        for r in bible.world_rules:
            print(f"  - {r}")
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
