from pathlib import Path

from .. import llm
from ..models.story_bible import StoryBible
from ..models.verdict import SceneVerdict

PROMPT = (Path(__file__).parent.parent / "prompts" / "evaluator" / "evaluate_scene.txt").read_text(
    encoding="utf-8"
)


def evaluate_scene(bible: StoryBible, scene_number: int, prose: str) -> SceneVerdict:
    scene = next(s for s in bible.scenes if s.scene_number == scene_number)
    present = {n: bible.characters[n] for n in scene.characters_present if n in bible.characters}

    # Pre-scene state + contract block, one chunk per character.
    char_blocks: list[str] = []
    for name, ch in present.items():
        delta = scene.planned_deltas.get(name)
        lines = [
            f"--- {name} ---",
            f"  profile: {ch.profile}",
            f"  knew going in: {ch.state.knowledge}",
            f"  felt going in: {ch.state.emotional_state}",
        ]
        if delta and delta.knowledge_to_gain:
            lines.append("  CONTRACT — must learn this scene:")
            for d in delta.knowledge_to_gain:
                attribution = f"source={d.source}"
                if d.delivered_by:
                    attribution += f", delivered_by={d.delivered_by}"
                if d.via:
                    attribution += f", via={d.via}"
                lines.append(f"    [{attribution}] {d.fact}")
            lines.append(f"  CONTRACT — must feel by EOS: {delta.target_emotional_state}")
        elif delta:
            lines.append("  CONTRACT — no new facts to learn this scene")
            lines.append(f"  CONTRACT — must feel by EOS: {delta.target_emotional_state}")
        else:
            lines.append("  CONTRACT — present but no planned change")
        char_blocks.append("\n".join(lines))

    state_block = "\n".join(char_blocks)
    synopsis_json = bible.synopsis.model_dump_json(indent=2)
    story_so_far = bible.story_so_far or "(this is scene 1 — nothing has happened yet)"

    user = (
        f"SYNOPSIS:\n{synopsis_json}\n\n"
        f"WORLD RULES:\n- " + "\n- ".join(bible.world_rules) + "\n\n"
        f"STORY SO FAR:\n{story_so_far}\n\n"
        f"SCENE OUTLINE (scene {scene_number}):\n"
        f"  arc_position: {scene.arc_position}\n"
        f"  summary: {scene.summary}\n"
        f"  beats: {scene.beats}\n"
        f"  characters_present: {scene.characters_present}\n\n"
        f"CHARACTERS (pre-scene state + contracts):\n{state_block}\n\n"
        f"PROSE:\n{prose}\n\n"
        f"Return the evaluator JSON now."
    )

    data = llm.chat_json(system=PROMPT, user=user, max_tokens=8192, temperature=0.2)
    # Ensure scene_number is set correctly regardless of what the model returned.
    data["scene_number"] = scene_number
    return SceneVerdict.model_validate(data)
