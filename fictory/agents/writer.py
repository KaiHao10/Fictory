from collections import defaultdict
from pathlib import Path

from .. import llm
from ..models.handoff import HandoffDocument

PROMPT = (Path(__file__).parent.parent / "prompts" / "writer" / "write_scene.txt").read_text(
    encoding="utf-8"
)


def write_scene(handoff: HandoffDocument) -> str:
    user = _format_handoff(handoff)
    for attempt in range(3):
        prose = llm.chat(system=PROMPT, user=user, max_tokens=6144, temperature=0.85)
        if prose and len(prose) > 50:
            return prose
        print(f"[writer] empty/short prose returned (len={len(prose)}); retrying (attempt {attempt + 1}/3)…")
    raise RuntimeError(f"Writer returned empty prose for scene {handoff.scene_outline.scene_number} after 3 attempts")


def _format_handoff(h: HandoffDocument) -> str:
    lines: list[str] = []

    if h.rewrite_notes:
        lines.append("=== REVISION NOTES (prior draft failed evaluation — fix these) ===")
        lines.append(h.rewrite_notes)
        lines.append("")

    lines.append(f"=== SCENE {h.scene_outline.scene_number} ===")
    lines.append(f"Arc position: {h.scene_outline.arc_position}")
    lines.append(f"Summary: {h.scene_outline.summary}")
    lines.append("Beats (all must occur):")
    for b in h.scene_outline.beats:
        lines.append(f"  - {b}")

    if h.story_so_far:
        lines.append("")
        lines.append("=== STORY SO FAR ===")
        lines.append(h.story_so_far)

    lines.append("")
    lines.append("=== WORLD RULES ===")
    for r in h.world_rules:
        lines.append(f"  - {r}")

    lines.append("")
    lines.append("=== CHARACTERS IN THIS SCENE ===")
    for name, ch in h.characters.items():
        lines.append(f"\n--- {name} ---")
        lines.append(f"Profile: {ch.profile}")
        lines.append(f"Knows now: {ch.state.knowledge}")
        lines.append(f"Feels now: {ch.state.emotional_state}")
        if ch.planned_delta:
            lines.append("CONTRACT for this scene (must land by end of scene):")
            deliveries = ch.planned_delta.knowledge_to_gain
            if deliveries:
                grouped: dict[str, list[str]] = defaultdict(list)
                for d in deliveries:
                    if d.source == "delivered_by_character":
                        key = f"From {d.delivered_by} ({d.via or 'dialogue'})"
                    elif d.source == "world_observation":
                        key = f"By observing in the prose ({d.via or 'direct observation'})"
                    elif d.source == "self_realization":
                        key = f"By realization ({d.via or 'connecting prior knowledge'})"
                    elif d.source == "external_document":
                        key = f"By reading ({d.via or 'document'})"
                    else:
                        key = "Other"
                    grouped[key].append(d.fact)
                lines.append("  Must learn:")
                for key in grouped:
                    lines.append(f"    {key}:")
                    for fact in grouped[key]:
                        lines.append(f"      → {fact}")
            else:
                lines.append("  Must learn: (no new facts)")
            if ch.planned_delta.target_emotional_state:
                lines.append(f"  Must feel by EOS: {ch.planned_delta.target_emotional_state}")
        if ch.voice_examples:
            lines.append("Voice examples (match this voice):")
            for ex in ch.voice_examples:
                lines.append(f'  • "{ex}"')

    lines.append("")
    lines.append("Write the scene now. Honor every contract. Output only the prose.")
    return "\n".join(lines)
