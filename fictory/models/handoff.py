from pydantic import BaseModel

from .story_bible import CharacterState, PlannedDelta, SceneOutline


class CharacterHandoff(BaseModel):
    name: str
    profile: str
    voice_examples: list[str]
    state: CharacterState
    # The contract for this scene: by EOS, this character must know these
    # facts and feel this way. None if the character has no planned change.
    planned_delta: PlannedDelta | None = None


class HandoffDocument(BaseModel):
    scene_outline: SceneOutline
    characters: dict[str, CharacterHandoff]
    world_rules: list[str]
    # Rolling synthesis of everything that has happened before this scene.
    # Empty string for scene 1.
    story_so_far: str = ""
    # Set on retry attempts: the evaluator's notes on why the previous draft
    # failed and what the writer should fix. None on the first attempt.
    rewrite_notes: str | None = None
