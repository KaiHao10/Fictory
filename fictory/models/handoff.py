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
    prior_scene_summary: str | None = None
