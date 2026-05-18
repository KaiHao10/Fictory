from typing import Literal

from pydantic import BaseModel, Field, field_validator

KnowledgeSource = Literal[
    "delivered_by_character",  # another character speaks/shows it; deliverer must already know
    "world_observation",       # character observes it directly in the prose (no precondition)
    "self_realization",        # character deduces it from facts already in their state
    "external_document",       # character reads it from a letter / ledger / artifact
]

# Convention: scene_number=0 means "seeded at outline time, before any prose
# was generated." It is the sentinel for the pre-story / initial state. The
# real prose scenes are numbered 1..N and live in StoryBible.scenes.


class PlannedKnowledgeDelivery(BaseModel):
    """One contracted fact a character will learn in a scene, with attribution
    for HOW they learn it. The attribution is what makes precondition checks
    possible — e.g. for `delivered_by_character`, the deliverer must already
    know the fact at the start of the scene.
    """

    fact: str
    source: KnowledgeSource
    delivered_by: str | None = None  # required iff source == "delivered_by_character"
    via: str | None = None           # short description: "interview", "letter", "deduction", etc.


class PlannedDelta(BaseModel):
    """The orchestrator's contract for a character in a single scene.

    Drafted at outline time. By the end of the scene, the character should
    KNOW everything in `knowledge_to_gain` and FEEL `target_emotional_state`.
    The writer is shown this contract; the evaluator verifies it landed.
    """

    knowledge_to_gain: list[PlannedKnowledgeDelivery] = Field(default_factory=list)
    target_emotional_state: str = ""

    @field_validator("knowledge_to_gain", mode="before")
    @classmethod
    def _accept_legacy_strings(cls, v):
        """Backward-compat: accept old `list[str]` shape from pre-attribution
        bibles. Plain strings get lifted to source='world_observation' (a
        no-precondition default) so legacy data still loads."""
        if isinstance(v, list):
            return [
                {"fact": x, "source": "world_observation"} if isinstance(x, str) else x
                for x in v
            ]
        return v


class CharacterState(BaseModel):
    """Latest dynamic state for a character."""

    knowledge: list[str] = Field(default_factory=list)
    emotional_state: str = ""
    last_updated_scene: int = 0  # 0 = initial seed, N = after scene N


class VoiceExample(BaseModel):
    line: str
    scene_number: int  # 0 = seeded at outline; 1..N = pulled from scene N's prose
    is_best: bool = False


class Character(BaseModel):
    name: str
    profile: str
    voice_examples: list[VoiceExample] = Field(default_factory=list)
    state: CharacterState = Field(default_factory=CharacterState)


class SceneOutline(BaseModel):
    scene_number: int
    summary: str
    beats: list[str]
    characters_present: list[str]
    arc_position: str
    # Per-character contract for this scene (name -> what they should learn / feel
    # by end-of-scene). Drafted at outline time; verified post-hoc by the evaluator.
    planned_deltas: dict[str, PlannedDelta] = Field(default_factory=dict)


class Synopsis(BaseModel):
    """Top-level story treatment. Drafted before the scene grid so the outline
    has a target to land. The ending image and emotional payoff are the
    load-bearing pieces — they tell the orchestrator how the story must end."""

    theme: str
    central_conflict: str
    arc: str            # 1-2 paragraphs covering setup → escalation → midpoint → climax → resolution
    ending_image: str   # the final beat the story lands on
    emotional_payoff: str


class StoryBible(BaseModel):
    title: str
    genre: str
    premise: str
    synopsis: Synopsis
    world_rules: list[str] = Field(default_factory=list)
    scenes: list[SceneOutline]
    characters: dict[str, Character]
    generated_prose: dict[int, str] = Field(default_factory=dict)
    scene_summaries: dict[int, str] = Field(default_factory=dict)
    # Rolling "story so far" narrative, ~300-500 words, updated by the
    # orchestrator after each scene. Empty before scene 1.
    story_so_far: str = ""
    current_scene: int = 1
