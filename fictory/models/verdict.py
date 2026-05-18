from pydantic import BaseModel, Field, field_validator


class VoiceExampleDraft(BaseModel):
    line: str
    is_best: bool = False

    @field_validator("is_best", mode="before")
    @classmethod
    def _coerce_none(cls, v):
        return False if v is None else v


class CharacterVerdict(BaseModel):
    """Per-character verdict for one scene."""

    # Map planned-fact-string -> did it land in the prose? (with optional quoted evidence)
    facts_landed: dict[str, bool] = Field(default_factory=dict)
    facts_evidence: dict[str, str] = Field(default_factory=dict)
    emotional_state_landed: bool = False
    # Lines from the prose that exemplify this character's voice (0-2 picks).
    voice_examples: list[VoiceExampleDraft] = Field(default_factory=list)

    @field_validator("emotional_state_landed", mode="before")
    @classmethod
    def _coerce_emo_none(cls, v):
        return False if v is None else v

    @field_validator("facts_landed", mode="before")
    @classmethod
    def _coerce_fact_values(cls, v):
        if isinstance(v, dict):
            return {k: (False if val is None else val) for k, val in v.items()}
        return v

    @field_validator("facts_evidence", mode="before")
    @classmethod
    def _coerce_evidence_values(cls, v):
        if isinstance(v, dict):
            return {k: ("" if val is None else val) for k, val in v.items()}
        return v


class SceneVerdict(BaseModel):
    """Transient evaluator output for one scene. Not persisted to the bible."""

    scene_number: int
    passed: bool
    per_character: dict[str, CharacterVerdict] = Field(default_factory=dict)
    # Invented facts/details that contradict the bible or hurt the planned arc.
    overcreativity_flags: list[str] = Field(default_factory=list)
    # Inconsistencies with synopsis, world rules, or prior scenes.
    coherence_flags: list[str] = Field(default_factory=list)
    # Concrete guidance handed back to the writer on retry.
    notes_for_rewrite: str = ""

    @field_validator("passed", mode="before")
    @classmethod
    def _coerce_passed_none(cls, v):
        return False if v is None else v

    @field_validator("overcreativity_flags", "coherence_flags", mode="before")
    @classmethod
    def _coerce_list_none(cls, v):
        return [] if v is None else v

    @field_validator("notes_for_rewrite", mode="before")
    @classmethod
    def _coerce_notes_none(cls, v):
        return "" if v is None else v

    @field_validator("per_character", mode="before")
    @classmethod
    def _coerce_per_character_none(cls, v):
        return {} if v is None else v
