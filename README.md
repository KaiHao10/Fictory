# Fictory

A system that writes long fiction one scene at a time, using three LLMs that pass notes to each other.

---

## What problem does this solve?

If you ask a single LLM to "write me a 30,000-word novel," it will fall apart. Characters will change personality between chapters. A secret revealed in scene 7 will be forgotten by scene 12. The villain's name will subtly drift from Eleanor to Elinor. Voice will flatten. Plot threads will dangle.

The instinct is to blame the **context window** — the model can't fit the whole novel in memory, so it forgets. But that framing leads to the wrong fix. The harder problem is **coherence**: keeping facts, voices, emotional arcs, and plot threads consistent across many separate model calls. A bigger context window doesn't fix coherence on its own — you need an explicit memory of what has been written, updated as the story grows.

Fictory takes that explicit-memory approach.

---

## The architecture: three LLMs with different jobs

```
                ┌─────────────────────────────────────────────┐
                │              ORCHESTRATOR                    │
                │  - drafts the synopsis (theme, arc, ending)  │
                │  - drafts the full scene outline upfront     │
                │  - holds the "story bible" (the memory)      │
                │  - prepares a handoff doc for each scene     │
                │  - applies the evaluator's verdict to state  │
                └─────────────────────────────────────────────┘
                       │ (handoff)              ▲ (verdict)
                       ▼                        │
                ┌─────────────────────────────────────────────┐
                │                WRITER                        │
                │  - receives the handoff for ONE scene        │
                │  - writes the prose, nothing else            │
                │  - has no memory of any other scene          │
                └─────────────────────────────────────────────┘
                       │ (prose)                ▲ (rewrite notes)
                       ▼                        │
                ┌─────────────────────────────────────────────┐
                │              EVALUATOR                       │
                │  - checks: did every contracted fact land?   │
                │  - flags overcreative inventions             │
                │  - flags continuity breaks                   │
                │  - on fail, returns notes for a rewrite      │
                └─────────────────────────────────────────────┘
```

**The orchestrator** is the showrunner. It plans the whole story — first a synopsis that locks in theme, arc, ending image, and emotional payoff, then a scene-by-scene outline engineered to land that ending. It owns the bible and is the only agent that ever has the whole story in mind. It also maintains a rolling "story so far" recap that grows with the story.

**The writer** is the prose specialist. It writes one scene given a handoff, and forgets immediately when the scene is done. It doesn't brainstorm, doesn't invent plot, doesn't decide what a character now knows — it just writes the scene the orchestrator asked for, using the story-so-far for continuity.

**The evaluator** is the gate. After each scene is written, it checks the prose against the contract: did every planned fact land? Did the emotional state arrive? Did the writer get overcreative and invent something that contradicts the bible or pre-empts a later beat? If anything fails, the evaluator returns concrete notes and the scene is regenerated (up to 2 retries).

This separation matters: the orchestrator does all the planning, the writer does all the prose, the evaluator does all the judgment. The writer is forced to be disciplined — if a fact isn't in the handoff, it cannot reference it. The orchestrator never reads prose for plot truth; it trusts the verdict.

---

## The story bible: the system's memory

The bible is one big Pydantic object that gets serialized to `bible.json` after every scene. Everything the system knows lives here. From [`fictory/models/story_bible.py`](fictory/models/story_bible.py):

```python
class StoryBible(BaseModel):
    title: str
    genre: str
    premise: str
    synopsis: Synopsis                        # theme, arc, ending image, emotional payoff
    world_rules: list[str]                    # facts about the world
    scenes: list[SceneOutline]                # the full outline, drafted upfront
    characters: dict[str, Character]          # everyone in the story
    generated_prose: dict[int, str]           # scene_number -> finished prose
    scene_summaries: dict[int, str]           # 1-2 sentence summaries
    story_so_far: str                         # rolling ~300-500 word recap
    current_scene: int                        # for resume
```

The `synopsis` is the load-bearing piece that locks in how the story lands:

```python
class Synopsis(BaseModel):
    theme: str
    central_conflict: str
    arc: str            # 1-2 paragraphs covering setup → climax → resolution
    ending_image: str   # the final beat the story lands on
    emotional_payoff: str
```

The scene grid is engineered to serve the synopsis — the last scene must land the `ending_image` and deliver the `emotional_payoff`. This is checked at outline-gen time; later, every scene is judged against the synopsis by the evaluator too.

Each `Character` has static stuff (profile, voice examples) and dynamic stuff (current knowledge, current emotional state). Dynamic state evolves mechanically: only when the evaluator confirms a planned fact landed does that fact get added to the character's knowledge. There's no "organic fact" channel from prose into bible — brainstorming is 100% the orchestrator's job.

```python
class Character(BaseModel):
    name: str
    profile: str                              # personality, speech, quirks
    voice_examples: list[VoiceExample]        # real lines used as voice anchors
    state: CharacterState                     # latest state (knowledge + feelings)
```

---

## Voice examples: how characters keep sounding like themselves

Describing a character's voice abstractly ("dry, sardonic, uses rhetorical questions") is weaker than showing real lines they've actually said. So Fictory keeps a bank of example lines for each character:

- **Seed examples** — invented by the orchestrator at outline time (these have `scene_number: 0`)
- **Earned examples** — pulled from prose the writer has actually generated, picked by the evaluator while it's already reading the scene (these have `scene_number: N` for the scene they came from), tagged `is_best: true` if they're unusually representative

The bank is capped at "10 most-recent + 5 best" — too few weakens anchoring, too many bloats the prompt. When a writer is about to write scene 8, the handoff includes a curated set of voice samples for each character present in the scene, so the writer can pattern-match instead of trying to reverse-engineer voice from a description.

Crucially, the voice samples evolve. The character that emerges in scene 12 is the one anchored by the prose written through scenes 1-11 — not just the one the orchestrator imagined at outline time.

---

## The handoff: the writer's contract

The handoff is the only thing the writer sees. From [`fictory/models/handoff.py`](fictory/models/handoff.py):

```python
class HandoffDocument(BaseModel):
    scene_outline: SceneOutline               # what happens in this scene
    characters: dict[str, CharacterHandoff]   # everyone present, with state + voice
    world_rules: list[str]                    # established facts about the world
    story_so_far: str                         # rolling recap of all prior scenes
    rewrite_notes: str | None                 # set on retry; evaluator's fix list

class CharacterHandoff(BaseModel):
    name: str
    profile: str
    voice_examples: list[str]                 # picked: 10 recent + 5 best
    state: CharacterState                     # what they know + how they feel NOW
    planned_delta: PlannedDelta | None        # the contract for THIS scene
```

The `planned_delta` is the load-bearing piece. It tells the writer:

- **What the character must KNOW by the end of this scene** (with attribution: who tells them? do they observe it? do they read it?)
- **What the character must FEEL by the end of this scene**

Here's a real handoff fragment the writer would receive (from the noir-mystery test run):

```
--- Lila Vance ---
Profile: A young auditor in her late twenties, sharp and observant,...
Knows now: ['She is an auditor for the regional office', ...]
Feels now: Professional and focused, with underlying curiosity about the remote setting
CONTRACT for this scene (must land by end of scene):
  Must learn:
    From Hank Vorhees (interview dialogue):
      → Hank was coming to report to Silas and found Lila with the body.
    From Elias Emery (medical examination):
      → The cause of death is blunt force trauma, time of death 2-3 a.m.
    By observing in the prose (direct observation):
      → The safe in the office is locked and appears undisturbed.
  Must feel by EOS: Determined to investigate beyond the polite cover story.
Voice examples (match this voice):
  • "Just the books, Mr. Vorhees. I'll start with the books."
```

The writer is now obligated to produce prose where:
- Hank specifically tells Lila about finding her with the body (not someone else)
- Elias specifically delivers the cause of death
- Lila herself observes the locked safe
- Lila ends the scene determined, not relieved or scared

This is what makes coherence enforceable. Vague beats ("Lila investigates the scene") are unmeasurable. Attributed contracts are testable.

---

## The per-scene loop

The pipeline lives in [`fictory/pipeline.py`](fictory/pipeline.py). Each scene runs a draft → evaluate → (maybe retry) → apply cycle:

```python
for n in range(start_at, num_scenes + 1):
    rewrite_notes = None
    for attempt in range(MAX_RETRIES + 1):           # up to 3 attempts
        handoff = orchestrator.assemble_handoff(bible, n, rewrite_notes=rewrite_notes)
        prose = writer.write_scene(handoff)
        verdict = evaluator.evaluate_scene(bible, n, prose)
        if verdict.passed:
            break
        rewrite_notes = verdict.notes_for_rewrite

    bible.generated_prose[n] = prose
    bible.scene_summaries[n] = orchestrator.summarize_scene(prose)
    bible.story_so_far = orchestrator.update_story_so_far(bible, n, prose)
    bible = orchestrator.apply_scene(bible, n, verdict)  # mechanically applies landed facts
    bible.current_scene = n + 1
    storage.save(story_id, bible)
```

Three LLM calls per accepted scene (writer + evaluator + story-so-far update + summarize — four if you count the summary). If the evaluator rejects a draft, it costs one more writer + evaluator round-trip per retry. After 2 retries the last draft is accepted with a warning rather than looping forever.

Outline-time costs (one-time per story): one synopsis call, one outline call, one normalize call.

---

## The evaluator: the quality gate

The evaluator reads each scene's prose against its contract and the larger story. It returns a structured verdict — never persisted to the bible, just used to drive retries and state application:

```python
class CharacterVerdict(BaseModel):
    facts_landed: dict[str, bool]            # planned fact → did it land?
    facts_evidence: dict[str, str]           # quoted prose evidence for landed facts
    emotional_state_landed: bool
    voice_examples: list[VoiceExampleDraft]  # 0-2 picks per character

class SceneVerdict(BaseModel):
    scene_number: int
    passed: bool
    per_character: dict[str, CharacterVerdict]
    overcreativity_flags: list[str]          # invented facts that hurt the big picture
    coherence_flags: list[str]               # continuity breaks
    notes_for_rewrite: str                   # concrete fix list for the writer
```

The evaluator checks four things:

1. **Contract landing.** For every planned fact, did the prose actually deliver it via the planned mechanism (specific speaker, observation, document)? Implied / off-screen delivery doesn't count.
2. **Emotional state.** Did each character end the scene plausibly feeling what the contract required?
3. **Overcreativity.** Did the writer invent things that contradict the bible, pre-empt a later beat, or introduce plot-material new entities the orchestrator didn't plan?
4. **Coherence.** Continuity breaks against world rules, the story-so-far, or the synopsis (dead characters speaking, wrong tone, etc.).

`passed` is true only if every contract item landed and there are zero flags. On failure, `notes_for_rewrite` is the next handoff's `rewrite_notes` block.

On a passed scene, `orchestrator.apply_scene` walks the planned deltas: every fact the verdict marked landed gets appended to the receiver's `state.knowledge`; if the emotional state landed, the character's `state.emotional_state` is set to the verbatim `target_emotional_state` from the outline. State evolution is strictly mechanical — the bible never trusts prose for plot truth.

---

## The contract system: source-attributed facts

Every contracted fact has a `source` declaring how the character will learn it. From [`fictory/models/story_bible.py`](fictory/models/story_bible.py):

```python
KnowledgeSource = Literal[
    "delivered_by_character",  # someone tells/shows them — deliverer must already know
    "world_observation",       # they observe it in the prose (no precondition)
    "self_realization",        # they deduce it from facts they already have
    "external_document",       # they read it from a letter/ledger/artifact
]

class PlannedKnowledgeDelivery(BaseModel):
    fact: str
    source: KnowledgeSource
    delivered_by: str | None = None  # required iff source == delivered_by_character
    via: str | None = None           # short description: "interview", "letter", etc.
```

The `delivered_by_character` source has a precondition: **the deliverer must already know the fact**, otherwise they're a mouthpiece for information they have no business having.

Two validation passes catch this at outline time, before any prose is written.

---

## Validation pass 1: normalize_fact_strings

The orchestrator paraphrases. It writes the same fact two different ways in two different places:

- Foreman's seed: *"Silas and Duke had a loud argument the night before Silas's death."*
- Scene 2 contract: *"Silas and Duke Marchetti had a loud argument the night before Silas's death."*

These are the same fact, but to a string-equality check they're distinct. So the next pass would falsely flag the foreman as not entitled to deliver.

The normalize pass asks the LLM to find paraphrase clusters across all fact strings (world_rules + every character's knowledge + every scene's planned_deltas) and pick one canonical wording per cluster. Substitutions are applied across the bible. The log is printed to stderr and discarded — it's an outline-time debug aid, not part of the persisted bible.

## Validation pass 2: validate_and_backfill

Now that fact strings are canonical, walk the outline scene by scene, projecting each character's knowledge forward through the planned deltas. For every `delivered_by_character` fact, check that the deliverer has it in their projected state at the start of that scene. If they don't, **seed-backfill**: add it to their starting knowledge so they're entitled to deliver it.

A real example from a run:

```
BACKFILL scene 2: 'Silas kept a hidden ledger of off-the-books expenses' added to
  Hank Vorhees.seed (needed to deliver to Lila Vance)
BACKFILL scene 4: 'Marjorie was embezzling funds by creating fake vendor invoices'
  added to Marjorie Phelps.seed (needed to deliver to Lila Vance)
```

These are correct repairs — the foreman *should* have known about the ledger from the start, the orchestrator just forgot to seed it. By the time the writer sees the handoff, the deliverer's state is already consistent. Like the normalize log, the backfill log is printed once and discarded.

---

## Robustness: what happens when things go wrong

LLM APIs fail in fun ways. Fictory is built to handle the common ones.

**Connection drops mid-stream.** Deepseek occasionally cuts the response mid-message. The base LLM client retries with exponential backoff (see [`fictory/llm.py`](fictory/llm.py)).

**Empty prose returned.** The writer's wrapper retries up to 3 times if the LLM returns nothing.

**Bad scene from the writer.** Caught by the evaluator. Up to 2 rewrite attempts with concrete fix notes, then the last attempt is accepted with a warning so the pipeline doesn't loop forever on an impossible scene.

**Mid-run crash.** If the script dies between scenes, the bible.json on disk still has everything done so far. Re-running with the same `--story-id` resumes from the saved `current_scene`. You don't pay for synopsis/outline twice. You don't redo finished scenes.

---

## How to use it

### macOS / Linux

**Install once:**
```bash
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env
# edit .env and put your DEEPSEEK_API_KEY in
```

**Run a story:**
```bash
.venv/bin/python main.py \
  --premise "A retired locksmith is hired to crack a safe she sealed forty years ago." \
  --genre "literary thriller" \
  --scenes 8 \
  --story-id "the-locksmith"
```

### Windows

**Install once:**
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e .
copy .env.example .env
# edit .env and put your DEEPSEEK_API_KEY in
```

**Run a story:**
```powershell
.\.venv\Scripts\python main.py `
  --premise "A retired locksmith is hired to crack a safe she sealed forty years ago." `
  --genre "literary thriller" `
  --scenes 8 `
  --story-id "the-locksmith"
```

Required arguments:
- `--premise` — one or two sentences. Be specific.
- `--genre` — guides tone (e.g. "noir mystery", "literary sci-fi", "cozy fantasy").
- `--scenes` — how many scenes. 4–8 is a good range to start.
- `--story-id` (optional) — the output directory name. If omitted, derived from the premise.

**Output structure:**
```
stories/the-locksmith/
├── bible.json           # the full story state (everything)
├── scene_001.txt        # raw prose for scene 1
├── scene_002.txt
└── ...
```

`bible.json` is the canonical artifact. The `scene_NNN.txt` files are convenience copies of the prose.

**Resume an interrupted run:**
Just re-run the same command. If `bible.json` exists for that story_id, it picks up where it left off.

---

## What lives where

```
fictory/
├── llm.py                      # Deepseek client wrapper, retries, truncation warning
├── models/
│   ├── story_bible.py          # all the data shapes (Synopsis, Character, Scene, PlannedDelta, etc.)
│   ├── handoff.py              # the HandoffDocument the writer receives
│   └── verdict.py              # the SceneVerdict the evaluator returns
├── agents/
│   ├── orchestrator.py         # synopsis + outline gen, normalize, validate+backfill,
│   │                           # assemble_handoff, update_story_so_far, summarize_scene, apply_scene
│   ├── writer.py               # write_scene + handoff-to-prompt formatter
│   └── evaluator.py            # evaluate_scene — contract + overcreativity + coherence
├── prompts/
│   ├── orchestrator/
│   │   ├── generate_synopsis.txt    # how to draft theme/arc/ending
│   │   ├── generate_outline.txt     # how to draft the scene grid + contracts
│   │   ├── normalize_outline.txt    # how to cluster paraphrased facts
│   │   ├── update_story_so_far.txt  # how to roll the recap forward
│   │   └── summarize_scene.txt      # 1-2 sentence summary of a scene
│   ├── evaluator/
│   │   └── evaluate_scene.txt       # how to judge a scene against its contract
│   └── writer/
│       └── write_scene.txt          # how to write a scene given the handoff
├── pipeline.py                 # the run loop (draft → evaluate → retry → apply, resume support)
└── storage.py                  # load/save bible.json + scene .txt files
```

Every prompt is a plain text file. If you want to change behavior (be terser, be more lyrical, weight beats differently), edit the prompts.

---

## TL;DR

Long fiction breaks not because the model can't see the whole story, but because nothing is enforcing consistency between calls. Fictory enforces consistency by:

1. **Separating planning (orchestrator), prose (writer), and judgment (evaluator)** so no single agent has to do all three
2. **Drafting a synopsis first** — theme, arc, ending image, emotional payoff — so the outline has a target to land
3. **Drafting an explicit per-scene contract** with attributed knowledge deliveries, validated for consistency before any prose is written
4. **Normalizing fact wording** so the same fact written two ways doesn't become two facts
5. **Seed-backfilling** when a planned speaker doesn't have the info they're scheduled to deliver
6. **Gating every scene through the evaluator** — facts must land, no overcreative inventions, no continuity breaks — with up to 2 rewrites on failure
7. **Updating state mechanically** from the verdict, never trusting prose to invent plot truth
8. **Carrying a rolling story-so-far** so the writer remembers the whole arc, not just the previous scene
9. **Persisting after every scene** so transient API failures don't lose work

Three agents, one synopsis, one outline, and a draft/evaluate/apply loop per scene. Output is a structured `bible.json` with everything the system knows about itself.
