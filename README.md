# Fictory

A system that writes long fiction one scene at a time, using two LLMs that pass notes to each other.

---

## What problem does this solve?

If you ask a single LLM to "write me a 30,000-word novel," it will fall apart. Characters will change personality between chapters. A secret revealed in scene 7 will be forgotten by scene 12. The villain's name will subtly drift from Eleanor to Elinor. Voice will flatten. Plot threads will dangle.

The instinct is to blame the **context window** — the model can't fit the whole novel in memory, so it forgets. But that framing leads to the wrong fix. The harder problem is **coherence**: keeping facts, voices, emotional arcs, and plot threads consistent across many separate model calls. A bigger context window doesn't fix coherence on its own — you need an explicit memory of what has been written, updated as the story grows.

Fictory takes that explicit-memory approach.

---

## The architecture: two LLMs with different jobs

```
                ┌─────────────────────────────────────────────┐
                │              ORCHESTRATOR                    │
                │  - drafts the full outline up front          │
                │  - holds the "story bible" (the memory)      │
                │  - prepares a handoff doc for each scene     │
                │  - reads each finished scene and updates the │
                │    bible with what's now true                │
                └─────────────────────────────────────────────┘
                       │ (handoff)              ▲ (prose)
                       ▼                        │
                ┌─────────────────────────────────────────────┐
                │                WRITER                        │
                │  - receives the handoff for ONE scene        │
                │  - writes the prose                          │
                │  - has no memory of any other scene          │
                └─────────────────────────────────────────────┘
```

Two roles, two LLM calls per scene (sometimes three).

**The orchestrator** is the showrunner. It plans the whole story before any prose is written, keeps a "story bible" with everything that matters, and decides what each scene needs to deliver. After each scene is written, it reads the prose and updates the bible.

**The writer** is the prose specialist. It writes one scene, gets handed everything it needs in a "handoff document," and forgets immediately when the scene is done. It doesn't even know what the other scenes are.

This separation matters: the orchestrator is the only thing that ever has the whole story in mind. The writer is forced to be disciplined — if a fact isn't in the handoff, it cannot reference it.

---

## The story bible: the system's memory

The bible is one big Pydantic object that gets serialized to `bible.json` after every scene. Everything the system knows lives here. From [`fictory/models/story_bible.py`](fictory/models/story_bible.py):

```python
class StoryBible(BaseModel):
    title: str
    genre: str
    premise: str
    world_rules: list[str]                    # facts about the world
    scenes: list[SceneOutline]                # the full outline, drafted upfront
    characters: dict[str, Character]          # everyone in the story
    generated_prose: dict[int, str]           # scene_number -> finished prose
    scene_summaries: dict[int, str]           # 1-2 sentence summaries
    scene_evaluations: dict[int, SceneEvaluation]  # eval data per scene
    backfill_log: list[str]                   # audit log of state repairs
    normalize_log: list[str]                  # audit log of fact normalizations
    current_scene: int                        # for resume
```

Each `Character` has both static stuff (profile, voice examples) and dynamic stuff (current knowledge, current emotional state) — and **a per-scene history** of how the dynamic stuff has changed:

```python
class Character(BaseModel):
    name: str
    profile: str                              # personality, speech, quirks
    voice_examples: list[VoiceExample]        # real lines used as voice anchors
    state: CharacterState                     # latest state (knowledge + feelings)
    state_history: dict[int, CharacterStateSnapshot]  # state at end of each scene
```

The `state_history` lets you watch a character's mind change one scene at a time. Scene 0 is the seeded "before the story starts" state. Scene N (where N ≥ 1) is the state at the *end* of scene N.

---

## Voice examples: how characters keep sounding like themselves

Describing a character's voice abstractly ("dry, sardonic, uses rhetorical questions") is weaker than showing real lines they've actually said. So Fictory keeps a bank of example lines for each character:

- **Seed examples** — invented by the orchestrator at outline time (these have `scene_number: 0`)
- **Earned examples** — pulled from prose the writer has actually generated (these have `scene_number: N` for the scene they came from), tagged `is_best: true` if they're unusually representative

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
    prior_scene_summary: str | None           # 1-2 sentence recap

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

## The four-pass loop per scene

The whole pipeline is just a loop in [`fictory/pipeline.py`](fictory/pipeline.py):

```python
for n in range(start_at, num_scenes + 1):
    # Pass 1: the orchestrator assembles the handoff (deterministic, no LLM call)
    handoff = orchestrator.assemble_handoff(bible, n)

    # Pass 2: the writer generates the prose (LLM call)
    prose = writer.write_scene(handoff)
    bible.generated_prose[n] = prose

    # Pass 3: the orchestrator summarizes the prose for the next scene's recap (LLM call)
    bible.scene_summaries[n] = orchestrator.summarize_scene(prose)

    # Pass 4: the orchestrator reads the prose and updates the bible (LLM call)
    bible = orchestrator.update_bible(bible, n, prose)
    bible.current_scene = n + 1
    storage.save(story_id, bible)
```

Three LLM calls per scene. The handoff assembly is pure Python — it just reads from the bible, picks voice examples, and packages them.

---

## Update_bible: the brain of the operation

`update_bible` is the most consequential prompt in the system. It does two passes over the just-written prose:

**Pass 1 — Contract check.** For every fact the contract said the character should learn, did the prose actually deliver it via the planned mechanism? If the contract said "Hank tells Lila about the body" and the prose has Eleanor narrating it instead, the verdict is `landed: false` — even if the fact is technically present.

**Pass 2 — Organic extraction.** Beyond the contract, what *unplanned* things did the prose introduce? A metaphor, an offhand promise, a small reveal. These get captured into character knowledge so they propagate to future scenes.

The orchestrator returns a structured verdict per character. Here's a real one (also from the noir-mystery test):

```
CONTRACT RESULTS:
  [Lila Vance] knowledge 5/5, emotional_state ✓
    ✓ The dead body is Silas Croft, the chief accountant.
    ✓ The safe in the office is locked and appears undisturbed.
    ✓ There are no signs of forced entry on the door or windows.
    ✓ Hank was coming to report to Silas and found Lila with the body.
    ✓ The cause of death is blunt force trauma, time of death 2-3 a.m.
  [Hank Vorhees] knowledge 2/3, emotional_state ✓
    ✓ Lila Vance is the auditor sent from headquarters.
    ✓ Silas Croft is dead.
    ✗ No signs of forced entry were found.
ORGANIC FACTS (13):
  + [Lila Vance] Eleanor hid instructions inside the statue's base, sealed with wax.
  + [Hank Vorhees] He notices Lila has a city accent and corporate manner.
  ...
```

That single `✗` means the writer skipped a contracted fact. Without this verification, the bible would silently advance as if everything had landed — and a beat scheduled for scene 5 that depended on Hank knowing X would land in a void.

There's also a hard rule baked into the prompt to prevent a subtle bug: **a character only learns what they could plausibly perceive in the scene as written.** They learn what they hear, see, or read. They do NOT learn the contents of another character's internal monologue. They do NOT learn anything from another character's `knowledge` field unless it was actually said or shown to them. This stops the orchestrator from leaking omniscient narrator-state into individual characters' minds.

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

Without this check, the orchestrator can plan a scene where the foreman tells the auditor "Silas kept a hidden ledger" — but the foreman's seeded knowledge says nothing about a ledger. The fact lands at the wrong layer (the auditor's state grows, but the foreman is now a magic oracle).

Two validation passes catch this at outline time, before any prose is written.

---

## Validation pass 1: normalize_fact_strings

The orchestrator paraphrases. It writes the same fact two different ways in two different places:

- Foreman's seed: *"Silas and Duke had a loud argument the night before Silas's death."*
- Scene 2 contract: *"Silas and Duke Marchetti had a loud argument the night before Silas's death."*

These are the same fact, but to a string-equality check they're distinct. So the next pass would falsely flag the foreman as not entitled to deliver.

The normalize pass asks the LLM to find paraphrase clusters across all fact strings (world_rules + every character's knowledge + every scene's planned_deltas) and pick one canonical wording per cluster. Substitutions are applied across the bible. From [`fictory/agents/orchestrator.py`](fictory/agents/orchestrator.py):

```python
def normalize_fact_strings(bible: StoryBible) -> list[str]:
    """Cluster paraphrased fact strings across the outline and replace each
    paraphrase with a chosen canonical wording. Mutates the bible in place."""
    facts = _collect_all_facts(bible)
    if len(facts) < 2:
        return []
    # ... LLM call returns clusters: [{canonical, members}, ...]
    # ... apply substitutions across world_rules, character.state.knowledge,
    #     and planned_deltas[*].knowledge_to_gain[*].fact
```

This runs once per story, between outline generation and the precondition check.

## Validation pass 2: validate_and_backfill

Now that fact strings are canonical, walk the outline scene by scene, projecting each character's knowledge forward through the planned deltas. For every `delivered_by_character` fact, check that the deliverer has it in their projected state at the start of that scene. From [`fictory/agents/orchestrator.py`](fictory/agents/orchestrator.py):

```python
def validate_and_backfill(bible: StoryBible) -> list[str]:
    log: list[str] = []
    world = set(bible.world_rules)
    projected: dict[str, set[str]] = {
        name: set(ch.state.knowledge) for name, ch in bible.characters.items()
    }

    for scene in sorted(bible.scenes, key=lambda s: s.scene_number):
        for receiver_name, delta in scene.planned_deltas.items():
            for delivery in delta.knowledge_to_gain:
                if delivery.source != "delivered_by_character":
                    continue
                deliverer = delivery.delivered_by
                fact = delivery.fact
                if fact in world or fact in projected[deliverer]:
                    continue  # precondition met
                # Seed-backfill: the deliverer must know this from the start.
                bible.characters[deliverer].state.knowledge.append(fact)
                projected[deliverer].add(fact)
                log.append(
                    f"BACKFILL scene {scene.scene_number}: '{fact}' added to "
                    f"{deliverer}.seed (needed to deliver to {receiver_name})"
                )
        # Project this scene's gains forward to the receivers.
        for receiver_name, delta in scene.planned_deltas.items():
            for delivery in delta.knowledge_to_gain:
                projected[receiver_name].add(delivery.fact)
    return log
```

If the deliverer doesn't have the fact, we **seed-backfill**: add it to their starting knowledge so they're entitled to deliver it. Every backfill is logged into `bible.backfill_log` for audit. A real example from a run:

```
BACKFILL scene 2: 'Silas kept a hidden ledger of off-the-books expenses' added to
  Hank Vorhees.seed (needed to deliver to Lila Vance)
BACKFILL scene 4: 'Marjorie was embezzling funds by creating fake vendor invoices'
  added to Marjorie Phelps.seed (needed to deliver to Lila Vance)
```

These are correct repairs — the foreman *should* have known about the ledger from the start, the orchestrator just forgot to seed it. By the time the writer sees the handoff, the deliverer's state is already consistent.

---

## Evaluation: what the system tells you about itself

After a run, `bible.json` contains rich evaluation data:

- **`backfill_log`** — every seed-state repair. Watch for big jumps: lots of backfills suggests the orchestrator is sloppy on attribution.
- **`normalize_log`** — paraphrase substitutions made.
- **`scene_evaluations[N]`** — per scene, for every character: which contracted facts landed (with quoted prose evidence), which missed, what organic facts surfaced, whether the emotional state landed.
- **`characters[X].state_history`** — per scene, every character's full state. Diff `state_history[N]` and `state_history[N-1]` to see exactly what changed during scene N.

Together these let you compute things like beat-delivery rate per scene, how a character's knowledge accumulates over time, which scenes generated the most organic content, and which characters drifted between contracted and actual emotional states.

---

## Robustness: what happens when things go wrong

LLM APIs fail in fun ways. Fictory is built to handle the common ones.

**Connection drops mid-stream.** Deepseek occasionally cuts the response mid-message. The base LLM client retries with exponential backoff. From [`fictory/llm.py`](fictory/llm.py):

```python
def chat(system: str, user: str, max_tokens: int = 4096, temperature: float = 0.7,
         max_retries: int = 8) -> str:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = _get_client().messages.create(...)
            if getattr(resp, "stop_reason", None) == "max_tokens":
                print(f"[llm] WARNING: response truncated at max_tokens={max_tokens}")
            return "".join(part.text for part in resp.content if part.type == "text").strip()
        except (APIConnectionError, APITimeoutError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"[llm] connection error ({type(e).__name__}); retrying in {wait}s…")
            time.sleep(wait)
        except APIStatusError as e:
            if e.status_code in (429, 500, 502, 503, 504):
                last_err = e
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {max_retries} retries") from last_err
```

**Empty prose returned.** Sometimes the writer LLM returns nothing — a content filter, a generation hiccup, who knows. If this slips through silently, scene 3 ends up empty and scenes 4-8 are written with no idea anything went wrong. The writer guards against this:

```python
def write_scene(handoff: HandoffDocument) -> str:
    user = _format_handoff(handoff)
    for attempt in range(3):
        prose = llm.chat(system=PROMPT, user=user, max_tokens=6144, temperature=0.85)
        if prose and len(prose) > 50:
            return prose
        print(f"[writer] empty/short prose returned (len={len(prose)}); retrying...")
    raise RuntimeError(f"Writer returned empty prose for scene {handoff.scene_outline.scene_number}")
```

**Truncated response.** If a generation hits `max_tokens`, the JSON will be incomplete and parsing will fail. The client now logs `[llm] WARNING: response truncated at max_tokens=N` so you can see when to bump the cap.

**Mid-run crash.** If the script dies between scenes, the bible.json on disk still has everything done so far. Re-running with the same `--story-id` resumes from the saved `current_scene`:

```python
def run(premise, genre, num_scenes, story_id):
    existing = (storage.STORIES_DIR / story_id / "bible.json")
    if existing.exists():
        bible = storage.load(story_id)
        start_at = bible.current_scene
        _banner(f"RESUMING '{story_id}' from scene {start_at} (of {len(bible.scenes)})")
    else:
        bible = orchestrator.generate_outline(premise, genre, num_scenes)
        storage.save(story_id, bible)
        start_at = 1
    # ... loop from start_at
```

You don't pay for the outline twice. You don't redo finished scenes.

---

## How to use it

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
│   ├── story_bible.py          # all the data shapes (Character, Scene, PlannedDelta, etc.)
│   └── handoff.py              # the HandoffDocument the writer receives
├── agents/
│   ├── orchestrator.py         # outline gen, normalize, validate+backfill,
│   │                           # assemble_handoff, summarize_scene, update_bible
│   └── writer.py               # write_scene + handoff-to-prompt formatter
├── prompts/
│   ├── orchestrator/
│   │   ├── generate_outline.txt    # how to draft the full outline + contracts
│   │   ├── normalize_outline.txt   # how to cluster paraphrased facts
│   │   ├── update_bible.txt        # how to verify contracts + extract organic facts
│   │   └── summarize_scene.txt     # 1-2 sentence summary of a scene
│   └── writer/
│       └── write_scene.txt         # how to write a scene given the handoff
├── pipeline.py                 # the run loop (4-pass per scene, resume support)
└── storage.py                  # load/save bible.json + scene .txt files
```

Every prompt is a plain text file. If you want to change behavior (be terser, be more lyrical, weight beats differently), edit the prompts.

---

## TL;DR

Long fiction breaks not because the model can't see the whole story, but because nothing is enforcing consistency between calls. Fictory enforces consistency by:

1. **Separating planning (orchestrator) from prose generation (writer)**, so the planner is never asked to also be a stylist
2. **Drafting an explicit per-scene contract** with attributed knowledge deliveries, validated for consistency before any prose is written
3. **Normalizing fact wording** so the same fact written two ways doesn't become two facts
4. **Seed-backfilling** when a planned speaker doesn't have the info they're scheduled to deliver
5. **Verifying after every scene** that the contract actually landed, with quoted prose evidence
6. **Snapshotting state per scene** so you can audit every character's mind one frame at a time
7. **Persisting after every scene** so transient API failures don't lose work

Three LLM calls per scene, plus one outline call up front, plus one normalize call. Output is a structured `bible.json` with everything ever generated and everything the system knows about itself.
