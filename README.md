# Fictory

A system that writes long fiction one scene at a time, using three LLMs that pass notes to each other.

## The problem

Ask one LLM to write a novel and it falls apart: characters drift, revealed secrets get forgotten, names mutate. The real issue isn't context window size — it's **coherence** across many separate calls. Fictory fixes that with an explicit, growing memory of what has been written.

## Architecture

Three agents with strictly separated jobs:

- **Orchestrator** — plans the synopsis (theme, arc, ending, payoff) and the full scene outline upfront, owns the story bible, assembles a per-scene handoff, applies the evaluator's verdict to state.
- **Writer** — receives one handoff, writes prose, forgets. No plot invention, no cross-scene memory.
- **Evaluator** — checks each scene against its contract: did every planned fact land? Any overcreative inventions? Any continuity breaks? Returns rewrite notes on failure.

The writer can't reference facts not in the handoff. The orchestrator never trusts prose for plot truth — only verdict-confirmed facts update the bible.

## Key mechanisms

**Story bible** (`bible.json`) — one Pydantic object holding synopsis, characters (with evolving knowledge + emotional state), scene outline, generated prose, scene summaries, and a rolling ~300–500 word `story_so_far`. Persisted after every scene.

**Contracts with source-attributed facts** — every planned fact declares how a character learns it (`delivered_by_character`, `world_observation`, `self_realization`, `external_document`). The writer must deliver each fact via its planned mechanism; the evaluator checks that it did.

**Two outline-time validation passes:**
1. `normalize_fact_strings` — clusters paraphrases ("Silas and Duke argued" vs "Silas and Duke Marchetti argued") to one canonical wording.
2. `validate_and_backfill` — projects knowledge forward through the outline; if a planned speaker doesn't know what they're scheduled to deliver, seed-backfill it into their starting knowledge.

**Voice examples** — a bank of real lines per character (10 most-recent + 5 best), seeded by the orchestrator and grown by the evaluator from actual prose. The writer pattern-matches voice instead of reverse-engineering it from a description.

**Per-scene loop:**
```python
for n in range(start_at, num_scenes + 1):
    for attempt in range(MAX_RETRIES + 1):
        handoff = orchestrator.assemble_handoff(bible, n, rewrite_notes)
        prose = writer.write_scene(handoff)
        verdict = evaluator.evaluate_scene(bible, n, prose)
        if verdict.passed: break
        rewrite_notes = verdict.notes_for_rewrite
    bible = orchestrator.apply_scene(bible, n, verdict)
    storage.save(story_id, bible)
```

Up to 2 rewrites; after that, the last draft is accepted with a warning. Mid-run crashes resume from `current_scene` on re-run.

## Usage

**Install (macOS / Linux):**
```bash
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env   # put your DEEPSEEK_API_KEY in
```

**Run:**
```bash
.venv/bin/python main.py \
  --premise "A retired locksmith is hired to crack a safe she sealed forty years ago." \
  --genre "literary thriller" \
  --scenes 8 \
  --story-id "the-locksmith"
```

On Windows, use `.\.venv\Scripts\...` and backticks for line continuation.

Arguments: `--premise` (specific, 1–2 sentences), `--genre`, `--scenes` (4–8 is a good start), `--story-id` (optional, output dir name).

**Output:** `stories/<story-id>/bible.json` plus one `scene_NNN.txt` per scene. Re-running with the same `--story-id` resumes.

## Layout

```
fictory/
├── llm.py                 # Deepseek client, retries
├── models/                # story_bible.py, handoff.py, verdict.py
├── agents/                # orchestrator.py, writer.py, evaluator.py
├── prompts/               # one .txt per LLM call — edit to change behavior
├── pipeline.py            # the draft → evaluate → retry → apply loop
└── storage.py             # load/save bible.json + scene .txt files
```

Every prompt is a plain text file. Edit prompts to change behavior.
