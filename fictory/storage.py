from pathlib import Path

from .models.story_bible import StoryBible

STORIES_DIR = Path("stories")


def story_dir(story_id: str) -> Path:
    d = STORIES_DIR / story_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(story_id: str, bible: StoryBible) -> None:
    d = story_dir(story_id)
    (d / "bible.json").write_text(bible.model_dump_json(indent=2))
    for n, prose in bible.generated_prose.items():
        (d / f"scene_{int(n):03d}.txt").write_text(prose)


def load(story_id: str) -> StoryBible:
    path = STORIES_DIR / story_id / "bible.json"
    return StoryBible.model_validate_json(path.read_text())
