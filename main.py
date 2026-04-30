import argparse
import re
import sys

from fictory.pipeline import run


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower()).strip("-")
    return s[:40] or "story"


def main() -> int:
    p = argparse.ArgumentParser(description="Fictory — multi-LLM long-form fiction generator")
    p.add_argument("--premise", required=True, help="One- or two-sentence premise")
    p.add_argument("--genre", required=True, help="Genre (e.g. 'thriller', 'literary sci-fi')")
    p.add_argument("--scenes", type=int, default=3, help="Number of scenes to generate")
    p.add_argument("--story-id", default=None, help="Output dir name under stories/ (auto-derived if omitted)")
    args = p.parse_args()

    story_id = args.story_id or _slug(args.premise)
    run(premise=args.premise, genre=args.genre, num_scenes=args.scenes, story_id=story_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
