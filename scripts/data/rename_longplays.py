import json
import re
from pathlib import Path
from urllib.parse import unquote


LONGPLAYS_JSON = Path(__file__).resolve().parents[2] / "agent_1" / "data" / "longplays.json"
LONGPLAYS_DIR = Path.home() / "datasets" / "longplays"


def build_rename_map() -> dict[str, str]:
    """Map original filenames (as .mp4) to cleaned game-name filenames."""
    with open(LONGPLAYS_JSON) as f:
        entries = json.load(f)

    rename_map = {}
    for entry in entries:
        url = entry["downloads"][0]
        original_stem = Path(unquote(url)).stem
        original_name = f"{original_stem}.mp4"

        safe_name = re.sub(r'[\\/:*?"<>|]', "", entry["game"])
        safe_name = safe_name.replace(" ", "_")
        new_name = f"{safe_name}.mp4"

        rename_map[original_name] = new_name
    return rename_map


def main():
    rename_map = build_rename_map()

    for old_name, new_name in sorted(rename_map.items()):
        old_path = LONGPLAYS_DIR / old_name
        new_path = LONGPLAYS_DIR / new_name
        if old_path.exists():
            print(f"{old_name}  ->  {new_name}")
            old_path.rename(new_path)
        else:
            print(f"MISSING: {old_name}")


if __name__ == "__main__":
    main()
