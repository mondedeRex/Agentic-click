import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = "configs/user_profiles.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate user profiles from YAML.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def build_profiles(
    identities: list[str], interests: list[str]
) -> list[dict[str, Any]]:
    profiles = []
    for identity_idx, identity in enumerate(identities):
        for interest_idx, interest in enumerate(interests):
            profiles.append(
                {
                    "profile_id": f"u{len(profiles):04d}",
                    "identity": identity,
                    "interest": interest,
                    "identity_idx": identity_idx,
                    "interest_idx": interest_idx,
                }
            )
    return profiles


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    config = load_config(parse_args().config)
    identities = config.get("identities") or []
    interests = config.get("interests") or []
    output_path = config.get("output_path", "data/user_profiles.jsonl")

    if not identities:
        raise ValueError("identities must not be empty.")
    if not interests:
        raise ValueError("interests must not be empty.")

    profiles = build_profiles(identities, interests)
    write_jsonl(output_path, profiles)
    print(f"Wrote {len(profiles)} user profiles to {output_path}")


if __name__ == "__main__":
    main()
