import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = "configs/substitute_ppied_q4_from_click_summary.yaml"
DEFAULT_CLICK_RATE_FIELDS = ("clickrate", "click_rate", "tool_call_profile_percent")


@dataclass
class PpiedQ4SubstitutionConfig:
    source_data_path: str
    click_summary_path: str
    output_path: str
    click_rate_field: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PpiedQ4SubstitutionConfig":
        valid_keys = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")

        missing = [
            key
            for key in ("source_data_path", "click_summary_path", "output_path")
            if key not in data or data[key] in (None, "")
        ]
        if missing:
            raise ValueError(f"Missing config key(s): {', '.join(missing)}")
        return cls(**data)


class JsonlProcessor:
    """Base helper for JSONL scripts that transform record streams."""

    def load_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}.")
                rows.append(row)
        return rows

    def write_jsonl(self, path: str | Path, rows: list[dict[str, Any]]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


class PpiedQ4Substituter(JsonlProcessor):
    """Substitute ppied_scores.q4 with click rates from a summary file."""

    def __init__(self, config: PpiedQ4SubstitutionConfig) -> None:
        self.config = config

    def run(self) -> Path:
        source_rows = self.load_jsonl(self.config.source_data_path)
        summary_rows = self.load_jsonl(self.config.click_summary_path)
        output_rows = self.transform(source_rows, summary_rows)
        output_path = Path(self.config.output_path)
        self.write_jsonl(output_path, output_rows)
        return output_path

    def transform(
        self,
        source_rows: list[dict[str, Any]],
        summary_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return source rows selected by summary item_index with q4 replaced."""

        output_rows = []
        seen_indexes: set[int] = set()
        for summary_offset, summary_row in enumerate(summary_rows):
            item_index = self.get_item_index(summary_row, summary_offset)
            if item_index in seen_indexes:
                raise ValueError(f"Duplicate item_index in click summary: {item_index}")
            seen_indexes.add(item_index)
            if item_index < 0 or item_index >= len(source_rows):
                raise ValueError(
                    f"Summary item_index {item_index} is outside source data range "
                    f"0..{len(source_rows) - 1}."
                )

            click_rate = self.get_click_rate(summary_row)
            output_rows.append(self.replace_ppied_q4(source_rows[item_index], click_rate))
        return output_rows

    @staticmethod
    def get_item_index(summary_row: dict[str, Any], fallback: int) -> int:
        """Return the source row index recorded by generate.py."""

        if "item_index" not in summary_row:
            return fallback
        return int(summary_row["item_index"])

    def get_click_rate(self, summary_row: dict[str, Any]) -> Any:
        """Read click rate from the configured or known summary field."""

        if self.config.click_rate_field:
            if self.config.click_rate_field not in summary_row:
                raise ValueError(
                    f"Missing click rate field: {self.config.click_rate_field}"
                )
            return summary_row[self.config.click_rate_field]

        for field in DEFAULT_CLICK_RATE_FIELDS:
            if field in summary_row:
                return summary_row[field]
        raise ValueError(
            "Missing click rate field. Expected one of: "
            + ", ".join(DEFAULT_CLICK_RATE_FIELDS)
        )

    @staticmethod
    def normalize_click_rate(click_rate: Any) -> float:
        """Return the click rate rounded to four decimal places."""

        return round(float(click_rate), 4)

    @classmethod
    def replace_ppied_q4(cls, source_row: dict[str, Any], click_rate: Any) -> dict[str, Any]:
        """Copy one row, remove ppied q1-q3, and set ppied q4."""

        row = copy.deepcopy(source_row)
        ppied_scores = row.get("ppied_scores")
        if not isinstance(ppied_scores, dict):
            raise ValueError("Source row is missing object field: ppied_scores")

        for key in ("q1", "q2", "q3"):
            ppied_scores.pop(key, None)
        ppied_scores["q4"] = cls.normalize_click_rate(click_rate)
        return row


def parse_cli_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Apply generate click summary rates to source ppied_scores.q4."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="YAML config path.",
    )
    return vars(parser.parse_args())


def load_yaml_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def build_config(cli_args: dict[str, Any]) -> PpiedQ4SubstitutionConfig:
    return PpiedQ4SubstitutionConfig.from_dict(load_yaml_config(cli_args["config"]))


def main() -> None:
    output_path = PpiedQ4Substituter(build_config(parse_cli_args())).run()
    print(f"Wrote updated data to {output_path}")


if __name__ == "__main__":
    main()
