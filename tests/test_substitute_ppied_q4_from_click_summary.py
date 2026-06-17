import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.substitute_ppied_q4_from_click_summary import (  # noqa: E402
    PpiedQ4SubstitutionConfig,
    PpiedQ4Substituter,
)


class PpiedQ4SubstituterTest(unittest.TestCase):
    def test_transform_replaces_ppied_q4_from_summary_click_rate(self) -> None:
        updater = PpiedQ4Substituter(
            PpiedQ4SubstitutionConfig(
                source_data_path="source.jsonl",
                click_summary_path="summary.jsonl",
                output_path="output.jsonl",
            )
        )
        source_rows = [
            {
                "query": "q0",
                "response": "r0",
                "ppied_scores": {"q1": 1, "q2": 2, "q3": 3, "q4": 4},
            },
            {
                "query": "q1",
                "response": "r1",
                "ppied_scores": {"q1": 10, "q2": 20, "q3": 30, "q4": 40},
            },
        ]
        summary_rows = [
            {"item_index": 1, "tool_call_profile_percent": 0.75555},
            {"item_index": 0, "tool_call_profile_percent": "0.24444"},
        ]

        rows = updater.transform(source_rows, summary_rows)

        self.assertEqual([row["query"] for row in rows], ["q1", "q0"])
        self.assertEqual(rows[0]["ppied_scores"], {"q4": 0.7556})
        self.assertEqual(rows[1]["ppied_scores"], {"q4": 0.2444})
        self.assertEqual(source_rows[0]["ppied_scores"]["q1"], 1)

    def test_run_writes_updated_jsonl(self) -> None:
        temp_path = Path(".tmp") / "test_substitute_ppied_q4"
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(temp_path, ignore_errors=True))

        source_path = temp_path / "source.jsonl"
        summary_path = temp_path / "summary.jsonl"
        output_path = temp_path / "missing" / "output.jsonl"
        source_path.write_text(
            json.dumps(
                {
                    "query": "q",
                    "response": "r",
                    "ppied_scores": {"q1": 1, "q2": 2, "q3": 3, "q4": 4},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        summary_path.write_text(
            json.dumps({"item_index": 0, "clickrate": 0.6}) + "\n",
            encoding="utf-8",
        )
        updater = PpiedQ4Substituter(
            PpiedQ4SubstitutionConfig(
                source_data_path=str(source_path),
                click_summary_path=str(summary_path),
                output_path=str(output_path),
            )
        )

        updater.run()

        row = json.loads(output_path.read_text(encoding="utf-8").strip())
        self.assertEqual(row["ppied_scores"], {"q4": 0.6})

    def test_transform_rejects_missing_ppied_scores(self) -> None:
        updater = PpiedQ4Substituter(
            PpiedQ4SubstitutionConfig(
                source_data_path="source.jsonl",
                click_summary_path="summary.jsonl",
                output_path="output.jsonl",
            )
        )

        with self.assertRaisesRegex(ValueError, "ppied_scores"):
            updater.transform(
                [{"query": "q", "response": "r"}],
                [{"item_index": 0, "tool_call_profile_percent": 0.5}],
            )


if __name__ == "__main__":
    unittest.main()
