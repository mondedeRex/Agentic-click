import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate import (
    ClickDistribution,
    Generator,
    GeneratorConfig,
    ItemToolCallSummary,
    configure_noisy_loggers,
)


class GeneratorMessageTest(unittest.TestCase):
    def test_build_message_last_message_contains_assistant_reply(self) -> None:
        generator = Generator(GeneratorConfig())
        messages = generator.build_message(
            system_prompt="SYSTEMPROMPT",
            reply="Here is the assistant reply.",
        )

        self.assertEqual(
            messages[-1],
            {
                "role": "user",
                "content": "THE ASSISTANT REPLY:\nHere is the assistant reply.",
            },
        )
        
        print(f"Generated messages: {messages}")


class ItemToolCallSummaryTest(unittest.TestCase):
    def test_rows_count_tool_call_profiles_per_item(self) -> None:
        summary = ItemToolCallSummary()

        for profile_index in range(10):
            summary.add_result(
                {
                    "item_index": 3,
                    "profile_index": profile_index,
                    "ok": True,
                    "input": {
                        "query": "query",
                        "response": "response",
                    },
                    "output": {
                        "tool_calls": [
                            {
                                "id": f"call_{profile_index}",
                                "type": "function",
                                "function": {
                                    "name": "click",
                                    "arguments": {"reason": "interested"},
                                },
                            }
                        ]
                        if profile_index in {1, 4, 7}
                        else []
                    },
                }
            )

        rows = summary.rows()

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["item_index"], 3)
        self.assertEqual(rows[0]["query"], "query")
        self.assertEqual(rows[0]["response"], "response")
        self.assertEqual(rows[0]["total_profiles"], 10)
        self.assertEqual(rows[0]["tool_call_profiles"], 3)
        self.assertEqual(rows[0]["total_assignments"], 10)
        self.assertEqual(rows[0]["tool_call_assignments"], 3)
        self.assertAlmostEqual(rows[0]["tool_call_profile_percent"], 0.3)

    def test_rows_count_broadcast_profiles_once(self) -> None:
        summary = ItemToolCallSummary()

        for model_index in range(2):
            summary.add_result(
                {
                    "item_index": 1,
                    "profile_index": 5,
                    "ok": True,
                    "input": {
                        "query": "query",
                        "response": "response",
                    },
                    "output": {
                        "tool_calls": [
                            {
                                "id": f"call_{model_index}",
                                "type": "function",
                                "function": {
                                    "name": "click",
                                    "arguments": {"reason": "interested"},
                                },
                            }
                        ]
                    },
                }
            )

        row = summary.rows()[0]

        self.assertEqual(row["total_profiles"], 1)
        self.assertEqual(row["tool_call_profiles"], 1)
        self.assertEqual(row["total_assignments"], 2)
        self.assertEqual(row["tool_call_assignments"], 2)

    def test_get_output_path_uses_run_directory(self) -> None:
        generator = Generator(GeneratorConfig(output="results/generate_20260611_120000"))

        path = generator.get_output_path()

        self.assertEqual(
            path,
            Path("results/generate_20260611_120000/generate.jsonl"),
        )

    def test_get_tool_summary_output_path_uses_fixed_filename(self) -> None:
        generator = Generator(GeneratorConfig(output="results/generate_20260611_120000"))

        path = generator.get_tool_summary_output_path()

        self.assertEqual(
            path,
            Path("results/generate_20260611_120000/generate_tool_summary.jsonl"),
        )

    def test_output_paths_share_cached_output_dir(self) -> None:
        generator = Generator(GeneratorConfig(output="results/first"))
        generator.output_dir = Path("results/cached")
        generator.config.output = "results/second"

        self.assertEqual(
            generator.get_output_path(),
            Path("results/cached/generate.jsonl"),
        )
        self.assertEqual(
            generator.get_tool_summary_output_path(),
            Path("results/cached/generate_tool_summary.jsonl"),
        )

    def test_write_tool_summary_creates_missing_output_directory(self) -> None:
        generator = Generator(GeneratorConfig())
        summary = ItemToolCallSummary()
        summary.add_result(
            {
                "item_index": 0,
                "profile_index": 0,
                "ok": True,
                "input": {
                    "query": "query",
                    "response": "response",
                },
                "output": {"tool_calls": []},
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing" / "generate_tool_summary.jsonl"

            generator.write_tool_summary(output_path, summary)

            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.parent.is_dir())

    def test_copy_config_snapshot_writes_config_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source.yaml"
            output_dir = temp_path / "out"
            source_path.write_text("limit: 1\n", encoding="utf-8")
            generator = Generator(
                GeneratorConfig(
                    output=str(output_dir),
                    config_path=str(source_path),
                )
            )

            generator.copy_config_snapshot()

            snapshot_path = output_dir / "config.yaml"
            self.assertEqual(
                snapshot_path.read_text(encoding="utf-8"),
                "limit: 1\n",
            )

    def test_click_distribution_renders_histogram(self) -> None:
        text = ClickDistribution(
            [
                {"tool_call_profiles": 0},
                {"tool_call_profiles": 2},
                {"tool_call_profiles": 2},
            ]
        ).render()

        self.assertIn("Click distribution", text)
        self.assertIn("Total items: 3", text)
        self.assertIn("  0 clicks |     1 items | 33.33%", text)
        self.assertIn("  2 clicks |     2 items | 66.67%", text)

    def test_write_text_creates_missing_output_directory(self) -> None:
        generator = Generator(GeneratorConfig())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "missing" / "click_distribution.txt"

            generator.write_text(output_path, "Click distribution")

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "Click distribution\n",
            )

    def test_configure_noisy_loggers_sets_warning_level(self) -> None:
        logging.getLogger("httpx").setLevel(logging.DEBUG)

        configure_noisy_loggers()

        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)

if __name__ == "__main__":
    unittest.main()
