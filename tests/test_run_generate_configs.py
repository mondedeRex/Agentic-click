import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_generate_batch import DEFAULT_CONFIG_DIR, ConfigDirectoryRunner  # noqa: E402


class ConfigDirectoryRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_path = Path(".tmp") / "test_run_generate_configs"
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
        self.temp_path.mkdir(parents=True)
        self.addCleanup(lambda: shutil.rmtree(self.temp_path, ignore_errors=True))

    def test_discover_configs_returns_yaml_files_in_name_order(self) -> None:
        (self.temp_path / "b.yml").write_text("limit: 2\n", encoding="utf-8")
        (self.temp_path / "a.yaml").write_text("limit: 1\n", encoding="utf-8")
        (self.temp_path / "notes.txt").write_text("ignore\n", encoding="utf-8")

        configs = ConfigDirectoryRunner(self.temp_path).discover_configs()

        self.assertEqual(
            [path.name for path in configs],
            ["a.yaml", "b.yml"],
        )

    def test_build_command_uses_generate_py_config_command(self) -> None:
        config_path = Path("configs/generate_batch/a.yaml")

        command = ConfigDirectoryRunner.build_command(config_path)

        self.assertEqual(
            command,
            [sys.executable, "generate.py", "--config", str(config_path)],
        )

    def test_run_continues_after_failed_config(self) -> None:
        (self.temp_path / "a.yaml").write_text("limit: 1\n", encoding="utf-8")
        (self.temp_path / "b.yaml").write_text("limit: 2\n", encoding="utf-8")

        with patch("run_generate_batch.subprocess.run") as run_mock:
            run_mock.side_effect = [
                type("Result", (), {"returncode": 3})(),
                type("Result", (), {"returncode": 0})(),
            ]

            exit_code = ConfigDirectoryRunner(self.temp_path).run()

        self.assertEqual(exit_code, 1)
        self.assertEqual(run_mock.call_count, 2)

    def test_default_config_dir_is_generate_batch(self) -> None:
        runner = ConfigDirectoryRunner()

        self.assertEqual(runner.config_dir, DEFAULT_CONFIG_DIR)

    def test_config_dir_can_be_passed_to_runner(self) -> None:
        runner = ConfigDirectoryRunner(self.temp_path)

        self.assertEqual(runner.config_dir, self.temp_path)


if __name__ == "__main__":
    unittest.main()
