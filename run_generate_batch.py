import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIG_DIR = Path("configs/generate_batch")
GENERATE_SCRIPT = Path("generate.py")


class ConfigDirectoryRunner:
    """Run generate.py once for every YAML config in a directory."""

    def __init__(self, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
        self.config_dir = config_dir

    def run(self) -> int:
        """Run all discovered configs and return a process-style exit code."""

        config_paths = self.discover_configs()
        if not config_paths:
            print(f"No YAML configs found in {self.config_dir}")
            return 0

        failed_configs = []
        for config_path in config_paths:
            command = self.build_command(config_path)
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(command)
            if result.returncode != 0:
                print(
                    f"Config failed with exit code {result.returncode}: {config_path}"
                )
                failed_configs.append((config_path, result.returncode))

        if failed_configs:
            print("Failed configs:")
            for config_path, returncode in failed_configs:
                print(f"- {config_path}: exit code {returncode}")
            return 1
        return 0

    def discover_configs(self) -> list[Path]:
        """Return YAML configs in stable filename order."""

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory does not exist: {self.config_dir}")
        if not self.config_dir.is_dir():
            raise NotADirectoryError(f"Config path is not a directory: {self.config_dir}")

        config_paths = [
            path
            for path in self.config_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        ]
        return sorted(config_paths, key=lambda path: path.name)

    @staticmethod
    def build_command(config_path: Path) -> list[str]:
        """Return the generate.py command for one config."""

        return [sys.executable, str(GENERATE_SCRIPT), "--config", str(config_path)]


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generate.py for every YAML config in a directory."
    )
    parser.add_argument(
        "--config_dir",
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing generate YAML configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()
    raise SystemExit(ConfigDirectoryRunner(Path(args.config_dir)).run())


if __name__ == "__main__":
    main()
