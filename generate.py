import argparse
import asyncio
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import random
import yaml
from openai import AsyncOpenAI
from tqdm import tqdm


DATA_PATH = "/mnt/model/liang.zeng/nips/datasets/NaiAD/NaiAD_main.jsonl"
USER_PROFILES_PATH = "data/user_profiles.jsonl"
SYSTEM_PROMPT_PATH = "system_prompt.txt"
SUPPORTED_MODELS_PATH = "supported_models.yaml"
DEFAULT_CONFIG_PATH = "configs/generate.yaml"
DEFAULT_OUTPUT_DIR = "results"
RESULTS_FILENAME = "generate.jsonl"
TOOL_SUMMARY_FILENAME = "generate_tool_summary.jsonl"
CONFIG_SNAPSHOT_FILENAME = "config.yaml"
CLICK_DISTRIBUTION_FILENAME = "click_distribution.txt"
NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3")



TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click an ad if you chooses to.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for clicking the ad.",
                    }
                },
                "required": ["reason"],
            },
        },
    }
]

@dataclass
class GeneratorConfig:
    data_path: str = DATA_PATH
    user_profiles_path: str = USER_PROFILES_PATH
    supported_models_path: str = SUPPORTED_MODELS_PATH
    system_prompt_path: str = SYSTEM_PROMPT_PATH
    models: str | list[str] = "all"
    dispatch_mode: str = "split"
    start: int = 0
    limit: int | None = None
    indices: str | list[int] | None = None
    concurrency: int = 8
    per_model_concurrency: int = 4
    timeout: float = 120.0
    max_retries: int = 2
    output: str | None = None
    log_level: str = "INFO"
    profile_limit: int | None = None
    config_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratorConfig":
        valid_keys = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")
        return cls(**data)


class ItemToolCallSummary:
    """Aggregate tool-call counts for every source item across profiles."""

    def __init__(self) -> None:
        self.items: dict[int, dict[str, Any]] = {}

    def add_result(self, result: dict[str, Any]) -> None:
        """Add one assignment result to the item-level summary."""

        item_index = int(result["item_index"])
        item = self.items.setdefault(
            item_index,
            {
                "item_index": item_index,
                "query": None,
                "response": None,
                "_profile_indexes": set(),
                "_tool_call_profile_indexes": set(),
                "total_assignments": 0,
                "ok_assignments": 0,
                "failed_assignments": 0,
                "tool_call_assignments": 0,
                "tool_call_count": 0,
            },
        )

        input_data = result.get("input") or {}
        if item["query"] is None and "query" in input_data:
            item["query"] = input_data["query"]
        if item["response"] is None and "response" in input_data:
            item["response"] = input_data["response"]

        profile_index = int(result["profile_index"])
        item["_profile_indexes"].add(profile_index)
        item["total_assignments"] += 1
        if result.get("ok"):
            item["ok_assignments"] += 1
        else:
            item["failed_assignments"] += 1

        tool_calls = ((result.get("output") or {}).get("tool_calls")) or []
        if tool_calls:
            item["_tool_call_profile_indexes"].add(profile_index)
            item["tool_call_assignments"] += 1
            item["tool_call_count"] += len(tool_calls)

    def rows(self) -> list[dict[str, Any]]:
        """Return stable JSONL-ready rows sorted by item index."""

        rows = []
        for item in sorted(self.items.values(), key=lambda row: row["item_index"]):
            total = item["total_assignments"]
            total_profiles = len(item["_profile_indexes"])
            tool_call_profiles = len(item["_tool_call_profile_indexes"])
            tool_call_assignments = item["tool_call_assignments"]
            row = {
                key: value
                for key, value in item.items()
                if not key.startswith("_")
            }
            row["total_profiles"] = total_profiles
            row["tool_call_profiles"] = tool_call_profiles
            row["tool_call_profile_percent"] = (
                tool_call_profiles / total_profiles if total_profiles else 0.0
            )
            row["tool_call_assignment_percent"] = (
                tool_call_assignments / total if total else 0.0
            )
            rows.append(row)
        return rows


class ClickDistribution:
    """Render item-level click counts as a terminal-friendly histogram."""

    def __init__(self, summary_rows: list[dict[str, Any]]) -> None:
        self.summary_rows = summary_rows

    def render(self) -> str:
        """Return the click distribution as plain text."""

        total_items = len(self.summary_rows)
        if total_items == 0:
            return "Click distribution\nNo items were summarized."

        distribution: dict[int, int] = {}
        click_counts = []
        max_clicks = 0
        for row in self.summary_rows:
            clicks = int(row["tool_call_profiles"])
            click_counts.append(clicks)
            distribution[clicks] = distribution.get(clicks, 0) + 1
            max_clicks = max(max_clicks, clicks)

        max_items_in_bucket = max(distribution.values())
        mean = sum(click_counts) / total_items
        variance = sum((clicks - mean) ** 2 for clicks in click_counts) / total_items
        stddev = math.sqrt(variance)
        lines = [
            "Click distribution",
            f"Total items: {total_items}",
            f"Mean clicked profiles per item: {mean:.4f}",
            f"Std clicked profiles per item: {stddev:.4f}",
            "Clicked profiles per item:",
        ]
        for clicks in range(max_clicks + 1):
            item_count = distribution.get(clicks, 0)
            percent = item_count / total_items
            bar = self.render_bar(item_count, max_items_in_bucket)
            lines.append(f"{clicks:>3} clicks | {item_count:>5} items | {percent:6.2%} | {bar}")
        return "\n".join(lines)

    @staticmethod
    def render_bar(value: int, maximum: int, width: int = 40) -> str:
        """Return an ASCII bar scaled to the largest bucket."""

        if value <= 0 or maximum <= 0:
            return ""
        return "#" * max(1, round(value / maximum * width))


class Generator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.system_template = ""
        self.supported_models: dict[str, dict[str, Any]] = {}
        self.selected_models: dict[str, dict[str, Any]] = {}
        self.clients: dict[str, AsyncOpenAI] = {}
        self.logged_message_example = False
        self.message_log_lock = asyncio.Lock()
        self.output_dir: Path | None = None

    async def run(self) -> Path:
        self.load_system_prompt()
        rows = self.load_jsonl(self.config.data_path)
        subset = self.select_subset(rows)
        if not subset:
            raise ValueError("Selected data subset is empty.")
        profiles = self.load_jsonl(self.config.user_profiles_path)
        if not profiles:
            raise ValueError("User profile set is empty.")

        if self.config.profile_limit is not None:
            # randomly select a subset of profiles to reduce total number of requests if needed
            random_indices = list(range(len(profiles)))
            random.seed(42)
            random.shuffle(random_indices)
            selected_indices = set(random_indices[: self.config.profile_limit])
            profiles = [profile for i, profile in enumerate(profiles) if i in selected_indices]

        self.supported_models = self.load_supported_models(
            self.config.supported_models_path
        )
        self.selected_models = self.select_models(self.supported_models)

        self.output_dir = self.get_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.get_output_path()
        self.copy_config_snapshot()

        work_items = self.build_work_items(subset, profiles)

        logging.info("Loaded %d rows, selected %d rows.", len(rows), len(subset))
        logging.info("Loaded %d user profiles.", len(profiles))
        logging.info("Built %d row/profile work items.", len(work_items))
        logging.info("Selected models: %s", ", ".join(self.selected_models))
        logging.info("Writing results to %s", output_path)

        self.clients = {
            name: self.make_client(config)
            for name, config in self.selected_models.items()
        }

        global_sem = asyncio.Semaphore(self.config.concurrency)
        model_sems = {
            name: asyncio.Semaphore(self.config.per_model_concurrency)
            for name in self.selected_models
        }

        assignments = self.build_assignments(work_items)
        logging.info(
            "Dispatch counts: %s",
            ", ".join(
                f"{model_name}={count}"
                for model_name, count in self.count_assignments(assignments).items()
            ),
        )

        tasks = [
            asyncio.create_task(
                self.run_one(
                    item_index=item_index,
                    item=item,
                    profile_index=profile_index,
                    profile=profile,
                    model_name=model_name,
                    model_config=self.selected_models[model_name],
                    client=self.clients[model_name],
                    global_sem=global_sem,
                    model_sem=model_sems[model_name],
                )
            )
            for item_index, item, profile_index, profile, model_name in assignments
        ]

        tool_summary = ItemToolCallSummary()
        with open(output_path, "w", encoding="utf-8") as f:
            progress = tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Generating All Assignments",
                unit="req",
            )
            for task in progress:
                result = await task
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                tool_summary.add_result(result)

        tool_summary_path = self.get_tool_summary_output_path()
        self.write_tool_summary(tool_summary_path, tool_summary)
        logging.info("Wrote item tool-call summary to %s", tool_summary_path)

        click_distribution_path = self.get_click_distribution_output_path()
        click_distribution = ClickDistribution(tool_summary.rows()).render()
        self.write_text(click_distribution_path, click_distribution)
        print(click_distribution)
        logging.info("Wrote click distribution to %s", click_distribution_path)

        await self.close()
        return output_path

    async def close(self) -> None:
        for client in self.clients.values():
            await client.close()

    def load_system_prompt(self) -> None:
        with open(self.config.system_prompt_path, "r", encoding="utf-8") as f:
            self.system_template = f.read()
        if not self.system_template.strip():
            raise ValueError("System prompt is empty.")

    def load_jsonl(self, path: str) -> list[tuple[int, dict[str, Any]]]:
        rows: list[tuple[int, dict[str, Any]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_no + 1}: {exc}"
                    ) from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no + 1}.")
                rows.append((line_no, item))
        return rows

    def select_subset(
        self, rows: list[tuple[int, dict[str, Any]]]
    ) -> list[tuple[int, dict[str, Any]]]:
        if self.config.start < 0:
            raise ValueError("start must be >= 0.")
        if self.config.limit is not None and self.config.limit < 0:
            raise ValueError("limit must be >= 0.")

        indices = self.normalize_indices(self.config.indices)
        if indices is not None:
            selected = [(idx, row) for idx, row in rows if idx in indices]
        else:
            selected = rows[self.config.start :]

        if self.config.limit is not None:
            selected = selected[: self.config.limit]
        return selected

    def load_supported_models(self, path: str) -> dict[str, dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        models = data.get("supported_models", data)
        if not isinstance(models, dict) or not models:
            raise ValueError(f"No supported models found in {path}.")
        return models

    def select_models(
        self, supported_models: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        models_arg = self.config.models
        if models_arg == "all":
            return supported_models

        if isinstance(models_arg, str):
            selected_names = [name.strip() for name in models_arg.split(",") if name.strip()]
        else:
            selected_names = [str(name) for name in models_arg]

        missing = [name for name in selected_names if name not in supported_models]
        if missing:
            available = ", ".join(supported_models)
            raise ValueError(
                f"Unknown model(s): {', '.join(missing)}. Available models: {available}"
            )
        return {name: supported_models[name] for name in selected_names}

    def make_client(self, model_config: dict[str, Any]) -> AsyncOpenAI:
        configured_key = model_config.get("api_key")
        api_key = str(configured_key or os.getenv("OPENAI_API_KEY") or "EMPTY")
        if api_key == "<API_KEY>":
            api_key = "EMPTY"

        base_url = model_config.get("base_url")
        if not base_url:
            raise ValueError("Each model config must include base_url.")
        base_url = str(base_url).rstrip("/")
        if model_config.get("engine") == "vllm" and not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    async def run_one(
        self,
        *,
        item_index: int,
        item: dict[str, Any],
        profile_index: int,
        profile: dict[str, Any],
        model_name: str,
        model_config: dict[str, Any],
        client: AsyncOpenAI,
        global_sem: asyncio.Semaphore,
        model_sem: asyncio.Semaphore,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "item_index": item_index,
            "profile_index": profile_index,
            "profile_id": profile.get("profile_id"),
            "model_name": model_name,
            "model": model_config.get("model"),
        }

        async with global_sem, model_sem:
            try:
                query = str(item["query"])
                reply = str(item["response"])
                profile_text = self.build_profile(profile)
                system_prompt = self.render_system_prompt(query, profile_text)
                messages = self.build_message(system_prompt, reply)
                await self.log_message_example(
                    item_index=item_index,
                    profile_index=profile_index,
                    model_name=model_name,
                    messages=messages,
                )
                result["input"] = {
                    "query": query,
                    "response": reply,
                    "profile": profile,
                }
                response = await client.chat.completions.create(
                    model=str(model_config["model"]),
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    extra_body={
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        },
                    },
                )
                result["ok"] = True
                result["output"] = self.parse_message(response.choices[0].message)
                if response.usage is not None:
                    result["usage"] = response.usage.model_dump()
            except Exception as exc:
                result["ok"] = False
                result["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
        return result

    def build_work_items(
        self,
        subset: list[tuple[int, dict[str, Any]]],
        profiles: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any], int, dict[str, Any]]]:
        return [
            (item_index, item, profile_index, profile)
            for item_index, item in subset
            for profile_index, profile in profiles
        ]

    def build_assignments(
        self, work_items: list[tuple[int, dict[str, Any], int, dict[str, Any]]]
    ) -> list[tuple[int, dict[str, Any], int, dict[str, Any], str]]:
        model_names = list(self.selected_models)
        if self.config.dispatch_mode == "broadcast":
            return [
                (item_index, item, profile_index, profile, model_name)
                for item_index, item, profile_index, profile in work_items
                for model_name in model_names
            ]
        if self.config.dispatch_mode != "split":
            raise ValueError(f"Unsupported dispatch mode: {self.config.dispatch_mode}")

        return [
            (
                item_index,
                item,
                profile_index,
                profile,
                model_names[offset % len(model_names)],
            )
            for offset, (item_index, item, profile_index, profile) in enumerate(
                work_items
            )
        ]

    def count_assignments(
        self, assignments: list[tuple[int, dict[str, Any], int, dict[str, Any], str]]
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, _, _, model_name in assignments:
            counts[model_name] = counts.get(model_name, 0) + 1
        return counts

    def render_system_prompt(self, query: str, profile: str) -> str:
        return self.system_template.replace("{{query}}", query).replace(
            "{{profile}}", profile
        )

    def build_message(
        self, system_prompt: str, reply: str
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"THE ASSISTANT REPLY:\n{reply}"},
        ]

    async def log_message_example(
        self,
        *,
        item_index: int,
        profile_index: int,
        model_name: str,
        messages: list[dict[str, str]],
    ) -> None:
        async with self.message_log_lock:
            if self.logged_message_example:
                return
            logging.debug(
                "Message example for item_index=%s profile_index=%s model_name=%s:\n%s",
                item_index,
                profile_index,
                model_name,
                json.dumps(messages, ensure_ascii=False, indent=2),
            )
            self.logged_message_example = True

    def build_profile(self, profile: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"- Your identity: {profile['identity']}",
                f"- Your interest: {profile['interest']}",
            ]
        )

    def parse_message(self, message: Any) -> dict[str, Any]:
        tool_calls = []
        for call in message.tool_calls or []:
            arguments = call.function.arguments
            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_args = arguments
            tool_calls.append(
                {
                    "id": call.id,
                    "type": call.type,
                    "function": {
                        "name": call.function.name,
                        "arguments": parsed_args,
                    },
                }
            )

        return {
            "content": message.content,
            "tool_calls": tool_calls,
        }

    

    def get_output_dir(self) -> Path:
        """Return the output directory for this run."""

        if self.config.output:
            return Path(self.config.output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(DEFAULT_OUTPUT_DIR) / f"generate_{timestamp}"

    def require_output_dir(self) -> Path:
        """Return the cached output directory after it has been initialized."""

        if self.output_dir is None:
            self.output_dir = self.get_output_dir()
        return self.output_dir
    
    def get_output_path(self) -> Path:
        """Return the raw assignment JSONL path for this run."""

        return self.require_output_dir() / RESULTS_FILENAME
    
    def get_tool_summary_output_path(self) -> Path:
        """Return the item-level tool summary JSONL path for this run."""

        return self.require_output_dir() / TOOL_SUMMARY_FILENAME

    def get_config_snapshot_output_path(self) -> Path:
        """Return the copied run config path for this run."""

        return self.require_output_dir() / CONFIG_SNAPSHOT_FILENAME

    def get_click_distribution_output_path(self) -> Path:
        """Return the click distribution report path for this run."""

        return self.require_output_dir() / CLICK_DISTRIBUTION_FILENAME

    def write_tool_summary(
        self, output_path: Path, tool_summary: ItemToolCallSummary
    ) -> None:
        """Write the item-level tool summary as JSONL."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for row in tool_summary.rows():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def copy_config_snapshot(self) -> None:
        """Copy the YAML config used for this run into the output directory."""

        if not self.config.config_path:
            return
        source_path = Path(self.config.config_path)
        if not source_path.exists():
            logging.warning("Config snapshot source does not exist: %s", source_path)
            return
        output_path = self.get_config_snapshot_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, output_path)

    def write_text(self, output_path: Path, text: str) -> None:
        """Write text output, creating the parent directory if needed."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    @staticmethod
    def normalize_indices(indices: str | list[int] | None) -> set[int] | None:
        if indices is None or indices == "":
            return None
        if isinstance(indices, str):
            return {int(x.strip()) for x in indices.split(",") if x.strip()}
        return {int(x) for x in indices}


def parse_cli_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Run click simulation from a YAML config."
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
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return data


def build_config(cli_args: dict[str, Any]) -> GeneratorConfig:
    data = load_yaml_config(cli_args["config"])
    data["config_path"] = cli_args["config"]
    return GeneratorConfig.from_dict(data)


def configure_noisy_loggers(level: int = logging.WARNING) -> None:
    """Suppress chatty HTTP/client loggers below the given level."""

    for logger_name in NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(level)


def main() -> None:
    config = build_config(parse_cli_args())
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    configure_noisy_loggers()
    output_path = asyncio.run(Generator(config).run())
    logging.info("Done. Results saved to %s", output_path)


if __name__ == "__main__":
    main()
