import argparse
import os

import yaml
from openai import OpenAI


DEFAULT_SUPPORTED_MODELS_PATH = "supported_models.yaml"
DEFAULT_MESSAGE = "Hello. Reply with one short sentence."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one test chat request to a local vLLM OpenAI-compatible server."
    )
    parser.add_argument(
        "--supported-models-path",
        default=DEFAULT_SUPPORTED_MODELS_PATH,
        help="Path to supported_models.yaml.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Key under supported_models. Defaults to the first configured model.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override OpenAI-compatible base URL, for example http://localhost:8000/v1.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model id/path sent to vLLM.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override API key. Defaults to config api_key, OPENAI_API_KEY, or EMPTY.",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="User message to send.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def load_supported_models(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("supported_models", data)
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No supported models found in {path}.")
    return models


def select_model_config(models: dict, model_name: str | None) -> tuple[str, dict]:
    if model_name is None:
        model_name = next(iter(models))
    if model_name not in models:
        available = ", ".join(models)
        raise ValueError(f"Unknown model_name={model_name}. Available: {available}")
    return model_name, models[model_name]


def normalize_base_url(base_url: str, engine: str | None) -> str:
    base_url = base_url.rstrip("/")
    if engine == "vllm" and not base_url.endswith("/v1"):
        return f"{base_url}/v1"
    return base_url


def normalize_api_key(api_key: str | None) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    if api_key == "<API_KEY>":
        return "EMPTY"
    return api_key


def main() -> None:
    args = parse_args()
    models = load_supported_models(args.supported_models_path)
    model_name, config = select_model_config(models, args.model_name)

    model = args.model or config["model"]
    base_url = args.base_url or config["base_url"]
    api_key = normalize_api_key(args.api_key or config.get("api_key"))
    base_url = normalize_base_url(base_url, config.get("engine"))

    print(f"Testing model_name: {model_name}")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Message: {args.message}")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)
    response = client.chat.completions.create(
        model=str(model),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.message},
        ],
    )

    print("\nResponse:")
    print(response.choices[0].message.content)

    if response.usage is not None:
        print("\nUsage:")
        print(response.usage.model_dump())


if __name__ == "__main__":
    main()
