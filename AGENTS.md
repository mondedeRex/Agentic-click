# Agent Instructions

## Project Overview

This repo generates user-perspective reactions to assistant replies. The main
entry point is `generate.py`.

Data rows are JSONL objects and must contain:

- `query`
- `response`

User profiles are JSONL objects and should contain:

- `profile_id`
- `identity`
- `interest`

`generate.py` crossovers selected data rows with user profiles, dispatches the
work items evenly across selected supported models by default, and writes
results as JSONL.

## Configuration

Runtime configuration must come from YAML files. Do not add direct CLI
arguments for individual runtime options.

Default config:

```bash
configs/generate.yaml
```

Run with:

```bash
python generate.py --config configs/generate.yaml
```

If `--config` is omitted, `generate.py` uses `configs/generate.yaml`.

Supported models are configured in:

```bash
supported_models.yaml
```

User profiles are configured/generated through:

```bash
configs/user_profiles.yaml
generate_user_profiles.py
data/user_profiles.jsonl
```

## Message Construction

Keep message construction simple and explicit:

- Render the system prompt with `{{query}}` and `{{profile}}`.
- The rendered variable should be named `system_prompt`.
- The model user message should contain the assistant reply with this label:

```text
THE ASSISTANT REPLY:
...
```

`build_profile` should only include:

- `identity`
- `interest`

Do not include profile indexes or generated metadata in the prompt.

## Dispatch Behavior

Default dispatch mode is `split`.

In `split` mode, work items are evenly assigned across selected models. For
example, 800 work items and 2 enabled models means each model handles 400 work
items.

Keep `broadcast` available only as an explicit config option.

## Results

Each run writes results to a timestamped directory:

```bash
results/generate_<timestamp>/
```

The raw assignment output is written inside that directory:

```bash
results/generate_<timestamp>/generate.jsonl
```

The YAML config used for the run is copied into the same directory:

```bash
results/generate_<timestamp>/config.yaml
```

Each raw output row should record input fields, profile metadata, selected
model, success/failure status, model output, and usage when available.

For each generated result file, `generate.py` also writes an item-level tool
summary JSONL inside the same run directory:

```bash
results/generate_<timestamp>/generate_tool_summary.jsonl
```

Each summary row records one input item (`query`/`response`), the number of
profiles evaluated for that item, how many profiles called tools, and the
corresponding percentage.

At the end of each run, `generate.py` prints a terminal-friendly click
distribution and saves it to:

```bash
results/generate_<timestamp>/click_distribution.txt
```

To apply item-level click rates back to the original source data, configure:

```bash
configs/apply_click_summary.yaml
```

Run with:

```bash
python scripts/apply_click_summary.py --config configs/apply_click_summary.yaml
```

The script reads the generate-time `data_path` source JSONL and a
`generate_tool_summary.jsonl`, writes one output row per summary item, removes
`ppied_scores.q1` through `ppied_scores.q3`, and replaces `ppied_scores.q4`
with the click rate from the summary row rounded to four decimal places.

## Logging

When `log_level: DEBUG`, log exactly one complete message example. Do not log
every request's full message because data/profile crossover can create many
work items.

Keep noisy HTTP/client logs suppressed unless they are warnings or errors.

## Testing

Use standard library `unittest`; do not add a test dependency unless needed.

Run tests with either:

```bash
python tests/test_generator.py
```

or:

```bash
python -m unittest discover
```

Before finishing code changes, run:

```bash
python -m py_compile generate.py generate_user_profiles.py
python -m unittest discover
```


# Rules
- 模块化。使用class和继承来实现核心功能
- plan first. 进行功能添加的时候先计划，然后给我看，然后再做
- 完成之后，及时修改AGENT.md对应的内容
- 在重要功能实现后，修改单元测试
- 不需要关注本地的包是否安装了
- 实现简洁清晰
- 重要的函数至少要简洁的给出 doc string
- 简单的设计不需要做plan，单元测试。
