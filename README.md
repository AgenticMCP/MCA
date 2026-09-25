# MCA: Model Context Agent

The Model Context Protocol (MCP) standardizes how a client *accesses* a service's tools, but not how those tools should be *used*. That is usually left to a reactive loop that picks one tool call at a time and decides for itself when it has succeeded. A **Model Context Agent (MCA)** ships that know-how with the server instead: it compiles a natural-language goal into a validated tool-call plan, executes the plan with no model in the loop, and verifies the outcome by re-querying the live service.

This repository contains an MCA for each of four MCP servers:

| Server | Tools | Directory | Engine package |
|---|---:|---|---|
| GitHub | 85 | `github-mca/github-mcp-server-stable/` | `pkg/agenticmcpe/` |
| PostgreSQL | 9 | `postgresql-mca/` | `pkg/agenticmcpe/` |
| Playwright (browser) | 24 | `playwright-mca/` | `pkg/agenticmcpe/` |
| Yahoo Finance | 9 | `finance-mca/` | `pkg/finance_agenticmcpe/` |

The overview and workflow apply to all four. The quick start, usage, and project structure use the GitHub MCA as the example.

## Overview


An MCA has four components, connected by an orchestrator:

| Component | Model use | Role |
|---|---|---|
| **Planner** | Writes the plan | Compiles the goal into an ordered tool-call plan, grounded on the live tool catalog and on verified precedents, and validated before any call |
| **Executor** | None | Replays the plan in order against the live server and records a trace |
| **Verifier** | Writes the dynamic checks | Decides whether the run succeeded, using format, static, and live read-only checks |
| **Knowledge base** | None when serving | Verified goal–plan pairs, grown offline by a task generator and retrieved by the planner |

## Workflow

```
     goal (natural language)
      │
      ▼
   ┌────────────────────────┐   top-3 precedents   ┌────────────────────────┐
   │ 1. PLAN          (LLM) │ ◀─────────────────── │     Knowledge base     │
   └────────────────────────┘                      │  verified task → plan  │
      │ plan.json     ▲                            └────────────────────────┘
      ▼               │ step failed:                           ▲
   ┌────────────────────────┐ replan ≤ R                       │ admit only if
   │ 2. EXECUTE    (no LLM) │                                  │ executed and
   └────────────────────────┘                                  │ verified, no replan
      │ trace.json                                 ┌────────────────────────┐
      ▼                                            │ 4. LEARN     (offline) │
   ┌────────────────────────┐                      │     task generator     │
   │ 3. VERIFY  (read-only) │                      └────────────────────────┘
   └────────────────────────┘
      │ verification.json
      ▼
   verified iff every check passes
```


## Quick start (GitHub)

Requirements:

- Python 3 (the pinned dependencies were resolved under CPython 3.14)
- Go 1.25 or later, to build the GitHub MCP server (the engine builds it on first use)
- An API key for one LLM provider (DeepSeek by default)
- A GitHub personal access token. Tasks run against live GitHub as the token's user, and write tasks create real repositories, so use a test account.

```bash
cd github-mca/github-mcp-server-stable

# Install the Python dependencies
python3 -m venv pkg/venv
pkg/venv/bin/python -m pip install -r pkg/agenticmcpe/requirements.txt

# Add credentials: an LLM key (DEEPSEEK_API_KEY by default) and GITHUB_PERSONAL_ACCESS_TOKEN
cp pkg/agenticmcpe/.env.example pkg/agenticmcpe/.env

# Show the resolved configuration (also builds pkg/mcp_wrapper/bin/github-mcp-server if missing)
pkg/venv/bin/python -m pkg.agenticmcpe config

# Smoke test: plan, execute, and verify a read-only task against live GitHub
pkg/venv/bin/python -m pkg.agenticmcpe selfcheck
```

The self-check looks up your GitHub profile and lists up to five open issues in `github/github-mcp-server`. It ends with `overall  : OK` when the run executed and every check passed.

## Basic usage

Run all commands from `github-mca/github-mcp-server-stable/`.

### Run a task

```bash
pkg/venv/bin/python -m pkg.agenticmcpe run --run-id demo \
    --task "Create a repository named mca-demo with a README.md that contains '# MCA demo'"
```

The planner writes a plan like this one (abridged) to `pkg/agenticmcpe/runs/demo/plan.json`:

```json
{
  "steps": [
    {"id": "s0", "tool": "get_me", "arguments": {}},
    {"id": "s1", "tool": "create_repository",
     "arguments": {"name": "mca-demo", "autoInit": false}},
    {"id": "s2", "tool": "create_or_update_file",
     "arguments": {"owner": "$s0.login", "repo": "mca-demo", "branch": "main",
                   "path": "README.md", "content": "# MCA demo",
                   "message": "Add README"}}
  ]
}
```

`$s0.login` is a binding: the executor fills it in with the `login` field of step `s0`'s result.

| Option | Effect |
|---|---|
| `--run-id ID` | Write the run to `pkg/agenticmcpe/runs/ID/` (default: `run-<timestamp>`) |
| `--provider NAME` | LLM provider: `deepseek` (default), `openai`, `anthropic`, `gemini`, `glm`, `minimax`, `qwen`, `kimi`, `grok`, `xiaomi`, or `custom` |
| `--max-replans N` | Replan budget (default 2) |
| `--no-rag` | Plan without precedents from the knowledge base |
| `--no-verify` | Skip verification |

To choose the model, set `AGENTICMCPE_LLM_MODEL` in `.env`. `.env.example` lists each provider's key variable.

### Run the stages one at a time

```bash
pkg/venv/bin/python -m pkg.agenticmcpe plan   --run-id demo --task "..."
pkg/venv/bin/python -m pkg.agenticmcpe exec   --run-id demo --plan pkg/agenticmcpe/runs/demo/plan.json
pkg/venv/bin/python -m pkg.agenticmcpe verify --run-id demo --plan pkg/agenticmcpe/runs/demo/plan.json \
    --trace pkg/agenticmcpe/runs/demo/trace.json
```

`exec` never calls a model. `verify --no-llm` skips the model-written dynamic checks and runs only the deterministic ones.

### Inspect a run

Each run writes a self-contained directory, `pkg/agenticmcpe/runs/<run-id>/`:

| File | Contents |
|---|---|
| `plan.json`, `plan.md` | The plan (the final attempt, if the run replanned) |
| `trace.json`, `trace.jsonl` | Each step's resolved arguments, attempts, result, and error |
| `server.log`, `execution.log` | The MCP server's raw JSON-RPC log and the executor's log |
| `verify.py`, `verification.json` | The generated verification program and its per-check results |
| `run.json` | Summary: `ok`, replans, each step's status, and checks passed per layer |
| `attempt{i}-*` | Earlier attempts, kept when the run replanned |
| `plan.effective.json` | For a replanned run that succeeded: the successful steps of all attempts, merged into one plan |

### Run the benchmark tasks

```bash
pkg/venv/bin/python pkg/agenticmcpe/run/run_batch.py            # all 28 run/task_NN.txt prompts
pkg/venv/bin/python pkg/agenticmcpe/run/run_batch.py 01 12 30   # a subset
```

Each task writes `runs/task_NN/`, and the results are merged into `runs/batch_summary.{json,md}`, so an interrupted batch can be resumed. Add `--delete-repos` to delete the repositories the batch created once it finishes (the token must be allowed to delete repositories). Set `AGENTICMCPE_RUNS_DIR` to keep each backbone's runs in a separate directory.

### Grow the knowledge base

```bash
pkg/venv/bin/python -m pkg.taskgen generate --count 3 --dry-run   # preview the generated goals; nothing runs
pkg/venv/bin/python -m pkg.taskgen generate --count 3             # read-only tasks
pkg/venv/bin/python -m pkg.taskgen generate --count 3 --mode write --cleanup
pkg/venv/bin/python -m pkg.taskgen stats                          # size and tool coverage
```

Accepted plans are added to `pkg/taskgen/knowledge_base.json`, which the planner reads by default (set `AGENTICMCPE_KB_PATH` to use another file). In write mode, tasks may only create repositories named `agenticmcpe-bench-*`, and `--cleanup` deletes them afterwards.

## Project structure (GitHub MCA, core files)

```
github-mca/github-mcp-server-stable/
├── cmd/github-mcp-server/      # GitHub MCP server entry point (Go)
├── pkg/github/                 # GitHub tool implementations and their tests (Go)
├── pkg/mcp_wrapper/            # stdio MCP client, $sN binding resolution, JSON-Schema validation
│   └── bin/github-mcp-server   #   compiled server, built on first use
├── pkg/agenticmcpe/            # MCA engine
│   ├── planner.py              #   1. Plan: prompt, local validation, replanning
│   ├── rag.py                  #      knowledge-base retrieval and the reuse gate
│   ├── executor.py             #   2. Execute: in-order replay, retries, recovery layer
│   ├── verifier.py             #   3. Verify: writes and runs verify.py
│   ├── tool_sources/           #      each tool's Go source and tests, for the dynamic checks
│   ├── orchestrator.py         #   plan → execute → replan → verify
│   ├── catalog.py              #   live tool catalog (tools/list), condensed for the planner
│   ├── config.py               #   .env loading, LLM providers, GitHub token pool
│   ├── cli.py                  #   python -m pkg.agenticmcpe …
│   ├── run/                    #   benchmark prompts (task_NN.txt) and the batch driver
│   └── runs/                   #   one directory per run
└── pkg/taskgen/                # 4. Learn: task generator and knowledge_base.json
```
